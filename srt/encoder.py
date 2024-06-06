import numpy as np
import torch
import torch.nn as nn
import math

from srt.layers import Transformer, FSRTPosEncoder


class SRTConvBlock(nn.Module):
    def __init__(self, idim, hdim=None, odim=None):
        super().__init__()
        if hdim is None:
            hdim = idim

        if odim is None:
            odim = 2 * hdim

        conv_kwargs = {'bias': False, 'kernel_size': 3, 'padding': 1}
        self.layers = nn.Sequential(
            nn.Conv2d(idim, hdim, stride=1, **conv_kwargs),
            nn.ReLU(),
            nn.Conv2d(hdim, odim, stride=2, **conv_kwargs),
            nn.ReLU())

    def forward(self, x):
        return self.layers(x)

    
class ImprovedFSRTEncoder(nn.Module):
    """
    Scene Representation Transformer Encoder with the improvements from Appendix A.4 in the OSRT paper.
    """
    def __init__(self, num_conv_blocks=3, num_att_blocks=5, pix_octaves=16, pix_start_octave=-1, num_kp = 10, expression_size=256, encode_with_expression=True, kp_octaves=4, kp_start_octave=-1):
        super().__init__()
        self.positional_encoder = FSRTPosEncoder(kp_octaves=kp_octaves,kp_start_octave=kp_start_octave,
                                        pix_octaves=pix_octaves,pix_start_octave=pix_start_octave)

        self.encode_with_expression = encode_with_expression
        if self.encode_with_expression:
            self.expression_size = expression_size
        else:
            self.expression_size=0
        conv_blocks = [SRTConvBlock(idim=3+pix_octaves*4+num_kp*kp_octaves*4+self.expression_size, hdim=96)]
        cur_hdim = 192
        for i in range(1, num_conv_blocks):
            conv_blocks.append(SRTConvBlock(idim=cur_hdim, odim=None))
            cur_hdim *= 2

        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.per_patch_linear = nn.Conv2d(cur_hdim, 768, kernel_size=1)

        self.transformer = Transformer(768, depth=num_att_blocks, heads=12, dim_head=64,
                                       mlp_dim=1536, selfatt=True) 
        self.num_kp = num_kp

    def forward(self, images, keypoints, pixels, expression_vector=None):
        """
        Args:
            images: [batch_size, num_images, 3, height, width]
            keypoints: [batch_size, num_images, num_kp, 2]
            pixels: [batch_size, num_images, height, width, 2]
            expression_vector: [batch_size, num_images, expression_size]
        Returns:
            scene representation: [batch_size, num_patches, channels_per_patch]
        """

        batch_size, num_images = images.shape[:2]

        x = images.flatten(0, 1)
        keypoints = keypoints.flatten(-2,-1).flatten(0,1)
        pixels = pixels.flatten(0, 1)

        pos_enc = self.positional_encoder(pixels,keypoints)
        if expression_vector is not None and self.encode_with_expression:
            expression_vector = expression_vector.flatten(0,1)[:,:,None,None].repeat(1,1,images.shape[-2],images.shape[-1])
            x = torch.cat([x,pos_enc,expression_vector], 1)
        else:
            x = torch.cat([x,pos_enc], 1)
        x = self.conv_blocks(x)
        x = self.per_patch_linear(x)
        x = x.flatten(2, 3).permute(0, 2, 1)

        patches_per_image, channels_per_patch = x.shape[1:]
        x = x.reshape(batch_size, num_images * patches_per_image, channels_per_patch)

        x = self.transformer(x)

        return x
    


