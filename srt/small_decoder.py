import numpy as np
import torch
import torch.nn as nn

from srt.layers import  Transformer, FSRTPosEncoder


class FSRTPixelPredictor(nn.Module): 
    def __init__(self, num_att_blocks=2,pix_octaves=16, pix_start_octave=-1, out_dims=3,
                 z_dim=768, input_mlp=True, output_mlp=False, num_kp=10, expression_size=0, kp_octaves=4, kp_start_octave=-1):
        super().__init__()

        self.positional_encoder = FSRTPosEncoder(kp_octaves=kp_octaves,kp_start_octave=kp_start_octave,
                                        pix_octaves=pix_octaves,pix_start_octave=pix_start_octave)
        self.expression_size = expression_size
        self.num_kp = num_kp
        self.feat_dim = pix_octaves*4+num_kp*kp_octaves*4+self.expression_size

        if input_mlp:  # Input MLP added with OSRT improvements
            self.input_mlp = nn.Sequential(
                nn.Linear(self.feat_dim, 720),
                nn.ReLU(),
                nn.Linear(720, self.feat_dim))
        else:
            self.input_mlp = None
        

        self.transformer = Transformer(self.feat_dim, depth=num_att_blocks, heads=6, dim_head=z_dim // 12,
                                       mlp_dim=z_dim, selfatt=False, kv_dim=z_dim)

        if output_mlp:
            self.output_mlp = nn.Sequential(
                nn.Linear(self.feat_dim, 128),
                nn.ReLU(),
                nn.Linear(128, out_dims))
        else:
            self.output_mlp = None

    def forward(self, z, pixels, keypoints, expression_vector=None):
        """
        Args:
            z: set-latent scene repres. [batch_size, num_patches, patch_dim]
            pixels: query pixels [batch_size, num_pixels, 2]
            keypoints: facial query keypoints [batch_size, num_pixels, num_kp, 2]
            expression_vector: latent repres. of the query expression [batch_size, expression_size]
        """
        bs = pixels.shape[0]
        nr = pixels.shape[1]
        nkp = keypoints.shape[-2]
        queries = self.positional_encoder(pixels, keypoints.view(bs,nr,nkp*2))
        
        if expression_vector is not None:
            queries = torch.cat([queries,expression_vector[:,None].repeat(1,queries.shape[1],1)],dim=-1)

        if self.input_mlp is not None:
            queries = self.input_mlp(queries)

        output = self.transformer(queries, z)
        
        if self.output_mlp is not None:
            output = self.output_mlp(output)
            
        return output
    

class ImprovedFSRTDecoder(nn.Module):
    """ Scene Representation Transformer Decoder with the improvements from Appendix A.4 in the OSRT paper."""
    def __init__(self, num_att_blocks=2,pix_octaves=16, pix_start_octave=-1, num_kp=10, kp_octaves=4, kp_start_octave=-1, expression_size=0):
        super().__init__()
        self.allocation_transformer = FSRTPixelPredictor(num_att_blocks=num_att_blocks,
                                                   pix_start_octave=pix_start_octave,
                                                   pix_octaves=pix_octaves,
                                                   z_dim=768,
                                                   input_mlp=True,
                                                   output_mlp=False,
                                                   expression_size=expression_size,
                                                   kp_octaves=kp_octaves,
                                                   kp_start_octave = kp_start_octave
                                                )
        self.expression_size = expression_size 
        self.feat_dim = pix_octaves*4+num_kp*kp_octaves*4+self.expression_size
        self.render_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, 1536),
            nn.ReLU(),
            nn.Linear(1536, 768),
            nn.ReLU(),
            nn.Linear(768, 3),
        )

    def forward(self, z, x, pixels, expression_vector=None):
        x = self.allocation_transformer(z, x, pixels, expression_vector = expression_vector)
        pixels = self.render_mlp(x)
        return pixels, {}

 
