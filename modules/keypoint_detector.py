from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm2d 
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid

class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs



class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        cnt = 0
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
            cnt+=1
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))




class KPDetector(nn.Module):
    """
    Detecting keypoints. Return keypoint positions.
    """

    def __init__(self, block_expansion=32, num_kp=10, num_channels=3, max_features=1024,
                 num_blocks=5, temperature=0.1, scale_factor=0.25, pad=0):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)
        
        self.num_kp = num_kp
        
        # We do not need the Jacobian (from FOMM).
        #if estimate_jacobian:
            #self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            #self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      #out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            #self.jacobian.weight.data.zero_()
            #self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        #else:
            #self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)



    def gaussian2kp(self, heatmap):
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3))
        return value

 
        
    def forward(self, x):
        with torch.no_grad():
            if self.scale_factor != 1:
                x = self.down(x)

            feature_map = self.predictor(x)
            prediction = self.kp(feature_map)

            final_shape = prediction.shape
            heatmap = prediction.view(final_shape[0], final_shape[1], -1)
            heatmap = F.softmax(heatmap / self.temperature, dim=2)
            heatmap = heatmap.view(*final_shape)

            out = self.gaussian2kp(heatmap)
            heatmap = heatmap.unsqueeze(2)
            
            # We do not need the Jacobian (from FOMM).
            #if self.jacobian is not None:
                #jacobian_map = self.jacobian(feature_map)
                #jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                    #final_shape[3])
                #jacobian = heatmap * jacobian_map
                #jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
                #jacobian = jacobian.sum(dim=-1)
                #jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)

            return out ,{'feature_map': feature_map, 'heatmap': heatmap}
    


        
 
