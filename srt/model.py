from torch import nn

from srt.encoder import ImprovedFSRTEncoder
from srt.decoder import ImprovedFSRTDecoder
from srt.small_decoder import ImprovedFSRTDecoder as SmallImprovedFSRTDecoder

class FSRT(nn.Module):
    def __init__(self, cfg, expression_encoder=None):
        super().__init__()
            
        self.encoder = ImprovedFSRTEncoder(expression_size=cfg['expression_size'],  **cfg['encoder_kwargs'])
        
        if cfg['small_decoder']:
            self.decoder = SmallImprovedFSRTDecoder(expression_size=cfg['expression_size'], **cfg['decoder_kwargs'])
            print('Loading small decoder')
        else:
            self.decoder = ImprovedFSRTDecoder(expression_size=cfg['expression_size'], **cfg['decoder_kwargs'])
            
        self.expression_encoder = expression_encoder
