from torch import nn
import torch
import torch.nn.functional as F

class ExpressionEncoder(nn.Module):
    """
    Extracts the latent expression vector.
    """

    def __init__(self, in_channels = 32, num_kp=10, expression_size_per_kp=32, expression_size=256, pad=0):
        super(ExpressionEncoder, self).__init__()

        self.expression_size = expression_size #Output dimension
        self.expression_size_per_kp = expression_size_per_kp #Number of output features of the convolutional layer for each keypoint 
        self.num_kp = num_kp
        self.expression = nn.Conv2d(in_channels=in_channels,
                                      out_channels=num_kp * self.expression_size_per_kp , kernel_size=(7, 7), padding=pad)
        self.expression_mlp = nn.Sequential(
            nn.Linear(self.expression_size_per_kp*self.num_kp, 640),
            nn.ReLU(),
            nn.Linear(640, 1280),
            nn.ReLU(),
            nn.Linear(1280, 640),
            nn.ReLU(),
            nn.Linear(640, self.expression_size),
        )

 
    def forward(self, feature_map, heatmap):
        
        latent_expression_feat = self.expression(feature_map)
        final_shape = latent_expression_feat.shape     
        latent_expression_feat = latent_expression_feat.reshape(final_shape[0], self.num_kp, self.expression_size_per_kp , final_shape[2], final_shape[3])
        
        heatmap = heatmap.unsqueeze(2)
        latent_expression = heatmap * latent_expression_feat
        latent_expression = latent_expression.view(final_shape[0], self.num_kp, self.expression_size_per_kp , -1)
        latent_expression = latent_expression.sum(dim=-1).view(final_shape[0],-1)
        latent_expression = self.expression_mlp(latent_expression)
        
        return latent_expression
         
