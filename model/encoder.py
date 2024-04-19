import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from layers import dilatedconv
from parameters import *

def generate_continuous_mask(B, T, n=5, l=0.1):
    """
    Generate a continuous mask for the input sequence.

    Args:
        B (int): Batch size.
        T (int): Sequence length.
        n (int or float): Number of continuous segments to mask. If float, it represents the ratio of the sequence length.
        l (int or float): Length of each continuous segment to mask. If float, it represents the ratio of the sequence length.

    Returns:
        torch.Tensor: Continuous mask of shape (B, T) with False values indicating masked segments.
    """
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    """
    Generate a binomial mask for the input sequence.

    Args:
        B (int): Batch size.
        T (int): Sequence length.
        p (float): Probability of each element being masked.

    Returns:
        torch.Tensor: Binomial mask of shape (B, T) with False values indicating masked elements.
    """
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class TemporalPyramidPooling(nn.Module):
    def __init__(self, in_channels, pyramid_levels=[1, 2, 4, 8]):
        """
        Temporal Pyramid Pooling module.

        Args:
            in_channels (int): Number of input channels.
            pyramid_levels (list): List of pyramid levels for pooling.

        """
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool1d(level) for level in pyramid_levels])

    def forward(self, x):
        """
        Forward pass of the Temporal Pyramid Pooling module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T), where B is the batch size, C is the number of channels, and T is the sequence length.

        Returns:
            torch.Tensor: Output tensor after applying temporal pyramid pooling of shape (B, C', T), where C' is the concatenated number of channels after pooling.
        """
        features = [x]
        for pool in self.pools:
            features.append(pool(x))
        return torch.cat([F.interpolate(feature, size=x.size(2), mode='nearest') for feature in features], dim=1)


class Encoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'): 
        """
        Encoder module.

        Args:
            input_dims (int): Number of input dimensions.
            output_dims (int): Number of output dimensions.
            hidden_dims (int): Number of hidden dimensions.
            depth (int): Depth of the dilated convolutional layers.
            mask_mode (str): Masking mode. Can be one of 'binomial', 'continuous', 'all_true', 'all_false', or 'mask_last'.
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = dilatedconv(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=[3, 5, 7] + [3] * (depth-2)
        )
        self.tpp = TemporalPyramidPooling(output_dims)
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):  
        """
        Forward pass of the Encoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T), where B is the batch size, C is the number of channels, and T is the sequence length.
            mask (str or torch.Tensor): Masking mode or custom mask tensor. If None, the mask is determined based on the mask_mode.

        Returns:
            torch.Tensor: Output tensor after encoding of shape (B, C', T), where C' is the concatenated number of channels after encoding.
        """
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x) 
        
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        x = x.transpose(1, 2)  
        x = self.repr_dropout(self.feature_extractor(x))  
        x_tpp = self.tpp(x)
        x = torch.cat([x, x_tpp], dim=1)
        x = x.transpose(1, 2)  
        
        return x
        