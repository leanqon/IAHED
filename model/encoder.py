import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from dilated_conv import DilatedConvEncoder
from parameters import *

def generate_continuous_mask(B, T, n=5, l=0.1):
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
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class TemporalPyramidPooling(nn.Module):
    def __init__(self, in_channels, pyramid_levels=[1, 2, 4, 8]):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool1d(level) for level in pyramid_levels])

    def forward(self, x):
        features = [x]
        for pool in self.pools:
            features.append(pool(x))
        # Upsample features back to original size and concatenate
        return torch.cat([F.interpolate(feature, size=x.size(2), mode='nearest') for feature in features], dim=1)


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'): #args.output_dim hidden_dims=64 depth=10
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=[3, 5, 7] + [3] * (depth-2)#3
        )
        self.tpp = TemporalPyramidPooling(output_dims)
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
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
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x_tpp = self.tpp(x)
        x = torch.cat([x, x_tpp], dim=1)
        x = x.transpose(1, 2)  # B x T x Co
        
        return x
        