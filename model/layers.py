import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        """
        SamePadConv is a custom convolutional layer that applies padding to ensure the output has the same length as the input.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            dilation (int, optional): Dilation rate for the convolutional kernel. Defaults to 1.
            groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
        """
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        """
        Forward pass of the SamePadConv layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, sequence_length).
        """
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        """
        ConvBlock is a building block for a dilated convolutional neural network.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            dilation (int): Dilation rate for the convolutional kernel.
            final (bool, optional): Whether this block is the final block in the network. Defaults to False.
        """
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        """
        Forward pass of the ConvBlock.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, sequence_length).
        """
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class dilatedconv(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        """
        dilatedconv is a dilated convolutional neural network.
        
        Args:
            in_channels (int): Number of input channels.
            channels (list): List of integers representing the number of channels in each ConvBlock.
            kernel_size (list): List of integers representing the size of the convolutional kernel in each ConvBlock.
        """
        super().__init__()
        assert len(channels) == len(kernel_size)
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size[i],
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        """
        Forward pass of the dilatedconv network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, sequence_length).
        """
        return self.net(x)
