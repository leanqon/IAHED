import torch
from torch import nn
import torch.nn.functional as F

def multi_scale_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    """
    Calculates the multi-scale contrastive loss between two input tensors z1 and z2.
    
    Args:
        z1 (torch.Tensor): The first input tensor of shape (B, T, C), where B is the batch size, 
                           T is the temporal dimension, and C is the number of channels.
        z2 (torch.Tensor): The second input tensor of shape (B, T, C).
        alpha (float, optional): The weight for the instance contrastive loss. Default is 0.5.
        temporal_unit (int, optional): The number of temporal units to skip before applying the 
                                       temporal contrastive loss. Default is 0.
    
    Returns:
        torch.Tensor: The calculated multi-scale contrastive loss.
    """
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2):
    """
    Calculates the instance contrastive loss between two input tensors z1 and z2.
    
    Args:
        z1 (torch.Tensor): The first input tensor of shape (B, T, C), where B is the batch size, 
                           T is the temporal dimension, and C is the number of channels.
        z2 (torch.Tensor): The second input tensor of shape (B, T, C).
    
    Returns:
        torch.Tensor: The calculated instance contrastive loss.
    """
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  
    z = z.transpose(0, 1) 
    sim = torch.matmul(z, z.transpose(1, 2)) 
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    """
    Calculates the temporal contrastive loss between two input tensors z1 and z2.
    
    Args:
        z1 (torch.Tensor): The first input tensor of shape (B, T, C), where B is the batch size, 
                           T is the temporal dimension, and C is the number of channels.
        z2 (torch.Tensor): The second input tensor of shape (B, T, C).
    
    Returns:
        torch.Tensor: The calculated temporal contrastive loss.
    """
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  
    sim = torch.matmul(z, z.transpose(1, 2)) 
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]   
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss
