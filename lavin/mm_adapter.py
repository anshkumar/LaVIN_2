import torch
from torch import nn
import lavin
from typing import Optional
from  torch.cuda.amp import autocast
from diht.model import ResidualAttentionBlock

class RepAdapter_Router(nn.Module):
    r"""Implementation of a Adapter
    
    Args:
        in_features (int): Number of input features in the tensor.
        hidden_dim (int): Dimension of hidden representations in the convolutional layers.
        groups (int): Number of groups for the convolutional layers involved in dynamic routing.
    
    Attributes:
        conv_A (nn.Conv1d): 1D convolutional layer for downsampling projections.
        conv_B (nn.Conv1d): 1D convolutional layer for upsampling projections.
        groups (int): Number of groups for dynamic routing convolutional layers.
    
    Methods:
        forward(x): Forward pass through the modality modulator with dynamic routing.
    
    Returns:
        torch.Tensor: Output tensor after applying the dynamic modality adaptation.
    """

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=1,
    ):
        super().__init__()
        self.conv_A = nn.Linear(in_features,hidden_dim, bias=True) # Down-scale conv
        self.conv_B = nn.Linear(hidden_dim, in_features, bias=True) # Up-scale conv 1
        self.groups = groups

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

    def forward(self, x):
        r"""Perform a forward pass through the modality modulator with dynamic routing.
        
        Args:
            x (torch.Tensor): Input tensor to the modality modulator.
            weights (torch.Tensor, optional): Routing weights for dynamic adaptation. If not provided,
                expert weights will be computed based on the input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying dynamic modality adaptation and routing.
        """
        with autocast():            
            x = self.conv_B(self.conv_A(x)) + x
        return x

def forward_llama_attn(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    if self.training and self.gradient_checkpointing:
        h = x + self.drop_path(torch.utils.checkpoint.checkpoint(self.attention, self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask))
        out = h + self.drop_path(torch.utils.checkpoint.checkpoint(self.feed_forward, self.ffn_norm(h)))
    else:
        h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask, adapter))
        out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
    return out

def forward_diht(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor]=None):
    x = x + self.attention(self.adapter_attn(self.ln_1(x)), attn_mask=attn_mask, key_padding_mask=key_padding_mask)
    x = x + self.mlp(self.ln_2(x))
    return x

def set_MMAdapter(model, dim=8, gradient_checkpointing=False):
    for _ in model.children():
        if type(_) == lavin.model.TransformerBlock:
            _.adapter_attn = RepAdapter_Router(_.dim, hidden_dim=dim)
            _.gradient_checkpointing = gradient_checkpointing
            bound_method = forward_llama_attn.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_MMAdapter(_, dim, gradient_checkpointing=gradient_checkpointing)


def set_Clip_Adapter(model, dim=8):
    for _ in model.children():
        if type(_) == ResidualAttentionBlock:
            _.adapter_attn = RepAdapter_Router(1024, hidden_dim=dim)
            bound_method = forward_diht.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_Clip_Adapter(_, dim)
