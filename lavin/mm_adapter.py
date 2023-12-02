
import torch
from torch import nn
import lavin
from typing import Optional
from  torch.cuda.amp import autocast
import lavin.eval_model

class RepAdapter_Router(nn.Module):
    r"""Implementation of a Mixture-of-Modality Adapter (MMA) with Dynamic Routing.
    
    This class defines a Mixture-of-Modality Adapter with a dynamic routing mechanism. 
    It is designed to adapt input features based on a modality token and dynamically 
    adjust the adaptations using routing weights as follows:
            router(fa1(Z), fa2(Z)) = ŵ0·fa1(Z) + ŵ1·fa2(Z),
    Here, fa1 and fa2 are RepAdapters. Z can be the single- or multi-modal features. 
    The downsampling projection of two adapters are shared. The routing weights are 
    generated by expert weights computed from the input features and a softmax 
    function as follows:
            ŵ = fw(tm) = softmax((tm Wm + bm)/τ)
    ŵ denotes the routing weights, and τ is the temperature of the softmax.
    
    Visual adapter (here, fa1 and fa2) is often a lightweight neural network
    with a bottleneck structure which can be formulated by
            f(X; θ) = X + φ_u(φ_d(X))
    Here, φ_d and φ_u denote the downsampling and upsampling projections, respectively.
    
    Args:
        in_features (int): Number of input features in the tensor.
        hidden_dim (int): Dimension of hidden representations in the convolutional layers.
        groups (int): Number of groups for the convolutional layers involved in dynamic routing.
        scale (float): Scaling factor applied to the routed features.
        t (float): Temperature parameter for the softmax function used to compute expert weights.
    
    Attributes:
        conv_A (nn.Conv1d): 1D convolutional layer for downsampling projections.
        conv_B (nn.Conv1d): 1D convolutional layer for upsampling projections.
        conv_D (nn.Conv1d): 1D convolutional layer for upsampling projections.
        expert_weights (nn.Linear): Linear layer to generate expert weights.
        dropout (nn.Dropout): Dropout layer for regularization.
        groups (int): Number of groups for dynamic routing convolutional layers.
        scale (float): Scaling factor for the routed features.
        t (float): Temperature parameter for expert weights computation.
    
    Methods:
        forward(x, weights=None): Forward pass through the modality modulator with dynamic routing.
    
    Returns:
        torch.Tensor: Output tensor after applying the dynamic modality adaptation.
    """

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=2,
    ):
        super().__init__()
        self.conv_A = nn.Conv1d(in_features,hidden_dim, 1, groups=1, bias=True) # Down-scale conv
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True) # Up-scale conv 1
        self.groups = groups

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

    def forward(self, x, batch_transpose=False):
        r"""Perform a forward pass through the modality modulator with dynamic routing.
        
        Args:
            x (torch.Tensor): Input tensor to the modality modulator.
            weights (torch.Tensor, optional): Routing weights for dynamic adaptation. If not provided,
                expert weights will be computed based on the input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying dynamic modality adaptation and routing.
        """
        with autocast():            
            if batch_transpose:
                x = x.transpose(0, 1)
            x = x.transpose(1,2)
            x = self.conv_B(self.conv_A(x)) + x
            x = x.transpose(1,2).contiguous()
            if batch_transpose:
                x = x.transpose(0, 1).contiguous()
        return x

def forward_llama_attn(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    if self.training and self.gradient_checkpointing:
        h = x + self.drop_path(torch.utils.checkpoint.checkpoint(self.attention, self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask))
        out = h + self.drop_path(torch.utils.checkpoint.checkpoint(self.feed_forward, self.ffn_norm(h)))
    else:
        h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x)), start_pos, freqs_cis, mask, adapter))
        out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
    return out

def forward_llama_attn_cache(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    if start_pos==0:
        # Use expert_weights/cache_weights only once during the chat/generation. (Or have the zero in self.attention_norm(x)[:,0] will give problem as the sentence now is starting from start_pos instead of 0)
        self.cache_weights=torch.sigmoid(self.adapter_attn.expert_weights(self.attention_norm(x)[:,0])/self.t).half()
    h = x + self.drop_path(self.attention.forward(self.adapter_attn(self.attention_norm(x), weights=self.cache_weights), start_pos, freqs_cis, mask, adapter))
    out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
    return out

def forward_diht(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor]=None):
    x = x + self.attention(self.adapter_attn(self.ln_1(x), batch_transpose=True), attn_mask=attn_mask, key_padding_mask=key_padding_mask)
    x = x + self.mlp(self.ln_2(x))
    return x

def forward_alip(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    x = x + self.ln_attn(self.adapter_attn(self.attention(self.ln_1(x), attn_mask=attn_mask)))
    x = x + self.mlp(self.ln_2(x))
    return x

def forward_clip(self, x: torch.Tensor):
    x = x + self.attention(self.adapter_attn(self.ln_1(x)))
    x = x + self.mlp(self.ln_2(x))
    return x

def set_MMAdapter(model, method, dim=8, s=1, set_forward=True,t=10,gradient_checkpointing=False):
    for _ in model.children():
        if type(_) == lavin.model.TransformerBlock or type(_) == lavin.eval_model.TransformerBlock:
            _.adapter_attn = RepAdapter_Router(_.dim,hidden_dim=dim)
            _.gradient_checkpointing = gradient_checkpointing
            bound_method = forward_llama_attn.__get__(_, _.__class__)
            if set_forward:
                setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_MMAdapter(_, method, dim, s, set_forward=set_forward,t=t,gradient_checkpointing=gradient_checkpointing)


from diht.model import ResidualAttentionBlock
def set_Clip_Adapter(model, method, dim=8, s=1, set_forward=True, t=10.):
    for _ in model.children():
        if type(_) == ResidualAttentionBlock:
            _.adapter_attn = RepAdapter_Router(1024, hidden_dim=dim)
            bound_method = forward_diht.__get__(_, _.__class__)
            if set_forward:
                setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_Clip_Adapter(_, method, dim, s, set_forward=set_forward, t=t)
