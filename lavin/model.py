# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Any, Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)

from torch.nn import Embedding, Linear
import torch
# import pdb
from timm.models.layers import  DropPath
# import clip
from diht import model_zoo
# from open_alip import create_model
from  torch.cuda.amp import autocast

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    hidden_proj: int=128

    max_batch_size: int = 32
    max_seq_len: int = 2048
    drop_path: float=0.

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    This function precomputes the frequency and returns it in complex form.
    Eq 34 in https://arxiv.org/abs/2104.09864 

    Parameters:
    dim (int): The dimension of the frequency, typically the dimension divided by the number of heads in a transformer model.
    end (int): The maximum sequence length multiplied by 2.
    theta (float, optional): A pre-defined parameter used in the frequency computation. Default is 10000.0.

    Returns:
    torch.Tensor: A tensor of complex numbers representing the precomputed frequencies.

    Note:
    The function freqs by θ_i = 10000^{-2(i-1)/d}, i ∈ [1, 2, ..., d/2], which can be re-written as:
    1/10000^{2(i-1)/d}. The equivalent code for 2(i-1)/d, i ∈ [1, 2, ..., d/2] is np.arange(0, d, 2)[: (d // 2)]. 
    This code generates a sequence of numbers starting from 0 and incrementing by 2 up to d, excluding d, and then selects 
    the first d // 2 elements from the sequence. Now, for various positions m, let's have m = torch.arange(seq_len, device=freqs.device).
    Getting each combination of m and θ, we will  use outer product and finally convert to polar form to get sin and cos terms.
    """
    
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    '''
    Convert freqs_cis sahpe from (seq length, head dim) to (1, seq length, 1, head dim)
    '''
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Eq 34 in https://arxiv.org/abs/2104.09864 
    Convert xq and xk into complex no as follows:
        xq, xk have shape of (batch size, seq length, local heads, head dim).
        Now reshape the last dim (head_dim) into (-1, 2). One for real part and another for complex part 
        of the complex no. So, now the new shape is (batch size, seq length, local heads, head_dim/2, 2).
    '''
    
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        #modified bias for reparameterizing
        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        # self.cache_k = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # ).cuda()
        # self.cache_v = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # ).cuda()
        # self.gate = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv


        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of, ffn_dim_multiplier=args.ffn_dim_multiplier
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.drop_path = DropPath(args.drop_path) if args.drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
        h = x + self.drop_path(self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter))
        out = h + self.drop_path(self.feed_forward.forward(self.ffn_norm(h)))
        return out

class AdapterMLP(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=128,
            out_features=4096
    ):
        super().__init__()
        self.conv_A=nn.Linear(in_features,hidden_dim)
        self.conv_B = nn.Linear(hidden_dim, out_features)


        nn.init.xavier_uniform_( self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.xavier_uniform_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

    def forward(self, x):
        with autocast():
            x=self.conv_B(F.silu(self.conv_A(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        # with init_empty_weights():
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        ########################################################
        # self.backbone = clip.load('ViT-L/14')[0] 
        _, _, self.backbone = model_zoo.load_model("diht_vitl14_336px", is_train=False)
        self.backbone = self.backbone.to(torch.device("cuda"))
        self.adapter_proj = AdapterMLP(1024, params.hidden_proj, params.dim).float() # (512, 128, 4096)
        self.adapter_modality_embedding=nn.Embedding(2, params.dim).float()

    def insert_image_embeds(self, examples, labels, image_embeds, prefix_img, prefix_nonimg, img_indicators):
        r"""Insert image embeddings into the input sequences and adjust labels accordingly.

        Args:
            examples (torch.Tensor): Batch of tokenized input sequences.
            labels (torch.Tensor): Batch of tokenized label sequences.
            image_embeds (torch.Tensor): Batch of image embeddings to insert.
            prefix_img (torch.Tensor): Prefix tensor for image insertion in examples. Ex: "Image: "
            prefix_nonimg (torch.Tensor): Prefix tensor for non-image insertion in examples. Ex: "Image: N/A"
            img_indicators (torch.Tensor): Batch of indicators (1 or 0) for image availability.

        Returns:
            new_examples (torch.Tensor): Updated batch of input sequences with inserted image embeddings.
            new_labels (torch.Tensor): Updated batch of label sequences.
        """
        _bsz, seqlen,_ = examples.shape
        new_examples=[]
        new_labels=[]
        for i, (example,label) in enumerate(zip(examples,labels)):
            if img_indicators[i]>0.:
                new_example=torch.cat([example[:1],prefix_img,image_embeds[i],example[1:]],0)
                new_label=torch.cat([label[:1],
                                     torch.zeros(prefix_img.shape[0]+image_embeds.shape[1]).to(examples.device).type_as(labels),
                                     label[1:]])
                new_example = new_example[:seqlen]
                new_label = new_label[:seqlen]
            else:
                new_example=torch.cat([example[:1],prefix_nonimg,example[1:]],0)
                new_label=torch.cat([label[:1],
                                     torch.zeros(prefix_nonimg.shape[0]).to(examples.device).type_as(labels),
                                     label[1:]])
                new_example = new_example[:seqlen]
                new_label = new_label[:seqlen]
            new_examples.append(new_example.unsqueeze(0))
            new_labels.append(new_label.unsqueeze(0))
        new_examples = torch.cat(new_examples, 0)
        new_labels = torch.cat(new_labels, 0)
        return new_examples,new_labels

    def forward(self, examples, labels,images=None, prefix_img=None, prefix_nonimg=None,img_indicators=None):
        image_embeds = self.backbone.encode_image(images.half()).half() # [1, 512]
        # image_embeds = image_embeds.unsqueeze(1)

        if isinstance(img_indicators,list):
            img_indicators = torch.Tensor(img_indicators).to(image_embeds.device).long() # [1]
        modality_embed=self.adapter_modality_embedding(img_indicators.unsqueeze(1)) # [1, 1, 4096]

        image_embeds=self.adapter_proj(image_embeds) # [1, 6, 4096]

        _bsz, seqlen = examples.shape # [1, 128] (batch size, sequence length)

        examples = self.tok_embeddings(examples) # [1, 128, 4096] (batch size, sequence length, embedding dim)
        prefix_img=self.tok_embeddings(prefix_img.unsqueeze(0)).squeeze(0) # [3, 4096]
        prefix_nonimg=self.tok_embeddings(prefix_nonimg.unsqueeze(0)).squeeze(0) # [5, 4096]

        h, labels=self.insert_image_embeds(examples, labels, image_embeds, prefix_img, prefix_nonimg, img_indicators)

        h=torch.cat([modality_embed.half(),h],1)[:,:seqlen]
        modality_labels=torch.zeros(_bsz,1).to(labels.device).type_as(labels)
        labels=torch.cat([modality_labels,labels],1)[:,:seqlen]
        
        freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        # mask decision token
        mask[:,:,1:,0]=float("-inf")

        start_pos = 0
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h)
        output = output[:, :-1, :].reshape(-1, self.vocab_size)
        labels = labels[:, 1:].flatten()

        c_loss = self.criterion(output, labels)
        return c_loss
