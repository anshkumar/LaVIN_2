import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class LowRank(nn.Module):
  def __init__(self,
               in_channels: int,
               out_channels: int,
               low_rank: int,
               kernel_size: int):
    super().__init__()
    self.T = nn.Parameter(
        torch.empty(size=(low_rank, low_rank, kernel_size)),
        requires_grad=True
    )
    self.O = nn.Parameter(
        torch.empty(size=(low_rank, out_channels)),
        requires_grad=True
    )
    self.I = nn.Parameter(
        torch.empty(size=(low_rank, in_channels)),
        requires_grad=True
    )
    self._init_parameters()
  
  def _init_parameters(self):
    # Initialization affects the convergence stability for our parameterization
    fan = nn.init._calculate_correct_fan(self.T, mode='fan_in')
    gain = nn.init.calculate_gain('relu', 0)
    std_t = gain / np.sqrt(fan)

    fan = nn.init._calculate_correct_fan(self.O, mode='fan_in')
    std_o = gain / np.sqrt(fan)

    fan = nn.init._calculate_correct_fan(self.I, mode='fan_in')
    std_i = gain / np.sqrt(fan)

    nn.init.normal_(self.T, 0, std_t)
    nn.init.normal_(self.O, 0, std_o)
    nn.init.normal_(self.I, 0, std_i)

  def forward(self):
    # torch.einsum simplify the tensor produce (matrix multiplication)
    return torch.einsum("xyz,xo,yi->oiz", self.T, self.O, self.I)

class Conv1d(nn.Module):
  def __init__(self,
               in_channels: int,
               out_channels: int,
               kernel_size: int=3,
               stride: int=1,
               padding: int=0,
               bias: bool=False,
               ratio: float=0.0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.bias = bias
    self.ratio = ratio
    self.low_rank = 1 #self._calc_from_ratio()

    self.W1 = LowRank(in_channels, out_channels, self.low_rank, kernel_size)
    self.W2 = LowRank(in_channels, out_channels, self.low_rank, kernel_size)
    self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

  def _calc_from_ratio(self):
    # Return the low-rank of sub-matrices given the compression ratio 
    r1 = int(np.ceil(np.sqrt(self.out_channels)))
    r2 = int(np.ceil(np.sqrt(self.in_channels)))
    r = np.max((r1, r2))

    num_target_params = self.out_channels * self.in_channels * \
      (self.kernel_size ** 2) * self.ratio
    r3 = np.sqrt(
        ((self.out_channels + self.in_channels) ** 2) / (4 *(self.kernel_size ** 4)) + \
        num_target_params / (2 * (self.kernel_size ** 2))
    ) - (self.out_channels + self.in_channels) / (2 * (self.kernel_size ** 2))
    r3 = int(np.ceil(r3))
    r = np.max((r, r3))

    return r

  def forward(self, x):
    # Hadamard product of two submatrices
    W = self.W1() * self.W2()
    out = F.conv1d(input=x, weight=W, bias=self.bias,
                 stride=self.stride, padding=self.padding)
    return out
