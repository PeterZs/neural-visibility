import torch
import numpy as np
import ocnn
from ocnn.octree import Octree
from ocnn.nn import OctreeMaxPool
from ocnn.nn import OctreeConv, OctreeDeconv
from typing import Dict, List
import torch.nn as nn
# from octformerseg import OctFormerSeg


class OctreeConvGnRelu(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
        super().__init__()
        self.conv = OctreeConv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.gn = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels) #, bn_eps, bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.gn(out)
        out = self.relu(out)
        return out

class OctreeConvBnRelu(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
        super().__init__()
        self.conv = OctreeConv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.bn(out)
        out = self.relu(out)
        return out

class OctreeDeconvGnRelu(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
        super().__init__()
        self.deconv = OctreeDeconv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.gn = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels) #, bn_eps, bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.deconv(data, octree, depth)
        out = self.gn(out)
        out = self.relu(out)
        return out

class OctreeDeconvBnRelu(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
        super().__init__()
        self.deconv = OctreeDeconv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.deconv(data, octree, depth)
        out = self.bn(out)
        out = self.relu(out)
        return out

class OctreeGnResBlock2(torch.nn.Module):
  r''' Basic Octree-based ResNet block. The block is composed of
  a series of :obj:`Conv3x3` and :obj:`Conv3x3`.

  Refer to :class:`OctreeResBlock` for the details of arguments.
  '''

  def __init__(self, in_channels, out_channels, stride=1, bottleneck=1,
               nempty=False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stride = stride
    channelb = int(out_channels / bottleneck)

    if self.stride == 2:
      self.maxpool = OctreeMaxPool(self.depth)
    self.conv3x3a = OctreeConvGnRelu(in_channels, channelb, nempty=nempty)
    self.conv3x3b = OctreeConvGn(channelb, out_channels, nempty=nempty)
    if self.in_channels != self.out_channels:
      self.conv1x1 = Conv1x1Gn(in_channels, out_channels)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    if self.stride == 2:
      data = self.maxpool(data, octree, depth)
      depth = depth - 1
    conv1 = self.conv3x3a(data, octree, depth)
    conv2 = self.conv3x3b(conv1, octree, depth)
    if self.in_channels != self.out_channels:
      data = self.conv1x1(data)
    out = self.relu(conv2 + data)
    return out

class OctreeBnResBlock2(torch.nn.Module):
    r''' Basic Octree-based ResNet block. The block is composed of
    a series of :obj:`Conv3x3` and :obj:`Conv3x3`.
    
    Refer to :class:`OctreeResBlock` for the details of arguments.
    '''
    
    def __init__(self, in_channels, out_channels, stride=1, bottleneck=1,
                 nempty=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        channelb = int(out_channels / bottleneck)
    
        if self.stride == 2:
            self.maxpool = OctreeMaxPool(self.depth)
        self.conv3x3a = OctreeConvBnRelu(in_channels, channelb, nempty=nempty)
        self.conv3x3b = OctreeConvBn(channelb, out_channels, nempty=nempty)
        if self.in_channels != self.out_channels:
            self.conv1x1 = Conv1x1Bn(in_channels, out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        r''''''
    
        if self.stride == 2:
            data = self.maxpool(data, octree, depth)
            depth = depth - 1
        conv1 = self.conv3x3a(data, octree, depth)
        conv2 = self.conv3x3b(conv1, octree, depth)
        if self.in_channels != self.out_channels:
            data = self.conv1x1(data)
        out = self.relu(conv2 + data)
        return out

class OctreeConvGn(torch.nn.Module):
  r''' A sequence of :class:`OctreeConv` and :obj:`BatchNorm`.

  Please refer to :class:`ocnn.nn.OctreeConv` for details on the parameters.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.conv = OctreeConv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.gn = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels) #, bn_eps, bn_momentum)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data, octree, depth)
    out = self.gn(out)
    return out

class OctreeConvBn(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
        super().__init__()
        self.conv = OctreeConv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.bn = torch.nn.BatchNorm1d(out_channels)
    
    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.bn(out)
        return out

class Conv1x1Gn(torch.nn.Module):
  r''' A sequence of :class:`Conv1x1` and :class:`BatchNorm`.
  '''

  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.conv = ocnn.modules.Conv1x1(in_channels, out_channels, use_bias=False)
    self.gn = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels)

  def forward(self, data: torch.Tensor):
    r''''''

    out = self.conv(data)
    out = self.gn(out)
    return out

class Conv1x1Bn(torch.nn.Module):
    r''' A sequence of :class:`Conv1x1` and :class:`BatchNorm`.
    '''
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ocnn.modules.Conv1x1(in_channels, out_channels, use_bias=False)
        self.bn = torch.nn.BatchNorm1d(out_channels)
    
    def forward(self, data: torch.Tensor):
        r''''''
    
        out = self.conv(data)
        out = self.bn(out)
        return out

class Conv1x1GnRelu(torch.nn.Module):
  r''' A sequence of :class:`Conv1x1`, :class:`BatchNorm` and :class:`Relu`.
  '''

  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.conv = ocnn.modules.Conv1x1(in_channels, out_channels, use_bias=False)
    self.gn = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor):
    r''''''

    out = self.conv(data)
    out = self.gn(out)
    out = self.relu(out)
    return out

class Conv1x1BnRelu(torch.nn.Module):
    r''' A sequence of :class:`Conv1x1`, :class:`BatchNorm` and :class:`Relu`.
    '''
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ocnn.modules.Conv1x1(in_channels, out_channels, use_bias=False)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, data: torch.Tensor):
        r''''''
    
        out = self.conv(data)
        out = self.bn(out)
        out = self.relu(out)
        return out



# Positional encoding (from NeRF)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    if input_dims == -1:
        return torch.nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class UNet(torch.nn.Module):
    def __init__(self, in_channels: int, interp: str = 'linear', nempty: bool = True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.nempty = nempty
        self.config_network()
        self.encoder_stages = len(self.encoder_blocks)
        self.decoder_stages = len(self.decoder_blocks)

    # encoder
        self.conv1 = OctreeConvBnRelu(in_channels, self.encoder_channel[0], nempty=nempty)
        self.downsample = torch.nn.ModuleList([OctreeConvBnRelu(
            self.encoder_channel[i], self.encoder_channel[i+1], kernel_size=[2],
            stride=2, nempty=nempty) for i in range(self.encoder_stages)])
        self.encoder = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
            self.encoder_channel[i+1], self.encoder_channel[i + 1],
            self.encoder_blocks[i], self.bottleneck, nempty, self.resblk)
            for i in range(self.encoder_stages)])

    # decoder
        channel = [self.decoder_channel[i+1] + self.encoder_channel[-i-2]
                for i in range(self.decoder_stages)]
        self.upsample = torch.nn.ModuleList([OctreeDeconvBnRelu(
            self.decoder_channel[i], self.decoder_channel[i+1], kernel_size=[2],
            stride=2, nempty=nempty) for i in range(self.decoder_stages)])
        self.decoder = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
            channel[i], self.decoder_channel[i+1],
            self.decoder_blocks[i], self.bottleneck, nempty, self.resblk)
            for i in range(self.decoder_stages)])
        self.header = ocnn.modules.Conv1x1(self.decoder_channel[-1], 63, use_bias=True)
        self.octree_interp = ocnn.nn.OctreeInterp(interp, nempty)

    def config_network(self):
        self.encoder_channel = [32, 32, 64, 128, 256]
        self.decoder_channel = [256, 256, 128, 96, 96]
        self.encoder_blocks = [2, 3, 4, 6]
        self.decoder_blocks = [2, 2, 2, 2]
        self.bottleneck = 1
        self.resblk = OctreeBnResBlock2


    def unet_encoder(self, data: torch.Tensor, octree: Octree, depth: int):

        convd = dict()
        convd[depth] = self.conv1(data, octree, depth)
        for i in range(self.encoder_stages):
            d = depth - i
            conv = self.downsample[i](convd[d], octree, d)
            convd[d-1] = self.encoder[i](conv, octree, d-1)
        return convd

    def unet_decoder(self, convd: Dict[int, torch.Tensor], octree: Octree, depth: int):
        deconv = convd[depth]
        for i in range(self.decoder_stages):
            d = depth + i
            deconv = self.upsample[i](deconv, octree, d)
            deconv = torch.cat([convd[d+1], deconv], dim=1)  # skip connections
            deconv = self.decoder[i](deconv, octree, d+1)
        return deconv

    def forward(self, data: torch.Tensor, octree: Octree, depth: int, query_pts: torch.Tensor):
        convd = self.unet_encoder(data, octree, depth)
        deconv = self.unet_decoder(convd, octree, depth - self.encoder_stages)
        deconv = self.header(deconv) # if mul instead of cat
        interp_depth = depth - self.encoder_stages + self.decoder_stages
        feature = self.octree_interp(deconv, octree, interp_depth, query_pts)
        return feature


class VisNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(VisNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp = torch.nn.Sequential(
        Conv1x1BnRelu(self.in_channels, 128),
        #ocnn.modules.Conv1x1BnRelu(128, 128),
        Conv1x1BnRelu(128, 64),
        #ocnn.modules.Conv1x1BnRelu(64, 64),
        ocnn.modules.Conv1x1(64, self.out_channels, use_bias=True)
        )
    def forward(self, feature):
        output = self.mlp(feature)
        return output


class FiLM(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(FiLM, self).__init__()
        
        # 全连接层，用于生成γ和β参数
        self.fc_gamma = nn.Linear(condition_dim, input_dim)
        self.fc_beta = nn.Linear(condition_dim, input_dim)
        
    def forward(self, x, condition):
        # 根据条件特征获取缩放scale参数和移位参数shift，即计算γ和β参数
        gamma = self.fc_gamma(condition)
        beta = self.fc_beta(condition)
        
        # 对输入特征x进行缩放和偏移，实现条件特征调整输入特征
        y = gamma * x + beta 

class MyNet(nn.Module):
    def __init__(self, in_channels: int, vp_channels: int, out_channels: int):
        super(MyNet, self).__init__()
        self.in_channels = in_channels
        self.vp_channels = vp_channels
        self.out_channels = out_channels
        self.UNet = UNet(in_channels=self.in_channels)
        # self.UNet = OctFormerSeg(in_channels=self.in_channels, out_channels=63)
        # self.film = FiLM(63, vp_channels)
        self.VisNet = VisNet(vp_channels, out_channels)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int, query_pts: torch.Tensor, viewdirs: torch.Tensor):
        feature = self.UNet(data, octree, depth, query_pts)
        # feature = torch.cat([feature, viewdirs], dim=1)
        num_v = viewdirs.shape[0] // feature.shape[0]
        feature = feature.unsqueeze(1).repeat(1, num_v, 1).reshape(-1, 63)
        feature = feature * viewdirs
        # feature = torch.cat([feature, viewdirs], dim=1)
        output = self.VisNet(feature)
        return output.view(-1,2)