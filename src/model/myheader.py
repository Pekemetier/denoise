import jittor as jt
import jsparse as sp
import numpy as np

from jsparse.nn import functional as F
from jittor import nn
from jsparse import SparseTensor
from jittor import init
from jsparse.utils.quantize import sparse_quantize, set_hash
from typing import Optional, Dict, Tuple, List

# jt.flags.lazy_execution=0
import jsparse.nn as spnn
jt.flags.use_cuda = 1

from .mymodel import OctFormer


class ClsHeader(nn.Module):
  def __init__(self, out_channels: int, in_channels: int,
               nempty: bool = False, dropout: float = 0.5):
    super().__init__()
    self.global_pool = spnn.GlobalPool(op="max")
    self.cls_header = nn.Sequential(
      nn.Linear(in_channels, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(),
      nn.Dropout(p=dropout),
      nn.Linear(256, out_channels)
    )

  def execute(self, data: SparseTensor):
    data = self.global_pool(data)
    logit = self.cls_header(data)
    return logit


class OctFormerCls(nn.Module):

  def __init__(self, in_channels: int, out_channels: int,
               channels: List[int] = [96, 192],
               num_blocks: List[int] = [6, 6],
               num_heads: List[int] = [6, 12],
               patch_size: int = 32, dilation: int = 2,
               drop_path: float = 0.5, nempty: bool = True,
               stem_down: int = 2, head_drop: float = 0.5, **kwargs):
    super().__init__()
    self.backbone = OctFormer(
        in_channels, channels, num_blocks, num_heads, patch_size, dilation,
        drop_path, nempty, stem_down)
    self.head = ClsHeader(
        out_channels, channels[-1], nempty, head_drop)
    self.apply(self.init_weights)

  def init_weights(self, m):
    if isinstance(m, nn.Linear):
      # 截断正态初始化，均值为0，标准差0.02
      nn.init.trunc_normal_(m.weight, std=0.02)
      if m.bias is not None:
          nn.init.constant_(m.bias, 0)

  def execute(self, data: SparseTensor):
    features = self.backbone(data)
    curr_depth = min(features.keys())
    output = self.head(features[curr_depth])
    return output



class DenoiseHeader(nn.Module):

  def __init__(
          self, inchannel: int, out_channel: int, channels: List[int], fpn_channels: List[int]):
    super().__init__()
    self.num_stages = len(channels)

    self.conv1x1 = nn.ModuleList([spnn.Conv3d(
        channels[i], channels[i], kernel_size=1, 
        stride=1) for i in range(self.num_stages-1, -1, -1)])
    self.conv3x3 = nn.ModuleList([spnn.SparseConvBlock(
        channels[self.num_stages-1] if i==0 
        else channels[self.num_stages-i-1] + fpn_channels[i-1],
        fpn_channels[i], kernel_size=3,
        stride=1) for i in range(self.num_stages)])
    self.upsample = nn.ModuleList([spnn.SparseConvTransposeBlock(
        fpn_channels[i], fpn_channels[i], kernel_size=2, # 由于sparsetensor的特性，只能原样负采样。如果需要调整卷积核大小，需要在下采样同时进行
        stride=2) for i in range(self.num_stages-1)])
    self.outconv = spnn.SparseConvBlock(
        fpn_channels[-1] + inchannel, out_channel, kernel_size=3,
        stride=1)

  def execute(self, features: Dict[int, SparseTensor], data: SparseTensor) -> SparseTensor:
    depth = min(features.keys())
    depth_max = max(features.keys())
    assert depth_max - depth + 1 == self.num_stages
    assert self.num_stages == len(features)

    encoder_feature = self.conv1x1[0](features[depth])     # channels[self.num_stages-1]
    feature         = self.conv3x3[0](encoder_feature)     # fpn_channels[0]
    for i in range(1, self.num_stages):
        depth_i = depth + i
        feature = self.upsample[i-1](feature)                # fpn_channels[i-1]
        encoder_feature = self.conv1x1[i](features[depth_i]) # channels[self.num_stages-i-1]
        feature = sp.cat([feature, encoder_feature])         # channels[self.num_stages-i-1] + fpn_channels[i-1]
        feature = self.conv3x3[i](feature)                   # fpn_channels[i]
    
    feature = sp.cat([feature, data])
    feature = self.outconv(feature)

    return feature


class OctFormerDenoise(nn.Module):

    def __init__(
            self, in_channels: int, out_channels: int,
            channels: List[int] = [96, 96, 96],
            num_blocks: List[int] = [4, 4, 4],
            num_heads: List[int] = [6, 6, 6],
            patch_size: int = 32, dilation: int = 4, drop_path: float = 0.5,
            nempty: bool = True, stem_down: int = 2, 
            fpn_channel: List[int] = [96, 96, 96], head_drop: List[float] = [0.0, 0.0, 0.0],
            use_dwconv: bool = True, **kwargs):
        super().__init__()
        self.backbone = OctFormer(
            in_channels, channels, num_blocks, num_heads, patch_size, dilation,
            drop_path, nempty, stem_down, use_dwconv)
        self.head = DenoiseHeader(
            in_channels, out_channels, channels, fpn_channel)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
        # 截断正态初始化，均值为0，标准差0.02
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def execute(self, data: SparseTensor):
        features = self.backbone(data)
        output = self.head(features, data)
        return output
  

def interpolate_voxel_to_point(
    x: SparseTensor,
    points: jt.Var,              # (N, 4) 带 batch 注意 voxelsize=？
    nearest: bool = False        # 是否使用最近邻（仅最近邻体素）
) -> jt.Var:
    """
    对每个点，从其周围8个体素中三线性插值特征。

    Args:
        voxel_tensor: SparseTensor，体素特征，步长 stride。
        point_coords: 点坐标，形状  (N, 4)（带 batch 维度）。
        nearest: 若为 True，则仅使用最近邻体素（等价于最近邻插值）。

    Returns:
        new_values: (N, C) 每个点的插值特征。
    """
    assert points.shape[1] == 4
    offsets = spnn.utils.get_kernel_offsets(kernel_size=2, stride=x.stride, dilation=1)
    cube_hash = F.sphash(
        jt.concat([
            points[:, 0].int().view(-1, 1),
            jt.floor(points[:, 1:] / x.stride[0]).int() * x.stride[0]
        ], 1), offsets)
    indices_hash = F.sphash(x.indices)
    idx_query = F.spquery(cube_hash, indices_hash).int()
    weights = F.calc_ti_weights(points, idx_query,
                        scale=x.stride[0]).t()
    
    idx_query = idx_query.t()
    if nearest:
        weights[:, 1:] = 0.
        idx_query[:, 1:] = -1
    new_values = F.spdevoxelize(x.values, idx_query, weights)


    # print("idx_query -1 ratio:", (idx_query == -1).float().mean().item())
    return new_values


   
   

def get_knn_idx(x, y, k, offset=0):
    """
    x: (B, N, d)
    y: (B, M, d)
    return: (B, N, k)
    """
    K = k + offset
    if x.shape[-1] == 3:
        _, idx = jt.misc.knn(x, y, K)
    else:
        dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
        _, idx = jt.topk(dist, k=K, dim=-1, largest=False)
    return idx[:, :, offset:]



def morton_encode(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    计算三维坐标的 Morton 码 (Z-order curve)。
    输入: x, y, z 为相同形状的 numpy 整数数组。
    输出: 一维 Morton 码数组 (uint64)。
    """
    # 将坐标值限制在 21 位以内 (0 ~ 2^21-1)，以保证 64 位整数不溢出
    # 若实际坐标范围更大，可以增加位数，但通常体素坐标不会太大。
    x = x.astype(np.uint64)
    y = y.astype(np.uint64)
    z = z.astype(np.uint64)

    def spread(v):
        # 交错位：将 21 位扩展到 63 位
        v = (v | (v << 32)) & 0x7fff0000ffffffff
        v = (v | (v << 16)) & 0x0000ffff0000ffff
        v = (v | (v << 8))  & 0x00ff00ff00ff00ff
        v = (v | (v << 4))  & 0x0f0f0f0f0f0f0f0f
        v = (v | (v << 2))  & 0x3333333333333333
        v = (v | (v << 1))  & 0x5555555555555555
        return v

    return (spread(x) << 2) | (spread(y) << 1) | spread(z)

class PCT(nn.Module):
    def __init__(self, num_classes = 40, voxel_size: float = 1.0):
        """
        Args:
            octformer: 已初始化的 OctFormerCls 实例，其 in_channels 应等于特征维度（通常为3）
            voxel_size: 体素大小，用于将连续坐标离散化到体素网格
        """
        super().__init__()
        self.octformer = OctFormerCls(3, num_classes)
        self.voxel_size = voxel_size

    def execute(self, points: jt.Var):
        """
        Args:
            points: (B, 1024, 3)
        Returns:
            output: 分类结果
        """
        B, N, _ = points.shape   # N = 1024

        voxel_coords = (points / self.voxel_size).int()         # (B, N, 3)

        # 展平并构建 batch 索引
        points_flat = points.reshape(-1, 3)          # (B*N, 3)
        voxel_coords_flat = voxel_coords.reshape(-1, 3) # (B*N, 3)
        batch_ids = jt.arange(B).reshape(-1, 1, 1).expand(B, N, 1).reshape(-1, 1)  # (B*N, 1)

        # ---- Morton 排序 ----
        batch_np = batch_ids.numpy().flatten().astype(np.int64)
        x_np = voxel_coords_flat[:, 0].numpy().astype(np.int64)
        y_np = voxel_coords_flat[:, 1].numpy().astype(np.int64)
        z_np = voxel_coords_flat[:, 2].numpy().astype(np.int64)
        morton = morton_encode(x_np, y_np, z_np)
        sorted_indices = np.lexsort((morton, batch_np))   # 形状 (N,)

        
        points_flat = points_flat[sorted_indices]
        voxel_coords_flat = voxel_coords_flat[sorted_indices]


        indices = jt.concat([batch_ids, voxel_coords_flat], dim=1)  # (B*N, 4)
        values = points_flat  # 使用点坐标作为特征，形状 (B*N, 3)

        # sp = SparseTensor(
        #     values=values,
        #     indices=indices,
        #     stride=1,
        #     quantize=True,
        #     coalesce_mode='sample'
        # )
        # print(sp.indices)
        # print(sp.vals)
        # --------- 库有bug，ave模式用不了，需要手动实现 -------
        
        from jsparse.utils.quantize import sparse_quantize, set_hash
        hash_multiplier = set_hash(ndim=3, seed=42)
        indices, mapping, inverse_mapping, count = \
                sparse_quantize(indices, hash_multiplier, self.voxel_size, return_index=True, return_inverse=True, return_count=True)
        out_size = (indices.shape[0], values.shape[-1])
        values = jt.zeros(out_size, dtype=values.dtype).scatter_(0, inverse_mapping, values, reduce='add')
        # print(count.shape)
        # print(values.shape)
        values /= count.reshape(-1,1)
        sp = SparseTensor(
            values=values,
            indices=indices,
            stride=1,
            quantize=True,
            coalesce_mode='sum'
        )
        
        # ------------------------------------------------------


        output = self.octformer(sp, depth=0)
        return output
