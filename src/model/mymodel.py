import jittor as jt
import jittor.nn as nn
import jsparse as sp
from jsparse import SparseTensor
from jittor import init
from jsparse.utils.quantize import sparse_quantize, set_hash
from typing import Optional, Dict, Tuple, List

# jt.flags.lazy_execution=0
import jsparse.nn as spnn
jt.flags.use_cuda = 1



# 后面所有activation都应用到SparseTensor上，所以需要从nn转到sp的
def spgelu(input: SparseTensor) -> SparseTensor:
    return spnn.utils.fapply(input, nn.gelu)
spGELU = jt.make_module(spgelu)

class spLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)

    def execute(self, x: SparseTensor) -> SparseTensor:
        return spnn.utils.fapply(x, self.layernorm)
   


class MLP(nn.Module):

  def __init__(self, in_features: int, hidden_features: Optional[int] = None,
               out_features: Optional[int] = None, activation=spGELU,
               drop: float = 0.0, **kwargs):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features or in_features
    self.hidden_features = hidden_features or in_features

    self.fc1 = spnn.Linear(self.in_features, self.hidden_features)
    self.act = activation()
    self.fc2 = spnn.Linear(self.hidden_features, self.out_features)
    self.drop = spnn.Dropout(drop)

  def execute(self, data: SparseTensor):
    data = self.fc1(data)
    data = self.act(data)
    data = self.drop(data)
    data = self.fc2(data)
    data = self.drop(data)
    return data




class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, bias: bool = False):
        super().__init__()
        # Depthwise: groups = in_channels，in/out 通道数相等
        self.depthwise = spnn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                     stride=stride, groups=in_channels, bias=bias)
        # Pointwise: 1x1x1 卷积混合通道
        self.pointwise = spnn.Conv3d(in_channels, out_channels, kernel_size=1, 
                                     stride=1, groups=1, bias=bias)

    def execute(self, x: SparseTensor) -> SparseTensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CPE_Jittor(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, 
                 stride: int = 1, use_dwconv: bool = True):
        super().__init__()
        if use_dwconv:
            self.conv = DepthwiseSeparableConv3d(in_channels, in_channels, 
                                                 kernel_size=kernel_size, stride=stride)
        else:
            # 分组卷积： group=8，更接近标准卷积的替代
            group = 8
            assert in_channels % group == 0
            self.conv = spnn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                    stride=stride, groups=group)
        self.bn = spnn.BatchNorm(in_channels)

    def execute(self, x: SparseTensor) -> SparseTensor:
        out = self.conv(x)
        out = self.bn(out)
        return out



class RPE(nn.Module):
    def __init__(self, patch_size: int, num_heads: int, dilation: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.pos_bnd = self.get_pos_bnd(patch_size)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = jt.zeros(3 * self.rpe_num, num_heads)
        init.trunc_normal_(self.rpe_table, std=0.02)

    def get_pos_bnd(self, patch_size: int):
        return int(0.8 * patch_size * self.dilation ** 0.5)

    def xyz2idx(self, xyz: jt.Var):
        mul = jt.arange(3, dtype=jt.int64) * self.rpe_num
        xyz = jt.clamp(xyz, -self.pos_bnd, self.pos_bnd)
        idx = xyz + (self.pos_bnd + mul)
        return idx

    def execute(self, xyz: jt.Var):
        idx = self.xyz2idx(xyz)                     # shape: (N, K, K, 3)
        idx_flat = idx.reshape(-1)                  # (N,)
        out = jt.index_select(self.rpe_table, 0, idx_flat)  # (N, num_heads)
        out = out.view(idx.shape + (-1,))            # (..., 3, num_heads)
        out = out.sum(dim=3)                         # (..., num_heads)
        out = out.permute(0, 3, 1, 2)                # (N, H, K, K)
        return out

    def extra_repr(self) -> str:
        return 'num_heads={}, pos_bnd={}, dilation={}'.format(
                self.num_heads, self.pos_bnd, self.dilation)
    



class dataprocessor(nn.Module):
    def __init__(self, patch_size: int = 24, dilation: int = 4,
                 nempty: bool = True, max_depth: Optional[int] = None,
                 start_depth: Optional[int] = None, **kwargs):
        """
        参数:
            patch_size: 每个 patch 内的节点数
            dilation: 膨胀系数
            nempty: 是否仅处理非空节点（外部输入已筛选）
            max_depth: 最大深度
            start_depth: 起始深度（默认为2）
        关于深度管理：
            原始八叉树需要深度管理，但是在系数卷积中，
            深度只是一个查询键值，没有使用需要
        """
        # 调用父类构造（仅用于满足继承语法，实际不使用）
        super().__init__(max_depth or 1, max_depth or 1)
        self.patch_size = patch_size
        self.dilation = dilation
        self.nempty = nempty
        self.max_depth = max_depth
        self.start_depth = start_depth or 2
        self.invalid_mask_value = -1e3
        self.invalid_batch_id = -1
        self.block_num = patch_size * dilation

        # 按深度存储节点数及填充后的节点数
        self.nnum_t = {}   # depth -> 实际节点数 N
        self.nnum_a = {}   # depth -> 填充后的节点数 (ceil(N/block_num)*block_num)

        # 存储各层数据结构
        self.patch_mask = {}    # attention mask for patches
        self.dilate_mask = {}   # attention mask for dilated groups
        self.rel_pos = {}       # relative positions inside patches
        self.dilate_pos = {}    # relative positions inside dilated groups

    def build(self, indices: jt.Var, depth: int):
        """
        为指定深度构建数据结构。
        
        参数:
            indices: 稀疏张量的坐标，形状 (N, 4)，每行 [batch, x, y, z]
            depth:  当前深度，必须在 [start_depth, max_depth] 范围内
        """
        if not (self.start_depth <= depth <= self.max_depth):
            raise ValueError(f"Depth {depth} out of range [{self.start_depth}, {self.max_depth}]")

        if depth in self.nnum_t:
            return    # 已缓存的深度不需要重新处理

        N = indices.shape[0]
        self.nnum_t[depth] = N
        block_num = self.block_num
        self.nnum_a[depth] = ((N + block_num - 1) // block_num) * block_num   # 大块的个数

        # 提取 batch id 和空间坐标
        batch = indices[:, 0]          # (N,)
        xyz = indices[:, 1:4]          # (N, 3)

        # 填充至长度 nnum_a（batch 填充无效ID，坐标填充0）
        batch_padded = self.patch_partition(batch, depth, fill_value=self.invalid_batch_id)
        xyz_padded = self.patch_partition(xyz, depth, fill_value=0)

        # ---------- patch_mask ----------
        batch_patches = batch_padded.view(-1, self.patch_size)        # (num_patches, patch_size)
        self.patch_mask[depth] = self._calc_attn_mask(batch_patches)  # (num_patches, patch_size, patch_size)

        # ---------- dilate_mask ----------
        # (num_patches, patch_size, dilation) -> (num_patches, dilation, patch_size) -> (num_groups, patch_size)
        batch_reshape = batch_padded.view(-1, self.patch_size, self.dilation)
        batch_reshape = batch_reshape.transpose(1, 2)
        batch_dilate = batch_reshape.reshape(-1, self.patch_size)
        self.dilate_mask[depth] = self._calc_attn_mask(batch_dilate)  # (num_patches, patch_size, patch_size)

        # ---------- rel_pos ----------
        xyz_patches = xyz_padded.view(-1, self.patch_size, 3)         # (num_patches, patch_size, 3)
        rel = xyz_patches.unsqueeze(2) - xyz_patches.unsqueeze(1)     # (num_patches, patch_size, patch_size, 3)
        self.rel_pos[depth] = rel

        # ---------- dilate_pos ----------
        # (num_patches, patch_size, dilation, 3) -> (num_patches, dilation, patch_size, 3) -> (num_groups, patch_size, 3)
        xyz_reshape = xyz_padded.view(-1, self.patch_size, self.dilation, 3)
        xyz_reshape = xyz_reshape.transpose(1, 2)
        xyz_dilate = xyz_reshape.reshape(-1, self.patch_size, 3)
        dilate_rel = xyz_dilate.unsqueeze(2) - xyz_dilate.unsqueeze(1)  # (num_groups, patch_size, patch_size, 3)
        self.dilate_pos[depth] = dilate_rel

    def _calc_attn_mask(self, mask: jt.Var):
        """
        计算注意力掩码：相同组内相同 batch id 的位置设为0，否则设为 invalid_mask_value。
        mask 形状: (num_groups, patch_size)
        返回形状: (num_groups, patch_size, patch_size)
        """
        # 计算差值矩阵，非零表示不同 batch
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(1)   # (num_groups, patch_size, patch_size)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, self.invalid_mask_value)
        return attn_mask

    def patch_partition(self, data: jt.Var, depth: int, fill_value=0):
        """
        将数据填充至 nnum_a[depth] 长度。
        data 形状: (nnum_t[depth], ...)
        返回形状: (nnum_a[depth], ...)
        """
        nnum_t = self.nnum_t[depth]
        nnum_a = self.nnum_a[depth]
        N = data.shape[0]
        if N != nnum_t:
            raise RuntimeError(f"Data length {N} != nnum_t[{depth}]={nnum_t}")
        num_pad = nnum_a - nnum_t
        if num_pad > 0:
            pad_shape = (num_pad,) + data.shape[1:]
            pad = jt.full(pad_shape, fill_value, dtype=data.dtype)
            return jt.cat([data, pad], dim=0)
        return data

    def patch_reverse(self, data: jt.Var, depth: int):
        """
        从填充后的数据中截取有效部分。
        data 形状: (nnum_a[depth], ...)
        返回形状: (nnum_t[depth], ...)
        """
        nnum_t = self.nnum_t[depth]
        return data[:nnum_t]



class spAttention(nn.Module):

    def __init__(self, dim: int, patch_size: int, num_heads: int,
                 qkv_bias: bool = True, qk_scale: float = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 dilation: int = 1, use_rpe: bool = True):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.use_rpe = use_rpe
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.rpe = RPE(patch_size, num_heads, dilation) if use_rpe else None

    def fexecute(self, data: jt.Var, processor: dataprocessor, depth: int):
        H = self.num_heads
        K = self.patch_size
        C = self.dim
        D = self.dilation

        # 1. 分块填充
        data = processor.patch_partition(data, depth)  # (K * D * _ , C)

        if D > 1:   # 膨胀模式
            rel_pos = processor.dilate_pos[depth]      # (N_groups, K, K, 3)
            mask = processor.dilate_mask[depth]        # (N_groups, K, K)
            # 形状变换: (N, K*D, C) -> (N, K, D, C) -> (N, D, K, C) -> (N*D, K, C)
            data = data.view(-1, K, D, C).transpose(1, 2)
        else:
            rel_pos = processor.rel_pos[depth]         # (N_patches, K, K, 3)
            mask = processor.patch_mask[depth]         # (N_patches, K, K)
        data = data.view(-1, K, C)

        # 2. QKV 投影
        qkv = self.qkv(data).reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]            # 各 (N, H, K, head_dim)
        q = q * self.scale

        # 3. 注意力计算
        attn = q @ k.transpose(-2, -1)               # (N, H, K, K)
        attn = self._apply_rpe(attn, rel_pos)        # 加上相对位置偏置
        attn = attn + mask.unsqueeze(1)              # mask 形状 (N, K, K) -> (N, 1, K, K)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(-1, C)  # (N, K, C)

        # 4. 恢复原始形状并反填充
        if D > 1:
            # (N*D, K, C) -> (N, D, K, C) -> (N, K, D, C) -> (N * K * D, C)
            out = out.view(-1, D, K, C).transpose(1, 2).reshape(-1, C)
        out = processor.patch_reverse(out, depth)       # 去掉填充部分

        # 5. FFN 输出投影
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def execute(self, sp: SparseTensor, processor: dataprocessor, depth: int):
        return spnn.utils.fapply(sp, self.fexecute, processor, depth)

    def _apply_rpe(self, attn, rel_pos):
        if self.use_rpe:
            attn = attn + self.rpe(rel_pos)
        return attn

    def extra_repr(self) -> str:
        return f'dim={self.dim}, patch_size={self.patch_size}, num_heads={self.num_heads}, dilation={self.dilation}'
    


class SparseDropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def execute(self, x: SparseTensor) -> SparseTensor:
        if not self.training or self.drop_prob == 0.:
            return x

        keep_prob = 1. - self.drop_prob
        batch_ids = x.indices[:, 0]                     # (N,)
        unique_batch_ids = jt.unique(batch_ids)         # (B,)
        B = len(unique_batch_ids)

        # 生成每个 batch 的掩码（B, 1）
        rnd = jt.rand(B, 1) < keep_prob
        rnd = rnd.float()
        if self.scale_by_keep:
            rnd = rnd / keep_prob   # 注意 keep_prob > 0

        # 将 batch 掩码广播到每个节点：直接索引
        per_point_mask = rnd[batch_ids.long(), 0]       # (N,)
        per_point_mask = per_point_mask.reshape(-1, 1)  # (N, 1)

        return spnn.utils.fapply(x, lambda v: v * per_point_mask)


class FormerBlock(nn.Module):

  def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
               dilation: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True,
               qk_scale: Optional[float] = None, attn_drop: float = 0.0,
               proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
               activation: nn.Module = spGELU, use_dwconv: bool = True,
               **kwargs):
    super().__init__()
    self.norm1 = spLayerNorm(dim)
    self.attention = spAttention(dim, patch_size, num_heads, qkv_bias,
                                     qk_scale, attn_drop, proj_drop, dilation)
    self.norm2 = spLayerNorm(dim)
    self.mlp = MLP(dim, int(dim * mlp_ratio), dim, activation, proj_drop)
    self.drop_path = SparseDropPath(drop_path, nempty)
    self.cpe = CPE_Jittor(dim, use_dwconv=use_dwconv)

  def execute(self, data: SparseTensor, processor: dataprocessor, depth: int):
    data = self.cpe(data) + data
    attn = self.attention(self.norm1(data), processor, depth)
    data = data + self.drop_path(attn)
    ffn = self.mlp(self.norm2(data))
    data = data + self.drop_path(ffn)
    return data


# class CheckpointFunction(jt.Function):
#     def __init__(self, run_function):
#         self.run_function = run_function

#     def execute(self, *inputs):
#         # 前向：只保存输入，不保存中间计算结果
#         self.inputs = inputs
#         self.outputs = self.run_function(*inputs)
#         return self.outputs

#     def grad(self, *grad_outputs):
#         # 反向时，重新执行前向计算，然后求导
#         with jt.enable_grad():
#             outputs = self.run_function(*self.inputs)
#             grads = jt.grad(outputs, self.inputs, grad_outputs)
#         return grads

# def jt_checkpoint(function, *args):
#     return CheckpointFunction(function).apply(*args)

class FormerStage(nn.Module):

  def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
               dilation: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True,
               qk_scale: Optional[float] = None, attn_drop: float = 0.0,
               proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
               activation: nn.Module = spGELU, interval: int = 6,
               use_checkpoint: bool = True, num_blocks: int = 2,
               octformer_block=FormerBlock, use_dwconv: bool = True, **kwargs):
    super().__init__()
    self.num_blocks = num_blocks
    self.use_checkpoint = use_checkpoint
    self.interval = interval  # normalization interval
    self.num_norms = (num_blocks - 1) // self.interval

    self.blocks = nn.ModuleList([octformer_block(
        dim=dim, num_heads=num_heads, patch_size=patch_size,
        dilation=1 if (i % 2 == 0) else dilation,
        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        attn_drop=attn_drop, proj_drop=proj_drop,
        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        nempty=nempty, activation=activation, use_dwconv=use_dwconv,)
        for i in range(num_blocks)])
    # self.norms = torch.nn.ModuleList([
    #     torch.nn.BatchNorm1d(dim) for _ in range(self.num_norms)])

  def execute(self, data: SparseTensor, processor: dataprocessor, depth: int):
    for i in range(self.num_blocks):
    #   if self.use_checkpoint and self.training:
        # data = checkpoint(self.blocks[i], data, processor, depth, use_reentrant=False)
        # data = jt_checkpoint(self.blocks[i], data, processor, depth)
    #   else:
        data = self.blocks[i](data, processor, depth)
      # if i % self.interval == 0 and i != 0:
      #   data = self.norms[(i - 1) // self.interval](data)
    return data



class OldPatchEmbed(nn.Module):

  def __init__(self, in_channels: int = 3, dim: int = 96, num_down: int = 2,
               nempty: bool = True, **kwargs):
    super().__init__()
    self.num_stages = num_down
    channels = [int(dim * 2**i) for i in range(-self.num_stages, 1)]

    self.convs = nn.ModuleList([spnn.SparseConvBlock(
        in_channels if i == 0 else channels[i], channels[i], kernel_size=3,
        stride=1) for i in range(self.num_stages)])
    self.downsamples = nn.ModuleList([spnn.SparseConvBlock(
        channels[i], channels[i+1], kernel_size=2, stride=2)
        for i in range(self.num_stages)])
    self.proj = spnn.SparseConvBlock(
        channels[-1] if self.num_stages != 0 else in_channels, dim, kernel_size=3, stride=1)

  def execute(self, data: SparseTensor):
    for i in range(self.num_stages):
      data = self.convs[i](data)
    #   print(data.indices.shape)
      data = self.downsamples[i](data)
    data = self.proj(data)
    return data
  
class PatchEmbed(nn.Module):

    def __init__(self, in_channels: int = 3, dim: int = 96, num_down: int = 0,
                nempty: bool = True, **kwargs):
        super().__init__()
        assert num_down == 0
        self.num = 2
        self.proj = nn.ModuleList([spnn.SparseConvBlock(
            in_channels if i == 0 else dim, dim, kernel_size=3, stride=1)
            for i in range(self.num)])

    def execute(self, data: SparseTensor):
        for i in range(self.num):
            data = self.proj[i](data)
        return data
  


class Downsample(nn.Module):
  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: int = 2, nempty: bool = True):
    super().__init__()
    self.norm = spnn.BatchNorm(out_channels)
    self.conv = spnn.Conv3d(in_channels, out_channels, kernel_size,
                                   stride=2, bias=True)

  def execute(self, data: SparseTensor):
    data = self.conv(data)
    data = self.norm(data)
    return data
  


class OctFormer(nn.Module):

  def __init__(self, in_channels: int,
               channels: List[int] = [96, 192, 384, 384],
               num_blocks: List[int] = [2, 2, 18, 2],
               num_heads: List[int] = [6, 12, 24, 24],
               patch_size: int = 26, dilation: int = 4, drop_path: float = 0.5,
               nempty: bool = True, stem_down: int = 2, use_dwconv: bool = True,
                use_patch_embd: bool = False,
               **kwargs):
    super().__init__()
    self.patch_size = patch_size
    self.dilation = dilation
    self.nempty = nempty
    self.num_stages = len(num_blocks)
    self.stem_down = stem_down
    drop_ratio = jt.linspace(0, drop_path, sum(num_blocks)).tolist()

    self.patch_embed = None
    if use_patch_embd:
        self.patch_embed = PatchEmbed(in_channels, channels[0], stem_down, nempty)
    self.layers = nn.ModuleList([FormerStage(
        dim=channels[i], num_heads=num_heads[i], patch_size=patch_size,
        drop_path=drop_ratio[sum(num_blocks[:i]):sum(num_blocks[:i+1])],
        dilation=dilation, nempty=nempty, num_blocks=num_blocks[i],
        use_dwconv=use_dwconv,) for i in range(self.num_stages)])
    self.downsamples = nn.ModuleList([Downsample(
        channels[i], channels[i + 1], kernel_size=2,
        nempty=nempty) for i in range(self.num_stages - 1)])

  def execute(self, data: SparseTensor):
    depth = 0 # 初始深度
    if self.patch_embed != None:
        data = self.patch_embed(data)
    processor = dataprocessor(self.patch_size, self.dilation, self.nempty,
                     max_depth=depth, start_depth=depth-self.num_stages+1)
    features = {}
    for i in range(self.num_stages):
      depth_i = depth - i
      processor.build(data.indices, depth_i) # 不在下采样之后，而是直接在attn之前建立，更加可靠
      data = self.layers[i](data, processor, depth_i)
      features[depth_i] = data
    #   print(data.indices.shape)
      if i < self.num_stages - 1:
        data = self.downsamples[i](data)
    return features
