from math import ceil
from typing import Dict, List, Union

import jittor as jt
import numpy as np
from jittor import nn
from jsparse import SparseTensor
from jsparse.utils.quantize import sparse_quantize, set_hash
import jittor.loss3d as loss3d

from .feature import FeatureExtraction, Decoder
from .spec import ModelSpec
from .myheader import OctFormerDenoise, interpolate_voxel_to_point
from .decode import ScoreNetLinear
from .encode import VoxelFeatureEncoder, pad_voxel_centers, dense_to_sparse


from ..data.asset import Asset

def get_random_indices(n, m):
    assert m < n
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()


def morton_encode(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  bits=12, offset=0) -> np.ndarray:
    """
    计算三维坐标的 Morton 码，支持负数范围 [-1024, 1024]。
    输入: x, y, z 为相同形状的整数数组 (支持负数)。
    输出: 一维 Morton 码数组 (uint64)。
    """
    x_off = (x + offset).astype(np.uint64)
    y_off = (y + offset).astype(np.uint64)
    z_off = (z + offset).astype(np.uint64)
    bits_arr = np.arange(bits, dtype=np.uint64)
    x_bits = (x_off[:, None] >> bits_arr) & 1
    y_bits = (y_off[:, None] >> bits_arr) & 1
    z_bits = (z_off[:, None] >> bits_arr) & 1
    shifts = 3 * bits_arr
    morton = (x_bits << shifts).sum(axis=1) \
           | (y_bits << (shifts + 1)).sum(axis=1) \
           | (z_bits << (shifts + 2)).sum(axis=1)
    return morton

def morton_sort(voxel_coords_flat: jt.Var, batch_ids: jt.Var):
    # ---- Morton 排序 ----
    batch_np = batch_ids.numpy().flatten().astype(np.int32)
    x_np = voxel_coords_flat[:, 0].numpy()
    y_np = voxel_coords_flat[:, 1].numpy()
    z_np = voxel_coords_flat[:, 2].numpy()
    morton = morton_encode(x_np, y_np, z_np)
    return np.lexsort((morton, batch_np))   # 形状 (N,)



def norm_quantile_report(var, quantiles=(0, 1, 10, 25, 50, 75, 90, 99, 100), 
                         precision=4, print_header=True):
    """
    计算输入张量第一个 batch 中所有向量的模长，并输出指定分位数（无需 NumPy）。

    参数:
        var: jt.Var，形状 (B, N, 3)，至少包含一个 batch
        quantiles: 分位点序列 (默认 0,1,10,25,50,75,90,99,100)
        precision: 浮点数小数位数 (默认 4)
        print_header: 是否打印表头 (第一次调用设为 True，后续 False)
    
    返回:
        分位数列表 (list of floats)
    """
    # 只取第一个 batch，形状 (N, 3)
    if len(var.shape) == 3:
        batch0 = var[0]  # (N, 3)
    else:
        batch0 = var
    
    # 计算每个向量的模长 (L2 范数) -> (N,)
    norms = jt.sqrt((batch0 ** 2).sum(dim=1))
    # 转为 Python 列表（避免 NumPy）
    norms_list = norms.tolist()
    N = len(norms_list)
    
    if N == 0:
        # 空输入处理
        values = [float('nan')] * len(quantiles)
    else:
        norms_sorted = sorted(norms_list)
        values = []
        for q in quantiles:
            # 线性插值方法，与 np.percentile 默认行为一致
            pos = (q / 100.0) * (N - 1)
            i = int(pos)
            frac = pos - i
            if i < 0:
                value = norms_sorted[0]
            elif i >= N - 1:
                value = norms_sorted[-1]
            else:
                value = norms_sorted[i] * (1 - frac) + norms_sorted[i + 1] * frac
            values.append(value)
    
    # 格式化输出：每个数值固定宽度 (precision+6)
    fmt = f"{{:>{precision+6}.{precision}f}}"
    row = " ".join(fmt.format(v) for v in values)
    
    if print_header:
        header_fmt = f"{{:>{precision+6}s}}"
        header = " ".join(header_fmt.format(f"q{q}") for q in quantiles)
        print(header)
    print(row)
    
    return values

def my_sparse_quantize(
    indices: jt.Var,
    *,
    return_index: bool = False,
    return_inverse: bool = False,
    return_count: bool = False,
) -> Union[jt.Var, List[jt.Var]]:
    """
    对已排序的整数坐标进行去重（体素大小=1）。
    
    参数:
        indices: (N, 3) 或 (N, 4) 的 jt.Var，dtype 为整数，已按莫顿码（或字典序）排序。
        return_index: 是否返回每个唯一组第一个元素在原数组中的索引。
        return_inverse: 是否返回原数组到唯一索引的映射。
        return_count: 是否返回每个唯一组的计数。
    
    返回:
        根据标志返回列表，顺序为 [unique_indices, index, inverse, count]。
    """
    # 输入校验
    assert isinstance(indices, jt.Var), "indices must be jt.Var"
    assert indices.dtype in (jt.int32, jt.int64), f"indices.dtype must be int, got {indices.dtype}"
    assert indices.ndim == 2, f"indices must be 2D, got {indices.ndim}D"
    assert indices.shape[1] in (3, 4), f"indices must have 3 or 4 columns, got {indices.shape[1]}"

    N = indices.shape[0]
    assert N != 0

    mask = jt.empty(N, dtype='bool')
    mask[:1] = True
    mask[1:] = (indices[1:] != indices[:-1]).any(dim=1)
    unique_indices = indices[mask]  # (M, C)

    # 计算逆映射（每个原始行对应的唯一索引）
    inverse = jt.cumsum(mask.astype(jt.int32)) - 1
    # 准备返回值
    outputs = [unique_indices]

    if return_index:
        index = jt.nonzero(mask).reshape(-1)   # shape (M,), 每个值为原数组中的行索引
        outputs.append(index)
    if return_inverse:
        outputs.append(inverse)
    if return_count:
        idx = jt.concat([jt.nonzero(mask).view(-1), jt.array(mask.shape[0])])
        count = idx[1:] - idx[:-1]
        outputs.append(count)

    return outputs[0] if len(outputs) == 1 else outputs


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


class octformerModule(ModelSpec):
    
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        
        cfg = self.model_config
        

        
        self.norm_size = cfg['norm_size']
        self.stem_down = 0
        self.out_feature_channel = cfg['out_feature_channel']
        self.fpn_channel      = cfg['fpn_channel']
        self.channels         = cfg['channels']
        self.embedding_dim    = self.channels[0]
        self.encoder_k        = cfg['encoder_k']
        self.loss_k           = cfg['loss_k']            # if -1, use clean - noisy
        self.decoder_type     = cfg['decoder_type']
        assert self.decoder_type in ['neighbor','near']
        self.decode_num_block = cfg['decode_num_block']
        self.encoder_gather   = cfg['encoder_gather']
        assert self.encoder_gather in ['mean','max']
        self.decoder_k        = cfg['decoder_k']

        self.num_predict_step = cfg['num_predict_step']
        self.predict_patch_size = cfg['predict_patch_size']
        self.predict_patch_mode = cfg['predict_patch_mode']
        assert self.predict_patch_mode in ['knn','morton']

        
        self.model = OctFormerDenoise(
                               self.embedding_dim,
                               self.out_feature_channel,
                               channels=    self.channels,   # default [96, 96, 96]
                               num_blocks=  cfg['num_blocks'], # default [4, 4, 4],
                               num_heads=   cfg['num_heads'],  # default [6, 6, 6],
                               patch_size=  cfg['patch_size'],
                               dilation=    cfg['dilation'],
                               drop_path=   cfg['drop_path'],
                               stem_down=   self.stem_down,
                               fpn_channel= cfg['fpn_channel'], # default [96, 96, 96]
                               head_drop=   cfg['head_drop'],
                               use_dwconv=  cfg['use_dwconv']
                               )
        self.encoder = VoxelFeatureEncoder(3, self.embedding_dim,
             k=self.encoder_k, coord_dim = 3, hidden_dim=cfg['encoder_hidden_dim'])
        # self.encoder = nn.Sequential(
        #     nn.Linear(3, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Linear(128, 96),)
        # self.encoder = FeatureExtraction(k=16, input_dim=3, embedding_dim=96)
        self.merge_voxel = cfg['merge_voxel']
        self.mlp = ScoreNetLinear(self.out_feature_channel, 3, 3, 
                            cfg['decoder_hidden_dim'], self.decode_num_block)

        
    def buildTensor(self, indices, values=None, points=None):
        '''
        这个哈希设计的非常不合理，parse_quantize是简单合并所有哈希值相同的点，
        这也就意味着即使两个点隔着十万八千里，只要哈希值相同就会合并。
        解决方案：重写哈希/去除异常点
        这里选择依据已经排序的特性重写去重代码
        '''
        # hash_multiplier = set_hash(ndim=3, seed=42)
        # hash_multiplier = jt.array([124, 119, 620, 692], dtype='int32') 
        # indices, mapping, inverse_mapping, count = \
        #         sparse_quantize(indices, hash_multiplier, 1, return_index=True, return_inverse=True, return_count=True)
        
        
        indices, mapping, inverse_mapping, count = \
                my_sparse_quantize(indices, return_index=True, return_inverse=True, return_count=True)
        
        # # 输出结果
        
        # count = count.reshape(-1,1)
        # ooo, _, www = jt.unique(count, return_inverse=True,return_counts=True)
        # print("值\t出现次数")
        # for v, c in zip(ooo.numpy(), www.numpy()):
        #     print(v,c)

        if self.merge_voxel:
            assert values is not None
            count = count.reshape(-1,1)
            out_size = (indices.shape[0], values.shape[-1])
            values_sum = jt.zeros(out_size, dtype=values.dtype).scatter_(0, inverse_mapping, values, reduce='add')
            values_sum /= count
            values = jt.concat([count, values_sum],dim=1)      # 额外将计数作为特征值输入
        else:
            assert values is None
            dense_coords, _mask, _mapping, _batch_size, _counts = pad_voxel_centers(indices)
            dense_feats = self.encoder(dense_coords + 0.5, points)
            values = dense_to_sparse(dense_feats, _mapping, indices.shape[0])
        
        assert values.shape[0] == indices.shape[0]
        return mapping, inverse_mapping, SparseTensor(
            values=values,
            indices=indices,
            stride=1,
            quantize=False,
            coalesce_mode='sum'
        )
    def encode(self, pc_noisy):
        '''
        in:(B,N,3)
        out:(B,N,3)
        数值范围:约 (-5,5)
        '''
        B, N, d = pc_noisy.shape
        
        voxel_coords = (pc_noisy).floor().int() # (B, N, 3)
        
        
        # 展平并构建 batch 索引
        points_flat       = pc_noisy.reshape(-1, 3)              # (B*N, 3)
        voxel_coords_flat = voxel_coords.reshape(-1, 3)          # (B*N, 3)
        batch_ids = jt.arange(B).reshape(-1, 1, 1).expand(B, N, 1).reshape(-1, 1) # (B*N, 1)
        
        # 排序
        sorted_indices    = morton_sort(voxel_coords_flat, batch_ids)
        points_flat       = points_flat[sorted_indices, :]       # (B*N, 3)
        voxel_coords_flat = voxel_coords_flat[sorted_indices, :] # (B*N, 3)


        # 构建 sparseTensor
        indices = jt.concat([batch_ids, voxel_coords_flat], dim=1)
        input_values = None
        if self.merge_voxel:
            input_values = self.encoder(pc_noisy).reshape(B*N, -1)        # (B*N, 64)


        mapping, inverse_mapping, tensor = self.buildTensor(
            indices, values=input_values, points=pc_noisy)
        assert tensor.values.shape[0] == mapping.shape[0]
        feature_tensors = self.model(tensor)

        return feature_tensors

    
    
    def interpolate_voxel_to_point_idx_neighbor(self,
        x: SparseTensor,
        points: jt.Var,              # (N, 4) 带 batch 注意 voxelsize=？
        nearest: bool = False        # 是否使用最近邻（仅最近邻体素）
    ) -> jt.Var:
        """
        对每个点，从其周围8个体素中特征。

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
        idx_query = F.spquery(cube_hash, indices_hash).int().t()
        if nearest:
            idx_query[:, 1:] = -1
        return idx_query
    
    

    def interpolate_voxel_to_point_idx_near(self,
        x: SparseTensor,
        points: jt.Var,              # (N, 4) 带 batch 注意 voxelsize=？
        nearest: bool = False,       # 是否使用最近邻（仅最近邻体素）
        k=8):
        """
        为每个点找到同 batch 内最近的 k 个体素，返回全局索引。

        Args:
            points:  (total_pts, 4)  每一行 [batch_id, x, y, z]
            indices: (total_vox, 4)  每一行 [batch_id, x, y, z]
            k:       int             最近邻个数，默认 8

        Returns:
            global_idx: (total_pts, k)  每个点对应的 k 个体素的全局索引（在 indices 中的行号）
        """

        indices = x.indices
        pts_batch = points[:, 0].int()
        pts_coords = points[:, 1:4].float()
        idx_batch = indices[:, 0].int()
        idx_coords = indices[:, 1:4].float()

        batch_size = int(pts_batch.max().item()) + 1
        P = points.shape[0]
        neighbor_ids = jt.full((P, k), -1, dtype='int32')

        for b in range(batch_size):
            # 当前batch中的点
            pts_mask = (pts_batch == b).reshape(-1)
            pts_pos = jt.nonzero(pts_mask).squeeze(1)
            if pts_pos.shape[0] == 0:
                continue

            # 当前batch中的体素
            idx_mask = (idx_batch == b).reshape(-1)
            idx_pos = jt.nonzero(idx_mask).squeeze(1)
            M = idx_pos.shape[0]
            if M == 0:
                continue

            # 整理为 (1, N_b, 3) 和 (1, M_b, 3) 以调用 get_knn_idx
            x = pts_coords[pts_pos].unsqueeze(0)   # (1, N_b, 3)
            y = idx_coords[idx_pos].unsqueeze(0)   # (1, M_b, 3)

            # 实际能取的最多邻居数
            k_actual = min(k, M)
            # 调用 get_knn_idx，offset=0 表示取第0~k_actual-1个邻居
            local_idx = get_knn_idx(x, y, k_actual, offset=0)  # (1, N_b, k_actual)
            local_idx = local_idx.squeeze(0)                   # (N_b, k_actual)

            # 局部索引 → 全局索引（原始 indices 中的行号）
            global_ids = idx_pos[local_idx]                    # (N_b, k_actual)

            # 填充输出
            neighbor_ids[pts_pos, :k_actual] = global_ids
            # 若 k_actual < k，剩余列已默认 -1
        return neighbor_ids
    
    
    def interpolate_voxel_to_point_offset_feat(self,
        x: SparseTensor,
        points: jt.Var,              # (N, 4) 带 batch 维度
        nearest: bool = False,
        k: int = 4
    ) -> jt.Var:
        """
        对每个点，预测其周围 8 个体素贡献的偏移量（不平均），
        与目标偏移量 grad_target 计算 L2 损失（仅有效邻居）。
        返回损失标量。
        """
        if self.decoder_type == 'neighbor':
            assert k == 8
            idx_query = self.interpolate_voxel_to_point_idx_neighbor(x, points, nearest)
        else:
            idx_query = self.interpolate_voxel_to_point_idx_near(x, points, nearest, k)

        N, C = idx_query.shape[0], x.values.shape[1]

        # 2. 并行收集 k 个体素的特征 (N, k, C)
        flat_idx = idx_query.reshape(-1)                # (N*k,)
        valid_mask = flat_idx >= 0                      # (N*k,)
        neighbor_feats_flat = jt.zeros((N * k, C), dtype=x.values.dtype)
        valid_idx = flat_idx[valid_mask]                # 有效索引
        valid_feats = x.values[valid_idx]               # (num_valid, C)
        flat_pos = jt.arange(N * k)[valid_mask]         # 有效位置
        neighbor_feats_flat[flat_pos] = valid_feats
        neighbor_feats = neighbor_feats_flat.view(N, k, C)   # (N,k,C)

        
        # 坐标收集：体素世界坐标（中心）
        neighbor_coords_flat = jt.zeros((N * k, 3), dtype=points.dtype)
        if valid_idx.numel() > 0:
            valid_coords = x.indices[valid_idx][:, 1:]   # (num_valid, 3) 整数索引
            valid_centers = valid_coords.float() + 0.5 * x.stride[0]   # 世界坐标中心
            neighbor_coords_flat[flat_pos] = valid_centers
        neighbor_centers = neighbor_coords_flat.view(N, k, 3)  # (N,k,3)

        # 3. 计算相对坐标 (N,k,3)：当前点坐标（不含batch）减去体素中心
        # points[:, 1:] 是 (N,3)，需要扩展为 (N,1,3) 然后广播
        points_xyz = points[:, 1:].unsqueeze(1)          # (N,1,3)
        delta = points_xyz - neighbor_centers            # (N,k,3)

        # 4. 拼接特征：体素特征 + 相对坐标 (N,k, C+3)
        concat_feats = jt.concat([neighbor_feats, delta], dim=2)   # (N,k,C+3)


        # 将 (N,k,C) 合并为 (N*k, C) 一次性前向
        flat_feats = concat_feats.view(N * k, C + 3)
        flat_pred_offsets = self.mlp(flat_feats)             # (N*k, 3)

        flat_pred_offsets[jt.logical_not(valid_mask)] = 0.0

        return valid_mask, flat_pred_offsets.view(N, k, 3) 
    
    def interpolate_voxel_to_point_knn(self,
        x: SparseTensor,
        points: jt.Var,              # (N, 4) 带 batch 注意 voxelsize=？
        nearest: bool = False,        # 是否使用最近邻（仅最近邻体素）
        k: int = 4
    ) -> jt.Var:
        """
        对每个点，从其周围k个体素中获得特征。

        Args:
            voxel_tensor: SparseTensor，体素特征，步长 stride。
            point_coords: 点坐标，形状  (N, 4)（带 batch 维度）。
            nearest: 若为 True，则仅使用最近邻体素（等价于最近邻插值）。

        Returns:
            new_values: (N, C) 每个点的插值特征。
        """
        valid_mask, offsets_k = self.interpolate_voxel_to_point_offset_feat(
            x, points, nearest, k)
        N = points.shape[0]

        # 4. 对有效邻居取平均
        valid_count = valid_mask.view(N, k).sum(dim=1, keepdim=True)  # (N,1)
        sum_offsets = offsets_k.sum(dim=1)                            # (N,3)
        final_offsets = sum_offsets / (valid_count + 1e-8)            # 避免除零

        # print("idx_query -1 ratio:", (idx_query == -1).float().mean().item())
        return final_offsets
    
    def interpolate_voxel_to_point_offset_loss(self,
        x: SparseTensor,
        points: jt.Var,              # (N, 4) 带 batch 维度
        grad_target: jt.Var,         # (N, 3) 每个点的目标偏移量（真实值）
        nearest: bool = False,
        k: int = 4 
    ) -> jt.Var:
        """
        对每个点，预测其周围 8 个体素贡献的偏移量（不平均），
        与目标偏移量 grad_target 计算 L2 损失（仅有效邻居）。
        返回损失标量。
        """
        valid_mask, pred_offsets = self.interpolate_voxel_to_point_offset_feat(
            x, points, nearest, k)             # (N*k, 3)

        # 4. 计算损失：扩展 grad_target 到 (N,k,3)，仅对有效位置计算 MSE
        grad_target_exp = grad_target.unsqueeze(1)       # (N,1,3)
        diff = pred_offsets - grad_target_exp           # (N,k,3)
        loss = (diff ** 2).sum(dim=2)            # (N,k)

        return loss.mean()

    def compute_grad_target(self, noisy_points, clean_points, k=8, coord_dim=3):
        """
        仿照体素特征提取的结构，为每个噪声点计算目标梯度（位移向量）。
        
        Args:
            noisy_points: (B, N_q, coord_dim) 噪声点坐标，通常是每个中心点的邻点（展平后）
            clean_points: (B, N_clean, coord_dim) 干净点云坐标
            k: 每个噪声点在干净点云中寻找的最近邻数量（如16）
            coord_dim: 坐标维度（默认3）
        
        Returns:
            grad_target: (B, N_q, coord_dim) 目标位移向量 = clean_avg - noisy_points
            clean_avg:   (B, N_q, coord_dim) 干净点云局部邻域平均坐标（可选）
        """
        B, N_q, _ = noisy_points.shape
        B_clean, N_clean, _ = clean_points.shape
        assert B == B_clean, "Batch size mismatch"

        # 1. 对每个噪声点，在干净点云中找 k 个最近邻索引
        #    此处假设有 get_knn_idx 函数，返回 (B, N_q, k)
        knn_idx = get_knn_idx(noisy_points, clean_points, k, offset=0)

        # 2. 收集邻域点的坐标（与体素代码中收集 neighbor_feats 类似）
        base = jt.arange(B).reshape(B, 1, 1) * N_clean          # (B, 1, 1)
        flat_idx = (knn_idx + base).reshape(-1)                 # (B * N_q * k,)
        clean_flat = clean_points.reshape(B * N_clean, coord_dim)
        neighbor_coords = clean_flat[flat_idx]                  # (B * N_q * k, coord_dim)
        neighbor_coords = neighbor_coords.reshape(B, N_q, k, coord_dim)   # (B, N_q, k, coord_dim)

        # 3. 在邻域维度上取平均（对应体素代码中的 .mean(dim=2)）
        clean_avg = neighbor_coords.mean(dim=2)                 # (B, N_q, coord_dim)

        # 4. 目标位移 = 干净局部平均 - 当前噪声点
        grad_target = clean_avg - noisy_points                  # (B, N_q, coord_dim)

        return grad_target, clean_avg

    def decode(self, feature_tensor, pc_noisy, pc_clean = None):
        B, N, d = pc_noisy.shape
        points_flat = pc_noisy.reshape(-1, 3)
        batch_ids = jt.arange(B).reshape(-1, 1, 1).expand(B, N, 1).reshape(-1, 1) # (B*N, 1)

        input = jt.concat([batch_ids, points_flat], dim=1)
        if pc_clean is not None:
            grad_target = None
            if self.loss_k == -1:
                clean_flat = pc_clean.reshape(-1, 3)
                grad_target = (clean_flat - points_flat)
            else:
                grad_target, _ = self.compute_grad_target(pc_noisy, pc_clean, self.loss_k)
                grad_target = grad_target.reshape(-1, 3)
            loss = self.interpolate_voxel_to_point_offset_loss(
                    feature_tensor, input, grad_target, self.decoder_k)
            return loss
        else:
            pred_dir = self.interpolate_voxel_to_point_knn(
                    feature_tensor, input, self.decoder_k)
        # feature = [input_values]
        # for feature_tensor in feature_tensors:
        #     feature.append(interpolate_voxel_to_point(feature_tensor, input))
        # feature = jt.concat(feature, dim=1)
        # pred_dir = self.model.outputapply(feature)

            return pred_dir.reshape(B, N, 3)

    
    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
        """
        pcl_noisy: (B, N, 3)
        pcl_clean: (B, N, 3)
        经过大量时间，负数不能够正常进行哈希。为了一劳永逸解决问题，永久禁止负数使用
        """
        
        pc_mix = (pc_mix + 1) * (1. / self.norm_size)
        pc_clean = (pc_clean + 1) * (1. / self.norm_size)
        pc_noisy = (pc_noisy + 1) * (1. / self.norm_size)
        assert pc_mix.min() > 0
        
        feature = self.encode(pc_noisy)
        return self.decode(feature, pc_mix, pc_clean)
        pred = self.go_one_step(pc_clean)
        pred_dir = pred[:,:,0:3]
        # pred_len = pred[:,:,3]
        # pred_normal = pred[:,:,4:7]


        # grad_dir_t_target = pc_clean - pc_mix
        loss_pre = jt.loss3d.chamfer_loss(pc_mix + pred_dir, pc_clean,
            reduction='mean', dims='BNC', bidirectional=True)
        norm_quantile_report(pred_dir)

        return loss_pre
    
    def training_step(self, batch: Dict) -> Dict:
        # return self.get_predict_loss(batch)
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(
            pc_noisy=pc_noisy,
            pc_mix=pc_mix,
            pc_clean=pc_clean,
        )
        return {"loss": loss}
    
    def execute(self, **kwargs) -> Dict: # type: ignore
        # return self.get_predict_loss(**kwargs)
        return self.training_step(**kwargs)
    
    def get_predict_loss(self, batch):
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        pred_dir = self.predict_step({'pc_noisy':pc_noisy})
        # ans=pc_noisy
        ans = pred_dir[0]['pc_denoised'].reshape(-1, patch_size, 3)
        
        loss_pre = jt.loss3d.chamfer_loss(ans, pc_clean,
            reduction='mean', dims='BNC', bidirectional=True)
        return {"loss":loss_pre}

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        
        assert pc_noisy_batch.ndim == 3
        
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            pc_next = pc_noisy.clone()
            '''
                pc_noisy: (50000,3) jt.var
                需要注意的是输出的时候并不能够进行任何处理，
                因此所有处理只能在这部分里面进行
                所有归一化的逆操作都只能在这里搞
            '''

            p_max = pc_next.max(dim=0)
            p_min = pc_next.min(dim=0)
            center = (p_max + p_min) / 2
            pc_next = pc_next - center
            scale = jt.sqrt((pc_next ** 2).sum(dim=1).max(dim=0)[0])
            pc_next = pc_next * (1. / scale)

            if self.predict_patch_mode == 'knn':
                pc_next = patch_based_denoise(
                    model=self,
                    pcl_noisy=pc_next,
                    patch_size=self.predict_patch_size,
                    seed_k=10,
                    seed_k_alpha=1,
                )
            else:
                for it in range(self.num_predict_step, 0, -1):
                    pred_dir = self.go_one_step_morton(pc_next)
                    pc_next = pc_next + (1.0 / it) * pred_dir
            
            pc_denoised = pc_next * scale + center
            pc_denoised = pc_denoised.detach().numpy()
            assert len(pc_denoised.shape) == 2
            assert pc_denoised.shape[0] == 50000
            assert pc_denoised.shape[1] == 3
            res.append({"pc_denoised": pc_denoised})
        return res
    
    def denoise_langevin_dynamics(self, pcl_noisy):
        """
        pcl_noisy: (B, N, 3)
        """
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            pc_max = pcl_next.max(dim=1).reshape(-1, 1, 3)
            pc_min = pcl_next.min(dim=1).reshape(-1, 1, 3)
            center = (pc_max + pc_min) / 2
            pcl_next = pcl_next - center

            pcl_next = (pcl_next + 1) / self.norm_size
            assert pcl_next.min() > 0
            feature = self.encode(pcl_next)
            # pcl_next += self.decode(feature, pcl_next)
            for it in range(self.num_predict_step, 0, -1):
                ret = self.decode(feature, pcl_next)
                pcl_next += 0.8 * ret
            pcl_next = pcl_next * self.norm_size - 1 + center
            # pcl_next = pcl_next + center - (1 / self.norm_size)
        return pcl_next, None
    
    def go_one_step_morton(self, pc_noisy):
        '''
            pc_noisy: (50000,3)
        '''
        raise NotImplementedError

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append({ # (num_patches, patch_size, 3) 32 倍于原始数值
                    "pc_noisy": b.meta['pc_noisy'],
                    "pc_clean": b.meta['pc_clean'],
                    "pc_mix": b.meta['pc_mix'],
                })
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy, # (N, 3)
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res




def farthest_point_sampling(pcls, num_pnts):
    """
    pcls: (B, N, 3)
    return:
        sampled: (B, num_pnts, 3)
        indices: (B, num_pnts)
    """
    B, N, _ = pcls.shape
    sampled = []
    indices = []
    for b in range(B):
        pts = pcls[b]  # (N, 3)
        selected = []
        dist = jt.ones((N,)) * 1e10
        farthest = 0
        for i in range(num_pnts):
            selected.append(farthest)
            centroid = pts[farthest]  # (3,)
            d = ((pts - centroid) ** 2).sum(dim=1)
            dist = jt.minimum(dist, d)
            farthest, _ = jt.argmax(dist, dim=-1)
            farthest = farthest.item()
        idx = jt.array(selected).int32()
        sampled.append(pts[idx][None, ...])
        indices.append(idx[None, ...])
    sampled = jt.concat(sampled, dim=0)
    indices = jt.concat(indices, dim=0)
    return sampled, indices

def knn_points(x, y, k):
    """
    x: (B, P, 3)
    y: (B, N, 3)
    return:
        dist: (B, P, k)
        idx:  (B, P, k)
        nn:   (B, P, k, 3)
    """
    dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    dist_k, idx = jt.topk(dist, k=k, dim=-1, largest=False)
    B = x.shape[0]
    nn = []
    for b in range(B):
        nn.append(y[b][idx[b]])
    nn = jt.stack(nn, dim=0)
    return dist_k, idx, nn

def patch_based_denoise(model , pcl_noisy, patch_size=1000, seed_k=6, seed_k_alpha=1) -> jt.Var:
    """
    pcl_noisy: (N, 3)
    """
    assert len(pcl_noisy.shape) == 2
    
    N, d = pcl_noisy.shape
    num_patches = int(seed_k * N / patch_size)
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    
    seed_pnts, seed_idx = farthest_point_sampling(pcl_noisy, num_patches)
    patch_dists, point_idxs, patches = knn_points(seed_pnts, pcl_noisy, patch_size)
    
    from ..data.asset import Exporter
    pts = patches[0].reshape(-1, 3).detach().numpy()
    
    patches = patches[0]              # (P, M, 3)
    patch_dists = patch_dists[0]      # (P, M)
    point_idxs = point_idxs[0]        # (P, M)
    
    seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_expand
    
    patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)
    
    all_dists = jt.ones((num_patches, N)) * 1e10
    
    for i in range(num_patches):
        all_dists[i][point_idxs[i]] = patch_dists[i]
        
    weights = jt.exp(-all_dists)
    best_weights_idx, _ = jt.argmax(weights, dim=0)
    patches_denoised = []
    
    i = 0
    patch_step = int(ceil(N / (seed_k_alpha * patch_size)))
    assert patch_step > 0
    while i < num_patches:
        curr = patches[i:i+patch_step]
        try:
            out, _ = model.denoise_langevin_dynamics(curr)
        except Exception as e:
            print("Denoise error:", e)
            return None
        patches_denoised.append(out)
        i += patch_step
    
    patches_denoised = jt.concat(patches_denoised, dim=0)
    patches_denoised = patches_denoised + seed_expand
    pcl_out = []
    for pidx in range(N):
        patch_id = best_weights_idx[pidx].item()
        mask = (point_idxs[patch_id] == pidx)
        pcl_out.append(patches_denoised[patch_id][mask])
    pcl_out = jt.concat(pcl_out, dim=0)
    return pcl_out





def estimate_normals(points, k, exclude_self=True):
    """
    通过 KNN + PCA 估计点云法向量

    Args:
        points: (B, N, 3)  批次点云
        k: int             使用的邻域点数（若 exclude_self=True，实际邻域数为 k+1 后再剔除自身）
        exclude_self: bool 是否排除查询点自身（推荐 True，避免退化）

    Returns:
        normals: (B, N, 3) 单位法向量（未统一方向，符号随机）
    """
    B, N, _ = points.shape

    # 1. 查询近邻（若排除自身，则多取一个点，之后去掉第一个最近邻）
    k_query = k + 1 if exclude_self else k
    _, _, nn = knn_points(points, points, k_query)          # nn: (B, N, k_query, 3)

    if exclude_self:
        nn = nn[:, :, 1:, :]                                # 去掉自身点 → (B, N, k, 3)

    # 2. 计算每个邻域的中心并中心化
    mean = nn.mean(dim=2, keepdim=True)                     # (B, N, 1, 3)
    centered = nn - mean                                    # (B, N, k, 3)

    # 3. 批量计算协方差矩阵 (3x3)
    # centered: (B, N, k, 3) → 转置为 (B, N, 3, k) 再与自身相乘 → (B, N, 3, 3)
    cov = (centered.transpose(-2, -1) @ centered) / k       # (B, N, 3, 3)

    # 4. 特征分解（eigh 默认升序排列特征值）
    
    cov_np = cov.numpy()
    # 计算特征值和特征向量
    eigvals_np, eigvecs_np = np.linalg.eigh(cov_np)
    # 取最小特征值对应的特征向量（第一列）
    normals_np = eigvecs_np[..., 0]  # (B, N, 3)
    # 转回 Jittor 张量
    normals = jt.array(normals_np)
    

    return normals