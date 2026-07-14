from typing import Optional,Tuple
import jittor as jt
from jittor import nn

# 复用原有的 get_knn_idx 函数
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


class VoxelFeatureEncoder(nn.Module):
    """
    体素特征编码器：为每个体素中心聚合邻域点特征（单轮，无动态构图）
    
    Args:
        in_channels: 输入点云的特征维度（通常包含坐标+额外特征）
        out_channels: 输出体素特征的维度
        k: 每个体素中心邻域包含的点数（默认为32）
        coord_dim: 用于距离计算的坐标维度（通常为3）
        activation: MLP 中的激活函数，可选 'ReLU' 或 None
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 128,
        k: int = 32,
        coord_dim: int = 3,
        activation: Optional[str] = 'ReLU',
        gather:  Optional[str] = 'mean'
    ):
        super().__init__()
        self.k = k
        self.coord_dim = coord_dim
        self.gather = gather

        # 构建 MLP 结构（两层全连接 + 激活）
        if activation == 'ReLU':
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(),
                # nn.Linear(hidden_dim, hidden_dim),
                # nn.ReLU(),
                nn.Linear(hidden_dim, out_channels),
                nn.ReLU()
            )
        elif activation is None:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(),
                # nn.Linear(hidden_dim, hidden_dim),
                # nn.ReLU(),
                nn.Linear(hidden_dim, out_channels),
            )
        else:
            raise ValueError("activation must be 'ReLU' or None")

    def execute(self, voxel_centers, points):
        """
        前向传播
        
        Args:
            voxel_centers: (B, N_v, C_v) 体素中心坐标，通常 C_v >= coord_dim
            points: (B, N_pts, C_in) 点云数据，前 coord_dim 维为坐标，剩余为特征
        
        Returns:
            voxel_features: (B, N_v, out_channels) 编码后的体素特征
        """
        B, N_v, _ = voxel_centers.shape
        _, N_pts, C_in = points.shape
        assert B == points.shape[0], "Batch sizes do not match"

        # 提取坐标部分用于 KNN 搜索
        centers_xyz = voxel_centers[..., :self.coord_dim]   # (B, N_v, coord_dim)
        points_xyz = points[..., :self.coord_dim]           # (B, N_pts, coord_dim)
        

        # 1. 对每个体素中心，找到最近的 k 个点的索引
        knn_idx = get_knn_idx(centers_xyz, points_xyz, self.k, offset=0)  # (B, N_v, k)

        # 2. 收集邻域点的完整特征（包括坐标+额外特征）
        # 将索引展平并加上 batch 偏移
        base = jt.arange(B).reshape(B, 1, 1) * N_pts          # (B, 1, 1)
        flat_idx = (knn_idx + base).reshape(-1)               # (B * N_v * k,)
        points_flat = points.reshape(B * N_pts, C_in)         # (B * N_pts, C_in)
        neighbor_feats = points_flat[flat_idx]                # (B * N_v * k, C_in)
        neighbor_feats = neighbor_feats.reshape(B, N_v, self.k, C_in)  # (B, N_v, k, C_in)

        # 3. 对每个邻域点独立进行 MLP 变换
        mlp_out = self.mlp(neighbor_feats)                    # (B, N_v, k, out_channels)

        # 4. 在邻域维度上取平均，得到每个体素的最终特征
        if self.gather == 'mean':
            voxel_features = mlp_out.mean(dim=2)                  # (B, N_v, out_channels)
        elif self.gather == 'max':
            voxel_features = mlp_out.max(dim=2)                  # (B, N_v, out_channels)
        else:
            raise NotImplementedError

        return voxel_features
    


def pad_voxel_centers(indices):
    """
    输入: indices (total, 4) -> [batch_id, x, y, z]
    输出:
        dense_coords (B, max_N, 3)
        mask        (B, max_N) bool
        mapping     (B, max_N) int (原始索引，填充处为 -1)
        batch_size  int
        counts      (B,) int
    """
    batch_ids = indices[:, 0].int()
    coords = indices[:, 1:4]
    total = indices.shape[0]

    batch_size = int(batch_ids.max().item()) + 1
    counts = jt.zeros((batch_size,), dtype='int32')
    for b in range(batch_size):
        counts[b] = (batch_ids == b).sum()
    max_N = int(counts.max().item())

    dense_coords = jt.zeros((batch_size, max_N, 3), dtype='float32')
    mask = jt.zeros((batch_size, max_N), dtype='bool')
    mapping = jt.full((batch_size, max_N), -1, dtype='int32')

    # 逐 batch 填充（batch_size 一般很小，简单循环即可）
    for b in range(batch_size):
        # 找出属于当前 batch 的索引
        is_b = (batch_ids == b).reshape(-1)          # bool 向量
        # 在 jittor 中，不能直接用 bool 索引 Var，但可以用 where 获取位置
        # 方法：获取所有符合条件的索引位置
        positions = jt.nonzero(is_b)                 # (num, 1)
        if positions.shape[0] == 0:
            continue
        pos_flat = positions[:, 0]                  # (num,)
        cnt = counts[b].item()
        # 取前 cnt 个（实际就是全部）
        dense_coords[b, :cnt] = coords[pos_flat]
        mask[b, :cnt] = True
        mapping[b, :cnt] = pos_flat

    return dense_coords, mask, mapping, batch_size, counts


def dense_to_sparse(dense_feats, mapping, total_voxels):
    """
    输入:
        dense_feats: (B, max_N, C)
        mapping:     (B, max_N) 每个位置的原始索引（-1 表示填充）
        total_voxels: 原始体素总数
    输出:
        sparse_feats: (total_voxels, C)
    """
    B, max_N, C = dense_feats.shape
    # 展平
    flat_mapping = mapping.reshape(-1)                     # (B*max_N,)
    flat_feats = dense_feats.reshape(-1, C)                # (B*max_N, C)
    # 有效位置的掩码
    valid = flat_mapping != -1
    valid_idx = flat_mapping[valid]                        # 原始索引
    valid_feats = flat_feats[valid]                        # 对应特征

    # 使用 scatter 散布到结果张量中
    sparse_feats = jt.zeros((total_voxels, C), dtype=dense_feats.dtype)
    # jt.misc.scatter(x, dim, index, src, reduce='add')
    # 将 valid_feats 按行（dim=0）散布到 valid_idx 指定的行
    sparse_feats = jt.misc.scatter(sparse_feats, 0, valid_idx.unsqueeze(1).broadcast(valid_feats.shape), valid_feats, reduce='add')
    return sparse_feats