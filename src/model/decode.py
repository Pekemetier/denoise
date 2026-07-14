import jittor as jt
import jittor.nn as nn

class ResnetBlockLinear(nn.Module):
    """全连接残差块，支持条件向量"""
    def __init__(self, c_dim, size_in, size_h=None, size_out=None):
        super().__init__()
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # 两个全连接层（等价于 1x1 卷积）
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        # 条件映射层
        self.fc_c = nn.Linear(c_dim, size_out)
        self.actvn = nn.ReLU()
        self.norm_0 = nn.LayerNorm(size_h)   # 代替 BatchNorm1d
        self.norm_1 = nn.LayerNorm(size_out)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)

        # 初始化：模仿原始代码将 fc_1 的权重置零
        jt.init.zero_(self.fc_1.weight)

    def execute(self, x, c):
        # x: (..., size_in)   c: (..., c_dim)
        net = self.fc_0(x)
        net = self.actvn(self.norm_0(net))
        dx = self.fc_1(net)
        dx = self.actvn(self.norm_1(dx))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        out = x_s + dx + self.fc_c(c)
        return out


class ScoreNetLinear(nn.Module):
    """使用 Linear 层的 ScoreNet，支持逐点条件"""
    def __init__(self, z_dim, dim=3, out_dim=3, hidden_size=128, num_blocks=4):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks

        c_dim = z_dim + dim   # 条件通道 = 特征维度 + 坐标维度
        # 首层映射：将 (dim+z_dim) 映射到 hidden_size
        self.fc_p = nn.Linear(c_dim, hidden_size)
        self.blocks = nn.ModuleList([
            ResnetBlockLinear(c_dim, hidden_size) for _ in range(num_blocks)
        ])
        self.norm_out = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_dim)
        self.actvn_out = nn.ReLU()

    def execute(self, x):
        """
        Args:
            x: (B, N, dim) 相对坐标（或绝对坐标）
            c: (B, N, z_dim) 逐点条件向量（例如中心点特征）
        Returns:
            (B, N, out_dim) 梯度预测
        """
        # 拼接坐标和条件: (B, N, dim+z_dim)
        # c_xyz = jt.concat([x, c], dim=-1)
        c_xyz = x
        # 首层
        net = self.fc_p(c_xyz)
        net = self.actvn_out(net)   # 原始代码在首层后没有激活？这里加一个保持结构
        # 残差块
        for block in self.blocks:
            net = block(net, c_xyz)
        # 输出层
        out = self.fc_out(self.actvn_out(self.norm_out(net)))
        return out