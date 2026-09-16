"""
ST-GWNN优化版本
- 使用固定地理距离图
- 添加注意力机制代替K近邻（更高效）
- 优化计算流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphWaveletLayer(nn.Module):
    """图小波层（优化版）"""
    def __init__(self, in_dim, out_dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 为每个尺度创建权重
        self.scale_weights = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_scales)
        ])
        
        self.bn = nn.BatchNorm1d(out_dim)
        
    def forward(self, x, adj):
        """
        x: (batch, num_nodes, in_dim)
        adj: (num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = x.shape
        
        # 计算归一化拉普拉斯（只计算一次）
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree + 1e-6))
        laplacian = torch.diag(degree) - adj
        norm_laplacian = torch.mm(torch.mm(degree_inv_sqrt, laplacian), degree_inv_sqrt)
        
        outputs = []
        for scale_idx in range(self.num_scales):
            # 计算小波基
            if scale_idx == 0:
                wavelet_base = torch.eye(num_nodes, device=x.device)
            else:
                wavelet_base = torch.matrix_power(norm_laplacian, scale_idx)
            
            # 应用小波变换
            x_transformed = torch.matmul(wavelet_base, x)
            
            # 线性变换
            out = self.scale_weights[scale_idx](x_transformed)
            outputs.append(out)
        
        # 聚合多尺度特征
        output = sum(outputs) / self.num_scales
        
        # Batch normalization
        output = output.transpose(1, 2)
        output = self.bn(output)
        output = output.transpose(1, 2)
        
        return F.relu(output)


class SpatialAttention(nn.Module):
    """空间注意力机制（代替K近邻）"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim // 2)
        self.key = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = (hidden_dim // 2) ** 0.5
        
    def forward(self, x, geo_adj):
        """
        x: (batch, num_nodes, hidden_dim)
        geo_adj: (num_nodes, num_nodes) 地理距离图
        """
        batch_size, num_nodes, _ = x.shape
        
        # 计算注意力分数
        Q = self.query(x)  # (batch, num_nodes, hidden_dim//2)
        K = self.key(x)    # (batch, num_nodes, hidden_dim//2)
        V = self.value(x)  # (batch, num_nodes, hidden_dim)
        
        # 注意力权重
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (batch, num_nodes, num_nodes)
        
        # 结合地理距离先验
        geo_adj_expanded = geo_adj.unsqueeze(0).expand(batch_size, -1, -1)
        attn_scores = attn_scores * geo_adj_expanded
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 应用注意力
        output = torch.bmm(attn_weights, V)  # (batch, num_nodes, hidden_dim)
        
        return output, attn_weights


class TemporalConvBlock(nn.Module):
    """时间卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        x: (batch, num_nodes, seq_len, features)
        """
        batch_size, num_nodes, seq_len, features = x.shape
        
        # 重塑为 (batch*num_nodes, features, seq_len)
        x = x.reshape(batch_size * num_nodes, seq_len, features)
        x = x.transpose(1, 2)
        
        # 时间卷积
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 重塑回 (batch, num_nodes, seq_len, out_channels)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, num_nodes, seq_len, -1)
        
        return x


class ST_GWNN_Optimized(nn.Module):
    """优化版ST-GWNN"""
    def __init__(self, config):
        super().__init__()
        
        self.num_nodes = config['num_nodes']
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.seq_len = config['seq_len']
        self.num_gwnn_layers = config.get('num_gwnn_layers', 3)
        self.num_temporal_blocks = config.get('num_temporal_blocks', 3)
        self.num_wavelet_scales = config.get('num_wavelet_scales', 3)
        
        # 固定地理距离图
        self.register_buffer('geo_adj', self._create_geo_adjacency())
        
        # 输入投影
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        
        # 时间卷积块
        self.temporal_blocks = nn.ModuleList([
            TemporalConvBlock(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_temporal_blocks)
        ])
        
        # 空间注意力
        self.spatial_attention = SpatialAttention(self.hidden_dim)
        
        # 图小波层
        self.gwnn_layers = nn.ModuleList([
            GraphWaveletLayer(self.hidden_dim, self.hidden_dim, self.num_wavelet_scales)
            for _ in range(self.num_gwnn_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_gwnn_layers)
        ])
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
        
    def _create_geo_adjacency(self):
        """创建基于地理距离的固定邻接矩阵"""
        coords = torch.tensor([
            [100.46, 38.05],  # Arou
            [98.94, 38.84],   # Dashalong  
            [101.12, 37.67],  # Jingyangling
            [102.14, 33.88],  # Maqu
            [97.60, 35.02],   # Ngoring_lake
            [100.24, 38.01],  # Yakou
        ], dtype=torch.float32)
        
        # 计算欧氏距离
        dist_matrix = torch.cdist(coords, coords, p=2)
        
        # 使用高斯核转换为相似度
        sigma = 2.0
        adj = torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
        
        # 归一化
        adj = adj / adj.sum(dim=1, keepdim=True)
        
        return adj
        
    def forward(self, x):
        """
        x: (batch, num_nodes, seq_len, input_dim)
        返回: (batch, num_nodes, output_dim)
        """
        batch_size = x.shape[0]
        
        # 输入投影
        x = self.input_proj(x)  # (batch, num_nodes, seq_len, hidden_dim)
        
        # 时间卷积
        for temporal_block in self.temporal_blocks:
            x = temporal_block(x) + x  # 残差连接
        
        # 取最后一个时间步
        h = x[:, :, -1, :]  # (batch, num_nodes, hidden_dim)
        
        # 空间注意力
        h_attn, _ = self.spatial_attention(h, self.geo_adj)
        h = h + h_attn  # 残差连接
        
        # 图小波层
        for i, (gwnn_layer, layer_norm) in enumerate(zip(self.gwnn_layers, self.layer_norms)):
            h_gwnn = gwnn_layer(h, self.geo_adj)
            h = layer_norm(h + h_gwnn)  # 残差连接 + 层归一化
        
        # 输出投影
        output = self.output_proj(h)  # (batch, num_nodes, output_dim)
        
        return output


def create_optimized_gwnn_model(config):
    """创建优化版ST-GWNN模型"""
    return ST_GWNN_Optimized(config)