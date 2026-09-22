import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# 使用原生PyTorch实现GCN，避免torch-geometric依赖问题


class GraphConvolution(nn.Module):
    """图卷积层 - 原生PyTorch实现"""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        """
        前向传播
        Args:
            input: 节点特征 [batch_size, num_nodes, in_features]
            adj: 邻接矩阵 [num_nodes, num_nodes]
        Returns:
            output: 输出特征 [batch_size, num_nodes, out_features]
        """
        # 线性变换
        support = torch.matmul(input, self.weight)  # [batch_size, num_nodes, out_features]
        
        # 图卷积
        output = torch.matmul(adj, support)  # [batch_size, num_nodes, out_features]
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GCNLayer(nn.Module):
    """单层GCN层"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(GCNLayer, self).__init__()
        self.gcn = GraphConvolution(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x, adj):
        """
        前向传播
        Args:
            x: 节点特征 [batch_size, num_nodes, in_channels]
            adj: 邻接矩阵 [num_nodes, num_nodes]
        Returns:
            output: 输出特征 [batch_size, num_nodes, out_channels]
        """
        batch_size, num_nodes, _ = x.shape
        
        # 图卷积
        x = self.gcn(x, adj)
        
        # 批归一化 (需要重塑维度)
        x = x.view(batch_size * num_nodes, -1)
        x = self.batch_norm(x)
        x = x.view(batch_size, num_nodes, -1)
        
        # 激活函数和dropout
        x = F.relu(x)
        x = self.dropout(x)
        
        return x


class TemporalGCN(nn.Module):
    """时间序列GCN模型，用于土壤湿度预测"""
    def __init__(self, 
                 num_features=7,
                 hidden_dim=64,
                 num_layers=3,
                 num_nodes=3,
                 sequence_length=30,
                 dropout=0.1):
        super(TemporalGCN, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        
        # 输入特征投影
        self.input_projection = nn.Linear(sequence_length * num_features, hidden_dim)
        
        # GCN层
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim, dropout))
            else:
                self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim, dropout))
        
        # 时间序列处理 - LSTM
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def create_adjacency_matrix(self, num_nodes, device):
        """创建邻接矩阵 - 全连接图"""
        # 创建全连接图的邻接矩阵
        adj = torch.ones(num_nodes, num_nodes, device=device)
        
        # 移除自环
        adj = adj - torch.eye(num_nodes, device=device)
        
        # 添加自环并归一化
        adj = adj + torch.eye(num_nodes, device=device)
        
        # 度矩阵归一化 D^(-1/2) * A * D^(-1/2)
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        # 创建度矩阵的逆平方根
        degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
        
        # 归一化邻接矩阵
        adj_normalized = torch.matmul(torch.matmul(degree_matrix_inv_sqrt, adj), degree_matrix_inv_sqrt)
        
        return adj_normalized
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, sequence_length, num_features]
        Returns:
            output: 预测结果 [batch_size, num_nodes, 1]
        """
        batch_size, seq_len, num_features = x.shape
        device = x.device
        
        # 扩展到节点维度 [batch_size, num_nodes, sequence_length, num_features]
        x_expanded = x.unsqueeze(1).repeat(1, self.num_nodes, 1, 1)
        
        # 重塑为 [batch_size, num_nodes, seq_len * num_features]
        x_flat = x_expanded.view(batch_size, self.num_nodes, seq_len * num_features)
        
        # 输入特征投影
        x_proj = self.input_projection(x_flat)  # [batch_size, num_nodes, hidden_dim]
        
        # 创建邻接矩阵
        adj = self.create_adjacency_matrix(self.num_nodes, device)
        
        # 通过GCN层
        for gcn_layer in self.gcn_layers:
            x_proj = gcn_layer(x_proj, adj)
        
        # 重塑为时间序列格式进行LSTM处理
        # [batch_size * num_nodes, 1, hidden_dim]
        lstm_input = x_proj.view(batch_size * self.num_nodes, 1, self.hidden_dim)
        lstm_out, (h_n, c_n) = self.temporal_lstm(lstm_input)
        lstm_out = lstm_out.view(batch_size, self.num_nodes, self.hidden_dim)
        
        # 注意力机制
        # 重塑为 [seq_len, batch_size * num_nodes, hidden_dim] 用于注意力
        attn_input = lstm_out.view(1, batch_size * self.num_nodes, self.hidden_dim)
        attn_out, _ = self.attention(attn_input, attn_input, attn_input)
        attn_out = attn_out.view(batch_size, self.num_nodes, self.hidden_dim)
        
        # 预测输出
        output = self.output_layers(attn_out)  # [batch_size, num_nodes, 1]
        
        return output


class SimpleGCN(nn.Module):
    """简化版GCN模型，用于快速对比"""
    def __init__(self, 
                 num_features=7,
                 hidden_dim=64,
                 num_nodes=3,
                 sequence_length=30,
                 dropout=0.1):
        super(SimpleGCN, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        
        # 特征投影
        self.feature_projection = nn.Linear(sequence_length * num_features, hidden_dim)
        
        # GCN层
        self.gcn1 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim // 2)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        
    def create_adjacency_matrix(self, num_nodes, device):
        """
        创建并归一化邻接矩阵
        Args:
            num_nodes: 节点数量
            device: 设备
        Returns:
            adj_normalized: 归一化的邻接矩阵
        """
        # 创建全连接图的邻接矩阵
        adj = torch.ones(num_nodes, num_nodes, device=device)
        
        # 移除自环
        adj.fill_diagonal_(0)
        
        # 添加自环
        adj += torch.eye(num_nodes, device=device)
        
        # 计算度矩阵
        degree = torch.sum(adj, dim=1)
        degree_matrix_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
        
        # 归一化邻接矩阵
        adj_normalized = torch.matmul(torch.matmul(degree_matrix_inv_sqrt, adj), degree_matrix_inv_sqrt)
        
        return adj_normalized
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, sequence_length, num_features]
        Returns:
            output: 预测结果 [batch_size, num_nodes, 1]
        """
        batch_size, seq_len, num_features = x.shape
        device = x.device
        
        # 扩展到节点维度 [batch_size, num_nodes, sequence_length, num_features]
        x_expanded = x.unsqueeze(1).repeat(1, self.num_nodes, 1, 1)
        
        # 重塑为 [batch_size, num_nodes, seq_len * num_features]
        x_flat = x_expanded.view(batch_size, self.num_nodes, seq_len * num_features)
        
        # 输入特征投影
        x_proj = self.feature_projection(x_flat)  # [batch_size, num_nodes, hidden_dim]
        
        # 创建邻接矩阵
        adj = self.create_adjacency_matrix(self.num_nodes, device)
        
        # GCN层
        x_proj = F.relu(self.gcn1(x_proj, adj))
        x_proj = self.dropout(x_proj)
        x_proj = F.relu(self.gcn2(x_proj, adj))
        x_proj = self.dropout(x_proj)
        x_proj = F.relu(self.gcn3(x_proj, adj))
        
        # 输出预测
        output = self.output_layer(x_proj)  # [batch_size, num_nodes, 1]
        
        return output


def create_gcn_model(model_type='temporal', **kwargs):
    """
    创建GCN模型的工厂函数
    
    Args:
        model_type: 模型类型 ('temporal' 或 'simple')
        **kwargs: 模型参数
    
    Returns:
        GCN模型实例
    """
    if model_type == 'temporal':
        return TemporalGCN(**kwargs)
    elif model_type == 'simple':
        return SimpleGCN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 4
    sequence_length = 30
    num_nodes = 3
    num_features = 7
    
    x = torch.randn(batch_size, sequence_length, num_nodes, num_features).to(device)
    
    # 测试时间序列GCN
    print("Testing Temporal GCN...")
    temporal_gcn = TemporalGCN(
        num_features=num_features,
        hidden_dim=64,
        num_layers=3,
        num_nodes=num_nodes,
        sequence_length=sequence_length
    ).to(device)
    
    with torch.no_grad():
        output = temporal_gcn(x)
        print(f"Temporal GCN output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 测试简单GCN
    print("\nTesting Simple GCN...")
    simple_gcn = SimpleGCN(
        num_features=num_features,
        hidden_dim=64,
        num_nodes=num_nodes,
        sequence_length=sequence_length
    ).to(device)
    
    with torch.no_grad():
        output = simple_gcn(x)
        print(f"Simple GCN output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 计算参数数量
    temporal_params = sum(p.numel() for p in temporal_gcn.parameters())
    simple_params = sum(p.numel() for p in simple_gcn.parameters())
    
    print(f"\nModel Parameters:")
    print(f"Temporal GCN: {temporal_params:,}")
    print(f"Simple GCN: {simple_params:,}")