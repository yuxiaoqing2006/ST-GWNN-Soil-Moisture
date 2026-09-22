

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphWaveletConv(nn.Module):

    
    def __init__(self, in_channels, out_channels, K=3, bias=True):
       
        super(GraphWaveletConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
      
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj_matrix):
       
        batch_size, num_nodes, in_channels = x.shape
        
     
        degree_matrix = torch.sum(adj_matrix, dim=-1)
        degree_inv_sqrt = torch.pow(degree_matrix + 1e-6, -0.5)
        
   
        norm_adj = degree_inv_sqrt.unsqueeze(-1) * adj_matrix * degree_inv_sqrt.unsqueeze(-2)
        
   
        identity = torch.eye(num_nodes, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        norm_adj = norm_adj + identity
        
      
        out = torch.zeros(batch_size, num_nodes, self.out_channels, device=x.device)
        
        
        x_k = x
        for k in range(self.K):
        
            filtered = torch.matmul(x_k, self.weight[k])
            out += filtered
            
          
            if k < self.K - 1:
                x_k = torch.matmul(norm_adj, x_k)
        
        if self.bias is not None:
            out += self.bias
            
        return out

class AdaptiveGraphWavelet(nn.Module):

    
    def __init__(self, in_channels, out_channels, num_scales=3):
       
        super(AdaptiveGraphWavelet, self).__init__()
        
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        
      
        self.wavelet_convs = nn.ModuleList([
            GraphWaveletConv(in_channels, out_channels, K=k+1)
            for k in range(num_scales)
        ])
        
    
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
       
        self.output_proj = nn.Linear(out_channels, out_channels)
        
    def forward(self, x, adj_matrix):
       
        scale_outputs = []
        
    
        for i, conv in enumerate(self.wavelet_convs):
            scale_out = conv(x, adj_matrix)
            scale_outputs.append(scale_out)
        
     
        weights = F.softmax(self.scale_weights, dim=0)
        fused_output = sum(w * out for w, out in zip(weights, scale_outputs))
        
      
        output = self.output_proj(fused_output)
        
        return output

class MultiScaleGraphWavelet(nn.Module):
   
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, num_scales=3):
       
        super(MultiScaleGraphWavelet, self).__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
       
        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim
                
            if i == num_layers - 1:
                out_dim = output_dim
            else:
                out_dim = hidden_dim
                
            layer = AdaptiveGraphWavelet(in_dim, out_dim, num_scales)
            self.layers.append(layer)
        
     
        self.use_residual = (input_dim == output_dim)
        if not self.use_residual and num_layers > 1:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        
       
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim if i < num_layers-1 else output_dim)
            for i in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, adj_matrix):
       
        residual = x
        
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            x = layer(x, adj_matrix)
            x = norm(x)
            
           
            if i == self.num_layers - 1:  
                if self.use_residual:
                    x = x + residual
                elif hasattr(self, 'residual_proj'):
                    x = x + self.residual_proj(residual)
            
            if i < self.num_layers - 1: 
                x = F.relu(x)
                x = self.dropout(x)
        
        return x

def create_graph_adjacency(coordinates, k=5, sigma=1.0):
   
    num_nodes = coordinates.shape[0]
    
    
    dist_matrix = torch.cdist(coordinates, coordinates, p=2)
    
  
    _, knn_indices = torch.topk(dist_matrix, k+1, dim=1, largest=False)
    knn_indices = knn_indices[:, 1:] 
  
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        for j in knn_indices[i]:
       
            weight = torch.exp(-dist_matrix[i, j] ** 2 / (2 * sigma ** 2))
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight
    
    return adj_matrix


def test_graph_wavelet():
    
    print("测试图小波网络...")
    
   
    batch_size = 8
    num_nodes = 20
    input_dim = 64
    hidden_dim = 128
    output_dim = 32
    
   
    x = torch.randn(batch_size, num_nodes, input_dim)
    
  
    adj_matrix = torch.rand(batch_size, num_nodes, num_nodes)
    adj_matrix = (adj_matrix + adj_matrix.transpose(-1, -2)) / 2
    
  
    model = MultiScaleGraphWavelet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=3,
        num_scales=4
    )
    
 
    output = model(x, adj_matrix)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
   
    loss = output.sum()
    loss.backward()
    print("梯度计算正常")
    
    print("图小波网络测试完成！")

if __name__ == "__main__":
    test_graph_wavelet()