# -*- coding: utf-8 -*-
# @Time    : 2024/01/15 10:00
# @Author  : Assistant
# @File    : HyperGraphConv.py
# @Description : 超图卷积神经网络实现

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix
import math

class HyperGraphConvolution(nn.Module):
    """
    超图卷积层实现
    基于论文: "Hypergraph Neural Networks" (AAAI 2019)
    """
    def __init__(self, in_features, out_features, bias=True, dropout=0.0):
        super(HyperGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        
        # 权重矩阵
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化参数"""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, x, hypergraph_adj):
        """
        前向传播
        Args:
            x: 节点特征矩阵 [N, in_features]
            hypergraph_adj: 超图邻接矩阵 [N, N]
        Returns:
            output: 输出特征矩阵 [N, out_features]
        """
        # 应用dropout
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 线性变换
        support = torch.mm(x, self.weight)
        
        # 超图卷积操作
        output = torch.sparse.mm(hypergraph_adj, support)
        
        if self.bias is not None:
            output = output + self.bias
            
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class HyperGraphAttentionConv(nn.Module):
    """
    带注意力机制的超图卷积层
    """
    def __init__(self, in_features, out_features, num_heads=1, dropout=0.0, alpha=0.2):
        super(HyperGraphAttentionConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        
        # 多头注意力
        self.head_dim = out_features // num_heads
        assert self.head_dim * num_heads == out_features
        
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2 * self.head_dim, 1))
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
    def forward(self, x, hyperedge_index):
        """
        Args:
            x: 节点特征 [N, in_features]
            hyperedge_index: 超边索引 [2, num_edges] 或 [num_nodes, num_hyperedges]
        """
        N = x.size(0)
        
        # 线性变换
        h = torch.mm(x, self.W)  # [N, out_features]
        h = h.view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        
        # 计算注意力权重
        if hyperedge_index.dim() == 2 and hyperedge_index.size(0) == 2:
            # COO格式的超边索引
            edge_h = self._compute_attention_coo(h, hyperedge_index)
        else:
            # 邻接矩阵格式
            edge_h = self._compute_attention_adj(h, hyperedge_index)
            
        # 聚合多头结果
        output = edge_h.view(N, -1)  # [N, out_features]
        
        return output
    
    def _compute_attention_coo(self, h, hyperedge_index):
        """使用COO格式计算注意力"""
        # 实现超边内的注意力机制
        # 这里简化实现，实际应用中需要更复杂的超边聚合
        return h.mean(dim=1)  # 简化版本
    
    def _compute_attention_adj(self, h, adj):
        """使用邻接矩阵计算注意力"""
        # 简化的注意力计算
        return torch.sparse.mm(adj, h.view(h.size(0), -1)).view(h.size())

class HyperGraphNeuralNetwork(nn.Module):
    """
    完整的超图神经网络模型
    """
    def __init__(self, input_dim, hidden_dims, output_dim, num_layers=2, 
                 dropout=0.5, use_attention=False, num_heads=1):
        super(HyperGraphNeuralNetwork, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        # 构建层
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            if use_attention:
                layer = HyperGraphAttentionConv(
                    dims[i], dims[i+1], num_heads=num_heads, dropout=dropout
                )
            else:
                layer = HyperGraphConvolution(
                    dims[i], dims[i+1], dropout=dropout
                )
            self.layers.append(layer)
            
        # 批归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(dims[i+1]) for i in range(num_layers)
        ])
        
    def forward(self, x, hypergraph_adj):
        """
        前向传播
        Args:
            x: 节点特征 [N, input_dim]
            hypergraph_adj: 超图邻接矩阵或超边索引
        Returns:
            output: 节点嵌入 [N, output_dim]
        """
        h = x
        
        for i, layer in enumerate(self.layers):
            h = layer(h, hypergraph_adj)
            
            # 批归一化
            if i < len(self.batch_norms):
                h = self.batch_norms[i](h)
            
            # 激活函数（最后一层除外）
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, self.dropout, training=self.training)
                
        return h

class HyperGraphBuilder:
    """
    超图构建工具类
    """
    @staticmethod
    def build_hypergraph_from_data(data, method='knn', k=5, threshold=0.5):
        """
        从数据构建超图
        Args:
            data: 输入数据 [N, features]
            method: 构建方法 ('knn', 'threshold', 'clustering')
            k: KNN中的邻居数量
            threshold: 阈值方法中的相似度阈值
        Returns:
            hypergraph_adj: 超图邻接矩阵
        """
        if method == 'knn':
            return HyperGraphBuilder._build_knn_hypergraph(data, k)
        elif method == 'threshold':
            return HyperGraphBuilder._build_threshold_hypergraph(data, threshold)
        elif method == 'clustering':
            return HyperGraphBuilder._build_clustering_hypergraph(data, k)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def _build_knn_hypergraph(data, k):
        """基于KNN构建超图"""
        from sklearn.neighbors import NearestNeighbors
        
        # 计算KNN
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(data)
        distances, indices = nbrs.kneighbors(data)
        
        n_nodes = data.shape[0]
        n_edges = n_nodes
        
        # 构建关联矩阵 H [n_nodes, n_edges]
        H = np.zeros((n_nodes, n_edges))
        for i in range(n_nodes):
            for j in indices[i]:  # 包括自己
                H[j, i] = 1.0
                
        return HyperGraphBuilder._compute_hypergraph_laplacian(H)
    
    @staticmethod
    def _build_threshold_hypergraph(data, threshold):
        """基于阈值构建超图"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 计算相似度矩阵
        similarity = cosine_similarity(data)
        
        # 基于阈值创建超边
        n_nodes = data.shape[0]
        hyperedges = []
        
        for i in range(n_nodes):
            edge = [j for j in range(n_nodes) if similarity[i, j] > threshold]
            if len(edge) > 1:  # 超边至少包含2个节点
                hyperedges.append(edge)
        
        # 构建关联矩阵
        n_edges = len(hyperedges)
        H = np.zeros((n_nodes, n_edges))
        
        for e_idx, edge in enumerate(hyperedges):
            for node in edge:
                H[node, e_idx] = 1.0
                
        return HyperGraphBuilder._compute_hypergraph_laplacian(H)
    
    @staticmethod
    def _build_clustering_hypergraph(data, n_clusters):
        """基于聚类构建超图"""
        from sklearn.cluster import KMeans
        
        # 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        
        n_nodes = data.shape[0]
        n_edges = n_clusters
        
        # 构建关联矩阵
        H = np.zeros((n_nodes, n_edges))
        for i, label in enumerate(labels):
            H[i, label] = 1.0
            
        return HyperGraphBuilder._compute_hypergraph_laplacian(H)
    
    @staticmethod
    def _compute_hypergraph_laplacian(H):
        """
        计算超图拉普拉斯矩阵
        Args:
            H: 关联矩阵 [n_nodes, n_edges]
        Returns:
            L: 归一化的超图拉普拉斯矩阵
        """
        n_nodes, n_edges = H.shape
        
        # 计算度矩阵
        D_v = np.diag(np.sum(H, axis=1))  # 节点度矩阵
        D_e = np.diag(np.sum(H, axis=0))  # 超边度矩阵
        
        # 避免除零
        D_v_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_v) + 1e-8))
        D_e_inv = np.diag(1.0 / (np.diag(D_e) + 1e-8))
        
        # 计算归一化拉普拉斯矩阵
        # L = D_v^(-1/2) * H * D_e^(-1) * H^T * D_v^(-1/2)
        L = D_v_inv_sqrt @ H @ D_e_inv @ H.T @ D_v_inv_sqrt
        
        # 转换为稀疏张量
        L_coo = coo_matrix(L)
        indices = torch.LongTensor([L_coo.row, L_coo.col])
        values = torch.FloatTensor(L_coo.data)
        shape = L_coo.shape
        
        return torch.sparse.FloatTensor(indices, values, shape)

# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    n_nodes = 100
    input_dim = 64
    hidden_dims = [128, 64]
    output_dim = 32
    
    # 随机节点特征
    x = torch.randn(n_nodes, input_dim)
    
    # 构建超图
    builder = HyperGraphBuilder()
    hypergraph_adj = builder.build_hypergraph_from_data(
        x.numpy(), method='knn', k=5
    )
    
    # 创建超图神经网络
    model = HyperGraphNeuralNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        num_layers=2,
        dropout=0.5,
        use_attention=False
    )
    
    # 前向传播
    output = model(x, hypergraph_adj)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("超图卷积神经网络创建成功！")