# -*- coding: utf-8 -*-
# @Time    : 2024/01/15 11:00
# @Author  : Assistant
# @File    : HerbHyperGraphNet.py
# @Description : 专门用于草药推荐的超图卷积神经网络

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix
import math
from .HyperGraphConv import HyperGraphConvolution, HyperGraphNeuralNetwork, HyperGraphBuilder

class HerbHyperGraphConv(nn.Module):
    """
    专门为草药推荐设计的超图卷积层
    考虑症状-草药的复杂关系和中医理论
    """
    def __init__(self, in_features, out_features, herb_features, sym_features, 
                 fusion_method='attention', dropout=0.0):
        super(HerbHyperGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.herb_features = herb_features
        self.sym_features = sym_features
        self.fusion_method = fusion_method
        self.dropout = dropout
        
        # 基础超图卷积
        self.hypergraph_conv = HyperGraphConvolution(in_features, out_features, dropout=dropout)
        
        # 草药特征变换
        self.herb_transform = nn.Linear(herb_features, out_features)
        
        # 症状特征变换
        self.sym_transform = nn.Linear(sym_features, out_features)
        
        # 融合机制
        if fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(out_features, num_heads=4, dropout=dropout)
        elif fusion_method == 'gate':
            self.gate = nn.Sequential(
                nn.Linear(out_features * 3, out_features),
                nn.Sigmoid()
            )
        
        # 中医理论约束层
        self.tcm_constraint = TCMConstraintLayer(out_features)
        
    def forward(self, x, hypergraph_adj, herb_features=None, sym_features=None):
        """
        前向传播
        Args:
            x: 基础节点特征 [N, in_features]
            hypergraph_adj: 超图邻接矩阵
            herb_features: 草药特征 [N_herbs, herb_features]
            sym_features: 症状特征 [N_syms, sym_features]
        """
        # 基础超图卷积
        h_base = self.hypergraph_conv(x, hypergraph_adj)
        
        # 如果有额外特征，进行融合
        if herb_features is not None and sym_features is not None:
            h_herb = self.herb_transform(herb_features)
            h_sym = self.sym_transform(sym_features)
            
            if self.fusion_method == 'attention':
                # 使用注意力机制融合
                h_fused, _ = self.attention(h_base.unsqueeze(0), 
                                          torch.cat([h_herb, h_sym], dim=0).unsqueeze(0),
                                          torch.cat([h_herb, h_sym], dim=0).unsqueeze(0))
                h_fused = h_fused.squeeze(0)
            elif self.fusion_method == 'gate':
                # 使用门控机制融合
                h_concat = torch.cat([h_base, h_herb[:h_base.size(0)], h_sym[:h_base.size(0)]], dim=1)
                gate = self.gate(h_concat)
                h_fused = gate * h_base + (1 - gate) * (h_herb[:h_base.size(0)] + h_sym[:h_base.size(0)]) / 2
            else:
                # 简单加权融合
                h_fused = h_base + 0.3 * h_herb[:h_base.size(0)] + 0.3 * h_sym[:h_base.size(0)]
        else:
            h_fused = h_base
            
        # 应用中医理论约束
        h_constrained = self.tcm_constraint(h_fused)
        
        return h_constrained

class TCMConstraintLayer(nn.Module):
    """
    中医理论约束层
    基于中医的君臣佐使理论和配伍禁忌
    """
    def __init__(self, features):
        super(TCMConstraintLayer, self).__init__()
        self.features = features
        
        # 君臣佐使权重
        self.role_weights = nn.Parameter(torch.FloatTensor(4, features))
        
        # 配伍约束矩阵
        self.compatibility_matrix = nn.Parameter(torch.FloatTensor(features, features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.role_weights)
        nn.init.xavier_uniform_(self.compatibility_matrix)
        
    def forward(self, x):
        """
        应用中医理论约束
        """
        # 君臣佐使约束
        role_constraint = torch.matmul(x, self.role_weights.t())  # [N, 4]
        role_weights = F.softmax(role_constraint, dim=1)  # 归一化权重
        
        # 配伍约束
        compatibility = torch.matmul(x, self.compatibility_matrix)
        compatibility = torch.sigmoid(compatibility)  # 配伍相容性
        
        # 结合约束
        constrained_x = x * compatibility
        
        return constrained_x

class HerbHyperGraphNet(nn.Module):
    """
    完整的草药超图推荐网络
    """
    def __init__(self, config):
        super(HerbHyperGraphNet, self).__init__()
        
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.emb_dim = config['emb_dim']
        self.hidden_dims = config.get('hidden_dims', [128, 64])
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.5)
        self.fusion_method = config.get('fusion_method', 'attention')
        
        # 基础嵌入
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)
        
        # 超图卷积层
        self.hypergraph_layers = nn.ModuleList()
        dims = [self.emb_dim] + self.hidden_dims
        
        for i in range(self.num_layers):
            layer = HerbHyperGraphConv(
                in_features=dims[i],
                out_features=dims[i+1],
                herb_features=self.emb_dim,
                sym_features=self.emb_dim,
                fusion_method=self.fusion_method,
                dropout=self.dropout
            )
            self.hypergraph_layers.append(layer)
            
        # 预测层 - 修复维度匹配问题
        # 输入维度应该是 emb_dim + dims[-1] (用户嵌入 + 超图卷积后的物品嵌入)
        predictor_input_dim = self.emb_dim + dims[-1]
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, dims[-1]),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(dims[-1], 1),
            nn.Sigmoid()
        )
        
        # 对比学习相关
        self.temperature = 0.07
        self.cl_weight = config.get('cl_weight', 0.1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def build_herb_hypergraph(self, sym_herb_pairs, herb_herb_pairs):
        """
        构建草药超图
        Args:
            sym_herb_pairs: 症状-草药对 [(sym_id, herb_id), ...]
            herb_herb_pairs: 草药-草药对 [(herb1_id, herb2_id), ...]
        Returns:
            hypergraph_adj: 超图邻接矩阵
        """
        # 创建超边
        hyperedges = []
        
        # 基于症状创建超边（同一症状对应的所有草药形成一个超边）
        sym_to_herbs = {}
        for sym_id, herb_id in sym_herb_pairs:
            if sym_id not in sym_to_herbs:
                sym_to_herbs[sym_id] = []
            sym_to_herbs[sym_id].append(herb_id)
            
        for sym_id, herbs in sym_to_herbs.items():
            if len(herbs) > 1:  # 超边至少包含2个节点
                hyperedges.append(herbs)
                
        # 基于草药配伍创建超边
        herb_groups = {}
        for herb1_id, herb2_id in herb_herb_pairs:
            # 简化处理：每对草药形成一个超边
            hyperedges.append([herb1_id, herb2_id])
            
        # 构建关联矩阵
        n_nodes = self.n_items  # 假设草药是items
        n_edges = len(hyperedges)
        
        H = np.zeros((n_nodes, n_edges))
        for e_idx, edge in enumerate(hyperedges):
            for node in edge:
                if node < n_nodes:  # 确保索引有效
                    H[node, e_idx] = 1.0
                    
        return self._compute_hypergraph_laplacian(H)
    
    def _compute_hypergraph_laplacian(self, H):
        """计算超图拉普拉斯矩阵"""
        n_nodes, n_edges = H.shape
        
        # 计算度矩阵
        D_v = np.diag(np.sum(H, axis=1))
        D_e = np.diag(np.sum(H, axis=0))
        
        # 避免除零
        D_v_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_v) + 1e-8))
        D_e_inv = np.diag(1.0 / (np.diag(D_e) + 1e-8))
        
        # 计算归一化拉普拉斯矩阵
        L = D_v_inv_sqrt @ H @ D_e_inv @ H.T @ D_v_inv_sqrt
        
        # 转换为稀疏张量
        L_coo = coo_matrix(L)
        indices = torch.LongTensor([L_coo.row, L_coo.col])
        values = torch.FloatTensor(L_coo.data)
        shape = L_coo.shape
        
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def forward(self, user_ids, item_ids, hypergraph_adj, train=True):
        """
        前向传播
        Args:
            user_ids: 用户ID [batch_size]
            item_ids: 物品ID [batch_size]
            hypergraph_adj: 超图邻接矩阵
            train: 是否训练模式
        """
        # 获取嵌入
        user_emb = self.user_embedding(user_ids)  # [batch_size, emb_dim]
        item_emb = self.item_embedding(item_ids)  # [batch_size, emb_dim]
        
        # 获取所有物品嵌入用于超图卷积
        all_item_emb = self.item_embedding.weight  # [n_items, emb_dim]
        
        # 超图卷积
        h = all_item_emb
        for layer in self.hypergraph_layers:
            h = layer(h, hypergraph_adj)
            h = F.relu(h)
            
        # 获取当前批次的物品嵌入
        item_emb_hypergraph = h[item_ids]  # [batch_size, hidden_dim]
        
        # 预测
        combined = torch.cat([user_emb, item_emb_hypergraph], dim=1)
        scores = self.predictor(combined).squeeze(-1)
        
        if train:
            # 计算对比学习损失
            cl_loss = self.compute_contrastive_loss(item_emb, item_emb_hypergraph)
            return scores, cl_loss
        else:
            return scores
    
    def compute_contrastive_loss(self, view1, view2):
        """
        计算对比学习损失
        """
        # 归一化
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        # 计算相似度
        similarity = torch.matmul(view1, view2.t()) / self.temperature
        
        # 标签
        labels = torch.arange(similarity.size(0)).to(similarity.device)
        
        # 对比损失
        loss = F.cross_entropy(similarity, labels)
        
        return loss

class HerbDataProcessor:
    """
    草药数据处理工具
    """
    @staticmethod
    def load_herb_data(data_path):
        """
        加载草药数据
        """
        # 读取症状-草药对
        sym_herb_pairs = []
        with open(f"{data_path}/symPair-5.txt", 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    sym_id, herb_id = int(parts[0]), int(parts[1])
                    sym_herb_pairs.append((sym_id, herb_id))
                    
        # 读取草药-草药对
        herb_herb_pairs = []
        with open(f"{data_path}/herbPair-40.txt", 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    herb1_id, herb2_id = int(parts[0]), int(parts[1])
                    herb_herb_pairs.append((herb1_id, herb2_id))
                    
        return sym_herb_pairs, herb_herb_pairs
    
    @staticmethod
    def create_herb_features(herb_mapping_path, feature_dim=64):
        """
        创建草药特征
        """
        # 读取草药映射
        herb_names = []
        with open(herb_mapping_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    herb_names.append(parts[1])
                    
        # 简化：使用随机特征（实际应用中可以使用草药的属性特征）
        n_herbs = len(herb_names)
        herb_features = torch.randn(n_herbs, feature_dim)
        
        return herb_features, herb_names

# 使用示例
if __name__ == "__main__":
    # 配置
    config = {
        'n_users': 1000,
        'n_items': 500,
        'emb_dim': 64,
        'hidden_dims': [128, 64],
        'num_layers': 2,
        'dropout': 0.5,
        'fusion_method': 'attention',
        'cl_weight': 0.1
    }
    
    # 创建模型
    model = HerbHyperGraphNet(config)
    
    # 模拟数据
    user_ids = torch.randint(0, config['n_users'], (32,))
    item_ids = torch.randint(0, config['n_items'], (32,))
    
    # 模拟超图邻接矩阵
    hypergraph_adj = torch.sparse.FloatTensor(
        torch.LongTensor([[0, 1], [1, 0]]),
        torch.FloatTensor([1.0, 1.0]),
        (config['n_items'], config['n_items'])
    )
    
    # 前向传播
    scores, cl_loss = model(user_ids, item_ids, hypergraph_adj, train=True)
    
    print(f"预测分数形状: {scores.shape}")
    print(f"对比学习损失: {cl_loss.item():.4f}")
    print("草药超图神经网络创建成功！")