import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 1. 超图神经网络层
class SimpleHGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, H):
        Dv_diag = H.sum(1)
        De_diag = H.sum(0)
        Dv_inv = torch.zeros_like(Dv_diag)
        De_inv = torch.zeros_like(De_diag)
        Dv_inv[Dv_diag != 0] = 1.0 / Dv_diag[Dv_diag != 0]
        De_inv[De_diag != 0] = 1.0 / De_diag[De_diag != 0]
        Dv_inv_mat = torch.diag(Dv_inv)
        De_inv_mat = torch.diag(De_inv)
        X = torch.mm(Dv_inv_mat, torch.mm(H, torch.mm(De_inv_mat, torch.mm(H.t(), X))))
        return self.linear(X)

class SimpleHGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SimpleHGNNLayer(in_dim, hidden_dim))
        for _ in range(num_layers-2):
            self.layers.append(SimpleHGNNLayer(hidden_dim, hidden_dim))
        self.layers.append(SimpleHGNNLayer(hidden_dim, out_dim))

    def forward(self, X, H):
        for layer in self.layers[:-1]:
            X = F.relu(layer(X, H))
        X = self.layers[-1](X, H)
        return X

# 2. 构建超图关联矩阵

def build_hypergraph_incidence(num_nodes, hyperedges):
    H = np.zeros((num_nodes, len(hyperedges)), dtype=np.float32)
    for e_idx, nodes in enumerate(hyperedges):
        for n in nodes:
            H[n, e_idx] = 1.0
    return torch.tensor(H)

# 3. 读取train.txt并生成超边

def parse_train_txt(file_path, symptom_max_id, herb_max_id):
    hyperedges = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            if '\t' not in line:
                continue
            sym_str, herb_str = line.strip().split('\t')
            sym_ids = [int(x) for x in sym_str.split()]
            herb_ids = [int(x) + symptom_max_id + 1 for x in herb_str.split()]
            hyperedges.append(sym_ids + herb_ids)
    num_nodes = symptom_max_id + 1 + herb_max_id + 1
    return hyperedges, num_nodes

# 4. 节点特征初始化

def get_node_features(num_nodes, feature_dim=None):
    if feature_dim is None or feature_dim == num_nodes:
        return torch.eye(num_nodes)
    else:
        return torch.randn(num_nodes, feature_dim)

if __name__ == "__main__":
    # 你可以根据实际情况修改最大ID
    symptom_max_id = 360
    herb_max_id = 753
    train_path = "data/Herb/train.txt"

    # 1. 读取train.txt，生成超边和节点总数
    hyperedges, num_nodes = parse_train_txt(train_path, symptom_max_id, herb_max_id)

    # 2. 构建超图关联矩阵
    H = build_hypergraph_incidence(num_nodes, hyperedges)

    # 3. 初始化节点特征
    X = get_node_features(num_nodes)

    # 4. 构建HGNN
    hgnn = SimpleHGNN(in_dim=num_nodes, hidden_dim=64, out_dim=256, num_layers=2)

    # 5. 前向传播，得到节点向量
    node_embeddings = hgnn(X, H)

    print("节点嵌入 shape:", node_embeddings.shape)
    # 保存为npy文件，便于后续加载
    np.save("data/Herb/hypergraph_node_embeddings.npy", node_embeddings.detach().cpu().numpy())