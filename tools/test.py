import hypernetx as hnx
import matplotlib.pyplot as plt

# 读取herb和symptom映射
def load_mapping(filename):
    mapping = {}
    with open(filename, encoding='utf-8') as f:
        for line in f:
            name, idx = line.strip().split()
            mapping[int(idx)] = name
    return mapping

herb_mapping = load_mapping('data/Herb/herb_mapping.txt')
symptom_mapping = load_mapping('data/Herb/sym_mapping.txt')

# 读取处方数据
edges = {}
with open('data/Herb/train.txt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        symptoms, herbs = line.strip().split('\t')
        symptom_ids = [int(x) for x in symptoms.split()]
        herb_ids = [int(x) for x in herbs.split()]
        # 超边的名字可以用处方编号
        edge_name = f'prescription_{i}'
        # 超边的节点集合：症状和中药都作为节点
        nodes = [f's_{sid}' for sid in symptom_ids] + [f'h_{hid}' for hid in herb_ids]
        edges[edge_name] = set(nodes)

# 构建超图
H = hnx.Hypergraph(edges)

# 可选：只画一个小子图（比如前10个处方），否则节点太多会很乱
sub_edges = dict(list(edges.items())[:10])
H_sub = hnx.Hypergraph(sub_edges)

# 绘制超图
plt.figure(figsize=(10, 8))
hnx.drawing.draw(H_sub, with_node_labels=True, with_edge_labels=True)
plt.title("Herb-Symptom Hypergraph (Sample)")
plt.savefig("herb_hypergraph.png", dpi=300)