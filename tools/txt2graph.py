from py2neo import Graph, Node, Relationship, NodeMatcher

# 连接数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "java0918"))
matcher = NodeMatcher(graph)

# 清空数据库
graph.delete_all()

# 读取映射函数
def load_mapping(file_path):
    mapping = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            name, idx = line.strip().split()
            mapping[int(idx)] = name
    return mapping

# 加载节点映射
herb_map = load_mapping("data/Herb/herb_mapping.txt")
sym_map = load_mapping("data/Herb/sym_mapping.txt")

# 创建节点
for sid, sname in sym_map.items():
    graph.merge(Node("Symptom", id=sid, name=sname), "Symptom", "id")

for hid, hname in herb_map.items():
    graph.merge(Node("Herb", id=hid, name=hname), "Herb", "id")

# 导入症状-草药关系
with open("data/Herb/train.txt", encoding='utf-8') as f:
    for line in f:
        if '\t' not in line: continue
        sym_str, herb_str = line.strip().split('\t')
        sym_ids = [int(x) for x in sym_str.split()]
        herb_ids = [int(x) for x in herb_str.split()]
        for sid in sym_ids:
            for hid in herb_ids:
                s_node = matcher.match("Symptom", id=sid).first()
                h_node = matcher.match("Herb", id=hid).first()
                if s_node and h_node:
                    graph.merge(Relationship(s_node, "HAS_HERB", h_node))

# 导入症状-症状相似关系
with open("data/Herb/symPair-5.txt", encoding='utf-8') as f:
    for line in f:
        s1, s2 = [int(x) for x in line.strip().split()]
        n1 = matcher.match("Symptom", id=s1).first()
        n2 = matcher.match("Symptom", id=s2).first()
        if n1 and n2:
            graph.merge(Relationship(n1, "SYM_SIMILAR", n2))

# 导入草药-草药相似关系
with open("data/Herb/herbPair-40.txt", encoding='utf-8') as f:
    for line in f:
        h1, h2 = [int(x) for x in line.strip().split()]
        n1 = matcher.match("Herb", id=h1).first()
        n2 = matcher.match("Herb", id=h2).first()
        if n1 and n2:
            graph.merge(Relationship(n1, "HERB_SIMILAR", n2))

print("✅ 数据导入完成！")