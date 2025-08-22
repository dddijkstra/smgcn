import numpy as np

# 读取 npy 文件
data = np.load("/Users/kevinz/ProjectRoot/Q/smgcn/data/KG/data/ss_graph.npy")
np.savetxt("/Users/kevinz/ProjectRoot/Q/smgcn/data/KG/symPair-5.txt", data, fmt="%d")
print(type(data))   # 看看是什么类型
print(data.shape)   # 看看维度
print(data.dtype)   # 数据类型
print(data)         # 打印内容（注意如果很大可能会刷屏）