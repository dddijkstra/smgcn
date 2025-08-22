# import pandas as pd

# # 读取 csv
# df = pd.read_csv("/Users/kevinz/ProjectRoot/Q/smgcn/data/KG/all.txt")

# # 保存为 txt，默认空格分隔
# df.to_csv("/Users/kevinz/ProjectRoot/Q/smgcn/data/KG/train.txt", sep="\t", index=False, header=False)
with open("/Users/kevinz/ProjectRoot/Q/smgcn/data/KG/all.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 分割
part1 = lines[:20259]
part2 = lines[20259 : 20259 + 13056]

# 写入新文件
with open("train.txt", "w", encoding="utf-8") as f:
    f.writelines(part1)

with open("test.txt", "w", encoding="utf-8") as f:
    f.writelines(part2)
