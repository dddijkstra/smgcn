Collecting workspace information
# SMGCN-torch 项目结构说明

这是一个基于 PyTorch 实现的图卷积神经网络推荐系统项目。以下是主要目录和文件的说明：

### 主要目录

- **model** - 存放核心模型实现
  - 包含 SMGCN (Symptom-aware Multi-modal Graph Convolutional Network) 的主要实现

- **utils** - 工具函数目录
  - `batch_test.py` - 包含测试核心代码，实现评估指标计算
  - `parser.py` - 命令行参数解析
  - `load_data.py` - 数据加载相关函数

- **data** - 数据目录
  - 存放训练和测试所需的数据集

- **output** - 输出目录
  - 存放模型运行结果和日志

- **weights** - 模型权重
  - 保存训练好的模型权重文件

### 核心文件

- **smgcn_main.py** - 项目主入口文件
  - 实现模型训练和测试的主流程

- **SMGCN.sh** - Shell 脚本
  - 用于运行实验和配置环境

### 其他重要文件

- **.gitignore** - Git 忽略配置
- **README.md** - 项目说明文档
- **herb_hypergraph.png** - 模型结构图
- 各种日志文件 (`.log`)

### 辅助目录

- **.idea** - PyCharm IDE 配置
- **.vscode** - VS Code 配置
- **AdaDR** - 相关子项目
- **tools** - 辅助工具脚本

该项目主要用于中医症状-药材推荐系统，使用图卷积神经网络处理多模态数据，实现症状到药材的精准推荐。主要评估指标包括 precision、recall、NDCG 和 RMRR。


# SMGCN 模型架构解析

这张图展示了 SMGCN (Symptom-aware Multi-modal Graph Convolutional Network) 的整体架构，主要包含两个核心层：

## 1. Multi-Graph Embedding Layer (多图嵌入层)

### Bipar-GCN 部分包含

- **Symptom-oriented GCN**：
  - 处理症状节点 {s₁, s₂, ..., sₖ}
  - 通过 Embedding Propagation 进行特征传播
  - 输出症状嵌入 e*s₁ 到 e*sₖ

- **Herb-oriented GCN**：
  - 处理药材节点 {h₁, h₂, ..., hₙ}
  - 同样通过 Embedding Propagation 进行特征传播
  - 输出药材嵌入 e*h₁ 到 e*hₙ

### 图结构

- **Symptom-Symptom Graph**：症状之间的关联图
- **Herb-Herb Graph**：药材之间的关联图
- **SGE (Synergy Graph Encoding)**：协同图编码

## 2. Syndrome-aware Prediction Layer (综合预测层)

### 主要组件

1. **Fusion 模块**：
   - 融合症状特征 (rs, bs)
   - 融合药材特征 (rh, bh)

2. **Syndrome Induction**：
   - 生成症候相关表示 esyndrome(SC)

3. **Stack**：
   - 堆叠 N 个药材嵌入 eH

### 输出

- 最终通过 Multilabel Loss 优化
- 输出预测结果 ŷsc

## 关键特点

- 采用双向图卷积架构
- 考虑症状和药材的协同关系
- 引入症候感知机制
- 使用多标签损失函数

这个架构设计有效地整合了中医症状、药材和症候之间的复杂关系，用于中医药推荐任务。
