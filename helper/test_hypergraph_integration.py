#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试超图功能集成
"""

import sys
import os
import torch
import numpy as np
import scipy.sparse as sp

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入参数解析器
from utils.parser import parse_args
from utils.load_data import Data

# 创建args参数
args = parse_args()
args.dataset = 'Herb'
args.use_hypergraph = True  # 启用超图功能
args.data_path = './data/'
args.batch_size = 1024
args.device = torch.device('cpu')
args.embed_size = 64
args.layer_size = '[64,64]'
args.lr = 0.001
args.decay = 1e-4
args.dropout = 0.1
args.keep_prob = 0.6
args.A_split = False
args.pretrain = 0
args.adj_type = 'norm'
args.regs = '[1e-5]'
args.mess_dropout = '[0.1, 0.1]'
args.node_dropout = '[0.1]'
args.fusion = 'add'
args.cl_weight = 0.1

# 设置设备
device = torch.device('cpu')

# 将args设置为全局变量，以便SMGCN模型可以访问
# 需要在utils.batch_test模块中设置args
import utils.batch_test
utils.batch_test.args = args

# 导入SMGCN模型
from model.SMGCN import SMGCN

# 全局变量
data_generator = None

def test_hypergraph_integration():
    """测试超图功能集成"""
    print("开始测试超图功能集成...")
    
    # 使用全局args
    global data_generator
    
    # 初始化数据生成器
    data_generator = Data(args=args, path=args.data_path + args.dataset, batch_size=args.batch_size)
    
    # 获取邻接矩阵
    adj_mat, norm_adj_mat, mean_adj_mat, sym_pair_adj_mat, herb_pair_adj_mat = data_generator.get_adj_mat()
    
    # 加载数据
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['norm_adj'] = norm_adj_mat
    config['herb_pair_adj'] = herb_pair_adj_mat
    config['sym_pair_adj'] = sym_pair_adj_mat
    
    try:
        # 创建模型
        model = SMGCN(data_config=config, pretrain_data=None)
        model = model.to(args.device)
        
        print(f"模型创建成功")
        print(f"使用超图: {model.use_hypergraph}")
        
        if model.use_hypergraph:
            print(f"超图邻接矩阵形状: {model.hypergraph_adj.shape if model.hypergraph_adj is not None else 'None'}")
            print(f"训练数据数量: {len(model.train_data) if hasattr(model, 'train_data') else 'None'}")
            print(f"草药配伍数据数量: {len(model.herb_pairs) if hasattr(model, 'herb_pairs') else 'None'}")
        
        # 创建测试数据
        batch_size = 32
        n_users = config['n_users']
        n_items = config['n_items']
        
        # 模拟用户-物品交互矩阵
        users = torch.zeros(batch_size, n_users).to(device)
        for i in range(batch_size):
            # 随机选择一些用户
            user_indices = torch.randint(0, n_users, (3,))
            users[i, user_indices] = 1.0
        
        # 模拟用户集合
        user_set = torch.randint(0, n_users, (batch_size,)).to(device)
        
        print("\n开始前向传播测试...")
        
        # 训练模式测试
        model.train()
        try:
            result = model.forward(users, user_set, train=True)
            if model.use_hypergraph:
                user_embeddings, all_user_embeddins, i_g_embeddings, cl_loss_user, cl_loss_item, cl_loss_hypergraph = result
                print(f"训练模式 - 用户嵌入形状: {user_embeddings.shape}")
                print(f"训练模式 - 物品嵌入形状: {i_g_embeddings.shape}")
                print(f"训练模式 - 用户对比损失: {cl_loss_user.item():.4f}")
                print(f"训练模式 - 物品对比损失: {cl_loss_item.item():.4f}")
                print(f"训练模式 - 超图对比损失: {cl_loss_hypergraph.item():.4f}")
            else:
                user_embeddings, all_user_embeddins, i_g_embeddings, cl_loss_user, cl_loss_item = result
                print(f"训练模式 - 用户嵌入形状: {user_embeddings.shape}")
                print(f"训练模式 - 物品嵌入形状: {i_g_embeddings.shape}")
                print(f"训练模式 - 用户对比损失: {cl_loss_user.item():.4f}")
                print(f"训练模式 - 物品对比损失: {cl_loss_item.item():.4f}")
        except Exception as e:
            print(f"训练模式测试失败: {e}")
            return False
        
        # 测试模式测试
        model.eval()
        try:
            pos_items = list(range(min(10, n_items)))  # 选择前10个物品
            result = model.forward(users, user_set, pos_items, train=False)
            user_embeddings, pos_i_g_embeddings = result
            print(f"测试模式 - 用户嵌入形状: {user_embeddings.shape}")
            print(f"测试模式 - 正样本物品嵌入形状: {pos_i_g_embeddings.shape}")
        except Exception as e:
            print(f"测试模式测试失败: {e}")
            return False
        
        print("\n超图功能集成测试成功！")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # 运行测试
    success = test_hypergraph_integration()
    if success:
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 测试失败！")
        sys.exit(1)