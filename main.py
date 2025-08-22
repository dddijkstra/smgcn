# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8，保证中文注释正常显示
# @Time    : 2022/1/17 16:22  # 文件创建/修改时间，仅作记录
# @Author  : Ywj  # 作者信息
# @File    : smgcn_main.py  # 文件名（历史命名），当前文件为项目主入口
# @Description :  SMGCN主函数  # 说明：本文件为 SMGCN 模型的训练与评测主流程脚本

import datetime  # 导入日期时间模块，用于打印起止时间和计时
import os  # 导入操作系统接口模块，用于环境变量设置、路径处理等
import sys  # 导入系统模块，用于异常退出等

import numpy as np  # 数值计算库，用于数组/矩阵运算和数据读写
import torch  # PyTorch 主库，提供张量和自动求导等
import torch.optim as optim  # PyTorch 优化器模块，这里使用 Adam

from model.SMGCN import SMGCN  # 从模型目录导入 SMGCN 模型类
from utils.batch_test import args, data_generator, test  # 导入命令行参数,args；数据加载器,data_generator；测试函数,test
from utils.helper import ensureDir, no_early_stopping  # 导入工具函数：创建目录ensureDir、早停判断no_early_stopping
from utils.load_data import sp, time  # 导入稀疏矩阵库别名sp，以及计时函数time


def load_pretrained_data():  # 定义函数：加载预训练的嵌入向量
    pretrain_path = "%spretrain/%s/%s.npz" % (args.proj_path, args.dataset, "embedding")  # 构造预训练权重文件路径
    try:  # 尝试读取预训练数据
        pretrain_data = np.load(pretrain_path)  # 使用 numpy 加载 .npz 格式文件
        print("load the pretrained embeddings.")  # 成功读取时提示
    except Exception:  # 读取失败（文件不存在或损坏等）
        pretrain_data = None  # 不使用预训练
    return pretrain_data  # 返回预训练数据或 None


if __name__ == "__main__":  # 仅当作为脚本直接运行时执行主流程
    startTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 记录开始时间（格式化字符串）
    print("start ", startTime)  # 打印开始时间
    print(  # 打印当前结果索引（用于结果文件区分）
        "result_index ",
        args.result_index,  # 打印结果的保存路径索引 例如：result_index=0 result_dir=xxx0.log
    )
    if torch.cuda.is_available():  # 判断是否有可用的 CUDA GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(  # 设置可见 GPU 设备（在多 GPU 环境下指定使用哪块）
            args.gpu_id
        )  # 多GPU环境下，指定程序使用某个GPU
        args.device = torch.device("cuda:" + str(args.gpu_id))  # 指定 torch 使用的设备（GPU）
        print(f"使用CUDA设备: {args.device}")  # 打印使用的 CUDA 设备
    else:  # 若无可用 CUDA
        args.device = torch.device("cpu")  # 回退到 CPU 设备
        print("CUDA不可用，使用CPU设备")  # 打印提示信息

    config = dict()  # 初始化模型配置字典
    config["n_users"] = data_generator.n_users  # 记录用户（症状组合）数量
    config["n_items"] = data_generator.n_items  # 记录物品（草药）数量

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """  # 说明：生成图的拉普拉斯相关邻接矩阵，用于图卷积传播
    plain_adj, norm_adj, mean_adj, sym_pair_adj, herb_pair_adj = (  # 从数据生成器获取多种形式的邻接矩阵
        data_generator.get_adj_mat()
    )
    args.node_dropout = eval(args.node_dropout)  # 将传入的字符串形式 node_dropout 转为 Python 对象
    args.mess_dropout = [float(x) for x in eval(args.mess_dropout)]  # 解析消息 dropout 列表并转为浮点

    if args.adj_type == "plain":  # 根据参数选择邻接矩阵类型
        config["norm_adj"] = plain_adj  # 使用原始邻接
        print("use the plain adjacency matrix")  # 打印使用的邻接类型
    elif args.adj_type == "norm":  # 选择对称归一化邻接
        config["norm_adj"] = norm_adj  # 使用归一化邻接
        print("use the normalized adjacency matrix")  # 打印使用的邻接类型
    elif args.adj_type == "gcmc":  # 选择 GCMC 的均值邻接
        config["norm_adj"] = mean_adj  # 使用均值邻接
        print("use the gcmc adjacency matrix")  # 打印使用的邻接类型
    else:  # 默认分支：使用 mean 邻接并加上单位阵（自连接）
        config["norm_adj"] = mean_adj + sp.eye(mean_adj.shape[0])  # 为稳定传播添加自环
        print("use the mean adjacency matrix")  # 打印使用的邻接类型

    config["sym_pair_adj"] = sym_pair_adj  # 症状-症状共现对的邻接矩阵（用于正则/约束）
    config["herb_pair_adj"] = herb_pair_adj  # 草药-草药共现对的邻接矩阵（用于正则/约束）

    t0 = time()  # 记录训练开始时间（秒）

    if args.pretrain == -1:  # 如果指定使用外部预训练嵌入
        pretrain_data = load_pretrained_data()  # 调用函数加载预训练数据
    else:  # 否则不使用预训练
        pretrain_data = None  # 将预训练数据置为 None

    model = SMGCN(data_config=config, pretrain_data=pretrain_data).to(args.device)  # 初始化模型并移动到指定设备
    print(model)  # 打印模型结构（__repr__）

    # 超图功能集成测试
    print("\n开始测试超图功能集成...")  # 打印开始测试超图功能提示
    print(f"symtom个数n_users={config['n_users']}, herb个数n_items={config['n_items']}")  # 打印用户和物品数量

    # 启用超图功能
    args.use_hypergraph = False  # 打开全局参数中的超图开关
    model.use_hypergraph = False  # 打开模型内部的超图开关

    # 加载草药训练数据并构建超图
    model.load_herb_train_data()  # 从训练文件中载入处方数据以构建超图节点/超边
    model.build_hypergraph_from_train_data()  # 基于训练数据构建超图邻接/结构

    print("✅ 超图功能集成成功！")  # 打印超图构建成功提示
    print(  # 打印超图邻接矩阵形状（用于确认构建是否正确）
        f"超图邻接矩阵形状: {model.hypergraph_adj.shape if model.hypergraph_adj is not None else 'None'}"
    )
    print(f"超图嵌入维度: {model.hypergraph_embed_size}")  # 打印超图嵌入维度配置
    print(f"对比学习权重: {model.contrastive_weight}")  # 打印对比学习损失权重

    """
    *********************************************************
    Save the model parameters.
    """  # 说明：设置模型权重保存路径
    weights_save_path = "%sweights/%s/%s/l%s_r%s" % (  # 组合生成权重保存路径（包含学习率与正则项）
        args.weights_path,
        args.dataset,
        model.model_type,
        str(args.lr),
        "-".join([str(r) for r in eval(args.regs)]),
    )
    ensureDir(weights_save_path)  # 确保保存路径存在（必要时自动创建）

    cur_best_pre_0, stopping_step = 0, 0  # 记录最优 Precision@K（K 列表中第一个）以及早停步数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 定义 Adam 优化器

    """
    *********************************************************
    Reload the pretrained model parameters.
    """  # 说明：根据参数决定是否加载预训练模型参数
    print("args.pretrain\t", args.pretrain)  # 打印预训练标志
    if args.pretrain == 1:  # 当为 1 时，按既定路径尝试加载预训练权重
        layer = "-".join([str(l) for l in eval(args.layer_size)])  # noqa: E741  # 解析层大小配置并拼接为字符串
        mess_dr = "-".join([str(d) for d in eval(args.mess_dropout)])  # 解析消息 dropout 并拼接（变量保留以兼容路径命名）
        weights_save_path = "%sweights/%s/%s/l%s_r%s" % (  # 重设权重路径以匹配预训练权重目录结构
            args.weights_path,
            args.dataset,
            model.model_type,
            layer,
            "-".join([str(r) for r in eval(args.regs)]),
        )
        pretrain_path = weights_save_path  # 预训练路径设置为上述目录
        print("load the pretrained model parameters from: ", pretrain_path)  # 打印即将加载的预训练路径

    """
    *********************************************************
    Train.
    """  # 说明：开始训练与评测主循环
    loss_loger, pre_loger, rec_loger, f1_loger, rmrr_loger = [], [], [], [], []  # 各指标日志列表：loss/precision/recall/f1/rmrr
    should_stop = False  # 早停标志，默认不早停

    for epoch in range(args.epoch):  # 外层循环：遍历训练 epoch
        t1 = time()  # 记录当前 epoch 的起始时间
        loss, mf_loss, emb_loss, reg_loss, cl_loss, hypergraph_cl_loss_total = (  # 初始化累计损失变量
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        n_batch = data_generator.n_train // args.batch_size + 1  # 计算本 epoch 的 batch 数

        for idx in range(n_batch):  # 内层循环：按批次迭代训练
            optimizer.zero_grad()  # 清空上一轮的梯度信息
            users, user_set, items, item_set = data_generator.sample()  # 采样一个批次的用户集合与正负样本集合
            users = torch.tensor(users, dtype=torch.float32).to(args.device)  # 将用户索引/特征转为张量并移至设备
            user_set = torch.tensor(user_set, dtype=torch.long).to(args.device)  # 将用户集合（症状组合）索引转为张量
            items = torch.tensor(items, dtype=torch.float32).to(args.device)  # 将物品（草药）索引/权重转为张量
            item_weights = torch.tensor(  # 将物品权重（重要性）转为张量
                data_generator.item_weights, dtype=torch.float32
            ).to(args.device)

            # 前向传播，获取用户和物品的嵌入以及对比学习损失
            (
                user_embeddings,  # 当前 batch 用户的嵌入
                all_user_embeddins,  # 所有用户嵌入（用于集合到集合的匹配）
                ia_embeddings,  # 物品（草药）嵌入
                cl_loss_user,  # 用户视角的对比学习损失
                cl_loss_item,  # 物品视角的对比学习损失
                cl_loss_hypergraph,  # 超图结构的对比学习损失
            ) = model(users, user_set)  # 调用模型前向过程

            # 超图对比学习损失已经在forward方法中计算  # 说明：cl_loss_hypergraph 已由前向过程返回

            # 计算损失
            batch_mf_loss, batch_emb_loss, batch_reg_loss, batch_cl_loss = (  # 分别计算矩阵分解损失、嵌入正则、权重正则和对比损失
                model.create_set2set_loss(
                    items,  # 当前 batch 的目标物品集合
                    item_weights,  # 物品权重
                    user_embeddings,  # 用户嵌入
                    all_user_embeddins,  # 所有用户嵌入
                    ia_embeddings,  # 物品嵌入
                    cl_loss_user,  # 用户对比损失（参与总损失）
                    cl_loss_item,  # 物品对比损失（参与总损失）
                )
            )

            # 添加超图对比学习损失
            batch_loss = (  # 汇总 batch 的总损失
                batch_mf_loss
                + batch_emb_loss
                + batch_reg_loss
                + batch_cl_loss
                + cl_loss_hypergraph
            )
            batch_loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 使用优化器更新参数

            loss += batch_loss.item()  # 累加总损失
            mf_loss += batch_mf_loss.item()  # 累加 MF 损失
            emb_loss += batch_emb_loss.item()  # 累加嵌入正则损失
            reg_loss += batch_reg_loss.item()  # 累加参数正则损失
            cl_loss += batch_cl_loss.item()  # 累加对比学习损失

            # 记录超图对比学习损失
            if isinstance(cl_loss_hypergraph, torch.Tensor):  # 若返回为张量则取标量值
                hypergraph_cl_loss_total += cl_loss_hypergraph.item()  # 累加超图对比损失
            else:  # 否则直接累加
                hypergraph_cl_loss_total += cl_loss_hypergraph  # 累加超图对比损失

        if np.isnan(loss):  # 若出现 NaN 损失，说明训练异常
            print("ERROR: loss is nan.")  # 打印错误提示
            sys.exit()  # 直接退出程序

        if (epoch + 1) % 10 != 0 and epoch != args.epoch - 1:  # 非每 10 轮且不是最后一轮时，按 verbose 打印简单训练日志并跳过评测
            if args.verbose > 0 and epoch % args.verbose == 0:  # 根据 verbose 频率输出训练日志
                perf_str = (  # 组装训练日志字符串
                    "Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f + %.5f]"
                    % (
                        epoch,
                        time() - t1,
                        loss,
                        mf_loss,
                        emb_loss,
                        reg_loss,
                        cl_loss,
                        hypergraph_cl_loss_total,
                    )
                )
                print(perf_str)  # 打印训练日志
            continue  # 跳过评测，进入下一轮

        t2 = time()  # 记录评测开始时间
        group_to_test = data_generator.test_group_set  # 获取测试集的 {用户: 真实物品集合} 映射
        ret = test(  # 调用评测函数，返回各指标在不同 K 上的结果
            model, list(data_generator.test_users), group_to_test, drop_flag=True
        )
        t3 = time()  # 记录评测结束时间

        loss_loger.append(loss)  # 记录本轮训练总损失
        rec_loger.append(ret["recall"])  # 记录本轮各 K 的 Recall
        pre_loger.append(ret["precision"])  # 记录本轮各 K 的 Precision
        f1_loger.append(ret["f1"])  # 记录本轮各 K 的 F1
        rmrr_loger.append(ret["rmrr"])  # 记录本轮各 K 的 RMRR

        if args.verbose > 0:  # 若开启详细日志，打印本 epoch 训练+评测汇总
            perf_str = (
                "Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f + %.5f]\n recall=[%.5f, %.5f], "
                "precision=[%.5f, %.5f],  f1=[%.5f, %.5f], RMRR=[%.5f, %.5f]"
                % (
                    epoch,  # 当前轮次
                    t2 - t1,  # 训练时间
                    t3 - t2,  # 评测时间
                    loss,  # 总损失
                    mf_loss,  # MF 损失
                    emb_loss,  # 嵌入正则损失
                    reg_loss,  # 参数正则损失
                    cl_loss,  # 对比学习损失
                    hypergraph_cl_loss_total,  # 超图对比学习损失
                    ret["recall"][0],  # Recall@K 的首个 K 值（如 K=5）
                    ret["recall"][-1],  # Recall@K 的最后一个 K 值（如 K=20）
                    ret["precision"][0],  # Precision@K 的首个 K 值
                    ret["precision"][-1],  # Precision@K 的最后一个 K 值
                    ret["f1"][0],  # F1@K 的首个 K 值
                    ret["f1"][-1],  # F1@K 的最后一个 K 值
                    ret["rmrr"][0],  # RMRR@K 的首个 K 值
                    ret["rmrr"][-1],  # RMRR@K 的最后一个 K 值
                )
            )
            print(perf_str)  # 打印训练+评测日志

        cur_best_pre_0, stopping_step, should_stop = no_early_stopping(  # 调用早停逻辑，基于 Precision@K 的第一个 K 值
            ret["precision"][0], cur_best_pre_0, stopping_step, expected_order="acc"
        )

        if should_stop:  # 若达到早停条件
            print("early stopping")  # 打印早停提示
            break  # 结束训练循环

        if ret["precision"][0] == cur_best_pre_0 and args.save_flag == 1:  # 若当前 Precision 达到最优且允许保存
            print("\n", "*" * 80, "model sava path", weights_save_path + "model.pkl")  # 打印保存路径
            torch.save(model, weights_save_path + "model.pkl")  # 保存整个模型对象（注意体积较大）
            print("save the weights in path: ", weights_save_path)  # 打印保存确认

    recs = np.array(rec_loger)  # 将各轮 Recall 记录转为数组（shape: [评测次数, len(Ks)]）
    pres = np.array(pre_loger)  # 将各轮 Precision 记录转为数组
    f1s = np.array(f1_loger)  # 将各轮 F1 记录转为数组
    rmrr = np.array(rmrr_loger)  # 将各轮 RMRR 记录转为数组

    if len(rec_loger) == 0:  # 若整个训练过程中没有进行评测（例如 epoch 设置导致）
        print("No evaluation results available. Training completed without evaluation.")  # 打印提示
        endTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 记录结束时间
        print("end ", endTime)  # 打印结束时间
        sys.exit()  # 退出程序

    best_rec_0 = max(recs[:, 0])  # 在所有评测轮中找到 Recall@K（第一个 K）的最佳数值
    idx = list(recs[:, 0]).index(best_rec_0)  # 找到对应的轮次索引

    final_perf = (  # 组装最终结果字符串，展示最佳轮的各指标
        "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], f1=[%s], RMRR=[%s]"
        % (
            idx,  # 最佳评测所在的迭代轮次
            time() - t0,  # 从训练开始到现在的总耗时
            "\t".join(["%.5f" % r for r in recs[idx]]),  # 最佳轮的各 K 的 Recall
            "\t".join(["%.5f" % r for r in pres[idx]]),  # 最佳轮的各 K 的 Precision
            "\t".join(["%.5f" % r for r in f1s[idx]]),  # 最佳轮的各 K 的 F1
            "\t".join(["%.5f" % r for r in rmrr[idx]]),  # 最佳轮的各 K 的 RMRR
        )
    )
    print(final_perf)  # 打印最终最佳表现

    save_path = "%soutput/%s/%s.result-SMGCN-%d" % (  # 结果保存文件路径（按数据集、模型类型与结果索引区分）
        args.proj_path,
        args.dataset,
        model.model_type,
        args.result_index,
    )
    ensureDir(save_path)  # 确认输出目录存在
    f = open(save_path, "a")  # 以追加方式打开结果文件
    f.write(  # 写入当前超参数配置与最终最佳表现
        "embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, adj_type=%s\n\t%s\n"
        % (
            args.embed_size,  # 嵌入维度
            args.lr,  # 学习率
            args.layer_size,  # 层结构
            args.keep_prob,  # 保留率/Dropout 概率
            args.regs,  # 正则项系数
            args.loss_type,  # 损失类型
            args.adj_type,  # 邻接类型
            final_perf,  # 最佳结果字符串
        )
    )
    f.close()  # 关闭文件句柄

    endTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 记录结束时间
    print("end ", endTime)  # 打印结束时间
