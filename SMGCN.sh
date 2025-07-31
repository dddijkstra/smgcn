#!/bin/bash

# 设置默认参数值，可通过命令行参数覆盖
DATASET=${1:-"Herb"}                    # 数据集名称
EMBED_SIZE=${2:-64}                     # 嵌入维度
LAYER_SIZE=${3:-"[64,64]"}              # 层大小
LR=${4:-0.001}                          # 学习率
BATCH_SIZE=${5:-1024}                   # 批次大小
EPOCH=${6:-2000}                        # 训练轮次
VERBOSE=${7:-1}                         # 详细输出
GPU_ID=${8:-0}                          # GPU ID
REG=${9:-"7e-3"}                        # 正则化参数
MESS_DROPOUT=${10:-"[0.0,0.0]"}         # Dropout参数
ADJ_TYPE=${11:-"norm"}                  # 邻接矩阵类型

# 显示参数使用说明
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "使用方法: $0 [dataset] [embed_size] [layer_size] [lr] [batch_size] [epoch] [verbose] [gpu_id] [reg] [mess_dropout] [adj_type]"
    echo "参数说明:"
    echo "  dataset      : 数据集名称 (默认: Herb)"
    echo "  embed_size   : 嵌入维度 (默认: 64)"
    echo "  layer_size   : 层大小 (默认: [64,64])"
    echo "  lr           : 学习率 (默认: 0.001)"
    echo "  batch_size   : 批次大小 (默认: 1024)"
    echo "  epoch        : 训练轮次 (默认: 2000)"
    echo "  verbose      : 详细输出 (默认: 1)"
    echo "  gpu_id       : GPU ID (默认: 0)"
    echo "  reg          : 正则化参数 (默认: 7e-3)"
    echo "  mess_dropout : Dropout参数 (默认: [0.0,0.0])"
    echo "  adj_type     : 邻接矩阵类型 (默认: norm)"
    echo ""
    echo "示例:"
    echo "  $0 Herb 64 '[64,64]' 0.001 1024 2000 1 0"
    echo "  $0 NetEase 128 '[128,128]' 0.002 512 1000"
    exit 0
fi

# 获取当前分支名、commit ID和日期
branch_name=$(git branch --show-current)
commit_id=$(git rev-parse --short HEAD)
current_date=$(date +"%Y%m%d_%H%M")

# 创建output目录（如果不存在）
mkdir -p output

# 生成日志文件名
log_file="output/${branch_name}_${commit_id}_${current_date}_${DATASET}.log"

echo "开始执行SMGCN训练脚本" | tee $log_file
echo "执行时间: $(date)" | tee -a $log_file
echo "当前分支: $branch_name" | tee -a $log_file
echo "Commit ID: $commit_id" | tee -a $log_file
echo "========================================" | tee -a $log_file
echo "实验参数配置:" | tee -a $log_file
echo "  数据集: $DATASET" | tee -a $log_file
echo "  嵌入维度: $EMBED_SIZE" | tee -a $log_file
echo "  层大小: $LAYER_SIZE" | tee -a $log_file
echo "  学习率: $LR" | tee -a $log_file
echo "  批次大小: $BATCH_SIZE" | tee -a $log_file
echo "  训练轮次: $EPOCH" | tee -a $log_file
echo "  GPU ID: $GPU_ID" | tee -a $log_file
echo "  正则化: $REG" | tee -a $log_file
echo "  Dropout: $MESS_DROPOUT" | tee -a $log_file
echo "  邻接矩阵类型: $ADJ_TYPE" | tee -a $log_file
echo "========================================" | tee -a $log_file

echo "" | tee -a $log_file
echo "开始训练..." | tee -a $log_file
echo "执行命令: python -u main.py --dataset $DATASET --embed_size $EMBED_SIZE --layer_size '$LAYER_SIZE' --lr $LR --batch_size $BATCH_SIZE --epoch $EPOCH --verbose $VERBOSE --gpu_id $GPU_ID" | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file
echo "================================" | tee -a $log_file

# 执行训练命令并将输出追加到日志文件
python -u main.py --dataset "$DATASET" --embed_size "$EMBED_SIZE" --layer_size "$LAYER_SIZE" --lr "$LR" --batch_size "$BATCH_SIZE" --epoch "$EPOCH" --verbose "$VERBOSE" --gpu_id "$GPU_ID" 2>&1 | tee -a $log_file

echo "" | tee -a $log_file
echo "结束时间: $(date)" | tee -a $log_file
echo "================================" | tee -a $log_file

echo "" | tee -a $log_file
echo "所有实验完成！" | tee -a $log_file
echo "日志文件保存在: $log_file" | tee -a $log_file

