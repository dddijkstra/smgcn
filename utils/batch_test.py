# -*- coding: utf-8 -*-
# @Time    : 2022/1/10 11:10
# @Author  : Ywj
# @File    : batch_test.py
# @Description : 测试核心代码
import numpy as np

from utils.parser import parse_args
from utils.load_data import *
import multiprocessing
import torch
# from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve
# import sklearn.metrics as m

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)

data_generator = Data(args=args, path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size


def test(model, users_to_test, test_group_list, drop_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)),
              'f1': np.zeros(len(Ks)), 'rmrr': np.zeros(len(Ks))}
    # test_users = users_to_test
    test_users = torch.tensor(users_to_test, dtype=torch.float32).to(args.device)
    item_batch = range(ITEM_NUM)
    if drop_flag == False:
        user_embeddings, pos_i_g_embeddings = model(test_users, pos_items=item_batch, train=False)
    else:
        args.mess_dropout = str([0.] * len(eval(args.layer_size)))
        args.mess_dropout = eval(args.mess_dropout)
        user_embeddings, pos_i_g_embeddings = model(test_users, pos_items=item_batch, train=False)
        print('drop_flag: ', drop_flag, ',\t mess_dropout: ', args.mess_dropout)
    rate_batch = model.create_batch_rating(pos_i_g_embeddings, user_embeddings)
    print('rate_batch ', rate_batch.shape)

    user_batch_rating_uid = zip(test_users, rate_batch)
    user_rating_dict = {}

    index = 0
    for entry in user_batch_rating_uid:
        rating = entry[1]         # (1, 753)
        # print("@@@@@@@@@@@@rating@@@@@@@@")
        # print(rating)
        temp = [(i, float(rating[i])) for i in range(len(rating))]
        user_rating_dict[index] = temp
        index += 1
    # user_rating_dict {sym-1: [(herb1, rate), (herb2, rate), ..., (herb753, rate)], ...,
    #                   sym-1162:[(herb1, rate), (herb2, rate), ..., (herb753, rate)]}

    precision_n = np.zeros(len(Ks))
    recall_n = np.zeros(len(Ks))
    f1_n = np.zeros(len(Ks))
    rmrr_n = np.zeros(len(Ks))
    topN = Ks

    gt_count = 0
    candidate_count = 0
    for index in range(len(test_group_list)):
        entry = test_group_list[index]
        v = entry[1]                              # sym-index's true herb list
        rating = user_rating_dict[index]

        candidate_count += len(rating)
        rating.sort(key=lambda x: x[1], reverse=True)
        gt_count += len(v)
        K_max = topN[len(topN) - 1]
        r = []
        # number = 0
        # herb_results = []  # 推荐列表中herb 集合
        for i in rating[:K_max]:
            herb = i[0]
            if herb in v:
                r.append(1)
            else:
                r.append(0)

        for ii in range(len(topN)):      # topN: [5, 10, 15, 20]
            number = 0
            herb_results = []  # 推荐列表中herb 集合
            for i in rating[:topN[ii]]:
                herb = i[0]
                herb_results.append(herb)
                if herb in v:
                    number += 1
            # todo: modified MRR to Rank-MRR
            mrr_score = 0.
            # print("-----herb_results:", herb_results)
            # print("-----ground truth:", v)
            for a_rank in range(len(v)):  # herb 在grand truth中的位置a_rank
                if v[a_rank] in herb_results:
                    a_refer = herb_results.index(v[a_rank])  # herb 在推荐列表中的位置a_refer
                    mrr_score += 1.0 / (abs(a_refer - a_rank) + 1)
            current_precision = float(number / topN[ii])
            current_recall = float(number / len(v))
            precision_n[ii] = precision_n[ii] + current_precision
            recall_n[ii] = recall_n[ii] + current_recall
            f1_n[ii] = f1_n[ii] + f1_at_k(current_precision, current_recall)
            rmrr_n[ii] = rmrr_n[ii] + mrr_score / len(v)
    print('gt_count ', gt_count)
    print('candidate_count ', candidate_count)
    print('ideal candidate count ', len(test_group_list) * ITEM_NUM)
    for ii in range(len(topN)):
        result['precision'][ii] = precision_n[ii] / len(test_group_list)
        result['recall'][ii] = recall_n[ii] / len(test_group_list)
        result['f1'][ii] = f1_n[ii] / len(test_group_list)
        result['rmrr'][ii] = rmrr_n[ii] / len(test_group_list)
    return result





def f1_at_k(precision, recall):
    """Calculate F1 score from precision and recall values.
    
    The F1 score is the harmonic mean of precision and recall, providing a balanced
    measure that considers both metrics equally. It is calculated as:
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        precision (float): Precision value, should be in range [0, 1]
        recall (float): Recall value, should be in range [0, 1]
    
    Returns:
        float: F1 score in range [0, 1]. Returns 0.0 if precision + recall = 0
               to avoid division by zero.
    
    Raises:
        TypeError: If precision or recall are not numeric types
        ValueError: If precision or recall are negative or greater than 1,
                   or if they are NaN or infinite values
    
    Examples:
        >>> f1_at_k(0.5, 0.5)
        0.5
        >>> f1_at_k(0.8, 0.6)
        0.6857142857142857
        >>> f1_at_k(1.0, 0.0)
        0.0
        >>> f1_at_k(0.0, 1.0)
        0.0
        >>> f1_at_k(0.0, 0.0)
        0.0
        >>> f1_at_k(1.0, 1.0)
        1.0
    """
    # Input validation - check if inputs are numeric
    if not isinstance(precision, (int, float, np.number)):
        raise TypeError(f"Precision must be a numeric type, got {type(precision)}")
    if not isinstance(recall, (int, float, np.number)):
        raise TypeError(f"Recall must be a numeric type, got {type(recall)}")
    
    # Convert to float for consistent handling
    precision = float(precision)
    recall = float(recall)
    
    # Check for NaN or infinite values
    if np.isnan(precision) or np.isinf(precision):
        raise ValueError(f"Precision must be a finite number, got {precision}")
    if np.isnan(recall) or np.isinf(recall):
        raise ValueError(f"Recall must be a finite number, got {recall}")
    
    # Check if values are in valid range [0, 1]
    if precision < 0 or precision > 1:
        raise ValueError(f"Precision must be in range [0, 1], got {precision}")
    if recall < 0 or recall > 1:
        raise ValueError(f"Recall must be in range [0, 1], got {recall}")
    
    # Handle division by zero case
    if precision + recall == 0:
        return 0.0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Ensure result is in valid range (should be guaranteed by math, but safety check)
    f1_score = max(0.0, min(1.0, f1_score))
    
    return f1_score





