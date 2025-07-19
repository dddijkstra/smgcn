# -*- coding: utf-8 -*-
# @Time    : 2022/1/17 16:22
# @Author  : Ywj
# @File    : smgcn_main.py
# @Description :  SMGCN主函数

import numpy as np
import os
import sys
from model.SMGCN import SMGCN
import datetime
from utils.helper import *
import torch
from utils.batch_test import *
import torch.optim as optim


def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (
        args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

from termcolor import colored

if __name__ == '__main__':
    startTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('start ', startTime)
    print('************SMGCN*************** ')
    print('result_index ', args.result_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.device = torch.device('cuda:' + str(args.gpu_id))
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, sym_pair_adj, herb_pair_adj = data_generator.get_adj_mat()
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = [float(x) for x in eval(args.mess_dropout)]
    print(colored('Epoch is set: '+str(args.epoch), 'red'))
    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')
    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')
    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')
    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')

    config['sym_pair_adj'] = sym_pair_adj
    config['herb_pair_adj'] = herb_pair_adj

    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    model = SMGCN(data_config=config,
                  pretrain_data=pretrain_data).to(args.device)
    print(model)

    """
    *********************************************************
    Save the model parameters.
    """
    weights_save_path = '%sweights/%s/%s/l%s_r%s' % (
        args.weights_path, args.dataset, model.model_type, str(args.lr),
        '-'.join([str(r) for r in eval(args.regs)]))
    ensureDir(weights_save_path)

    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    print("args.pretrain\t", args.pretrain)
    if args.pretrain == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        mess_dr = '-'.join([str(d) for d in eval(args.mess_dropout)])
        weights_save_path = '%sweights/%s/%s/l%s_r%s' % (
            args.weights_path, args.dataset, model.model_type, layer,
            '-'.join([str(r) for r in eval(args.regs)]))
        pretrain_path = weights_save_path
        print('load the pretrained model parameters from: ', pretrain_path)

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, rmrr_loger = [], [], [], [], []
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss, cl_loss, cl_user_fusion_loss, cl_item_fusion_loss = 0., 0., 0., 0., 0., 0., 0.

        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            optimizer.zero_grad()
            users, user_set, items, item_set = data_generator.sample()
            users = torch.tensor(users, dtype=torch.float32).to(args.device)
            user_set = torch.tensor(user_set, dtype=torch.long).to(args.device)
            items = torch.tensor(items, dtype=torch.float32).to(args.device)
            item_weights = torch.tensor(
                data_generator.item_weights, dtype=torch.float32).to(args.device)

            user_embeddings, all_user_embeddins, ia_embeddings, cl_loss_user_fusion, cl_loss_item_fusion = model(
                users, user_set)

            batch_mf_loss, batch_emb_loss, batch_reg_loss, batch_cl_loss = \
                model.create_set2set_loss(
                    items, item_weights, user_embeddings, all_user_embeddins, ia_embeddings)
            alpha = 0.1
            batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss + batch_cl_loss \
                + alpha * (cl_loss_user_fusion + cl_loss_item_fusion)
            batch_loss.backward()
            optimizer.step()

            def to_float(val):
                import torch
                return val.item() if isinstance(val, torch.Tensor) else float(val)

            loss += to_float(batch_loss)
            mf_loss += to_float(batch_mf_loss)
            emb_loss += to_float(batch_emb_loss)
            reg_loss += to_float(batch_reg_loss)
            cl_loss += to_float(batch_cl_loss)
            cl_user_fusion_loss += to_float(cl_loss_user_fusion)
            cl_item_fusion_loss += to_float(cl_loss_item_fusion)

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss, cl_loss, cl_user_fusion_loss, cl_item_fusion_loss)
                print(perf_str)
            continue

        t2 = time()
        group_to_test = data_generator.test_group_set
        ret = test(model, list(data_generator.test_users),
                   group_to_test, drop_flag=True)
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        rmrr_loger.append(ret['rmrr'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f]\n recall=[%.5f, %.5f], ' \
                'precision=[%.5f, %.5f],  ndcg=[%.5f, %.5f], RMRR=[%.5f, %.5f]' % \
                (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss,
                       ret['recall'][0], ret['recall'][-1], ret['precision'][0], ret['precision'][-1],
                       ret['ndcg'][0], ret['ndcg'][-1], ret['rmrr'][0], ret['rmrr'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = no_early_stopping(ret['precision'][0], cur_best_pre_0,
                                                                       stopping_step, expected_order='acc')

        if should_stop == True:
            print('early stopping')
            break

        if ret['precision'][0] == cur_best_pre_0 and args.save_flag == 1:
            print("\n", "*" * 80, "model sava path",
                  weights_save_path + 'model.pkl')
            torch.save(model, weights_save_path + 'model.pkl')
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    rmrr = np.array(rmrr_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], ndcg=[%s], RMRR=[%s]" % \
        (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
         '\t'.join(['%.5f' % r for r in pres[idx]]),
         '\t'.join(['%.5f' % r for r in ndcgs[idx]]),
                 '\t'.join(['%.5f' % r for r in rmrr[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result-SMGCN-%d' % (
        args.proj_path, args.dataset, model.model_type, args.result_index)
    ensureDir(save_path)
    f = open(save_path, 'a')
    f.write(
        'embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.layer_size, args.keep_prob, args.regs,
           args.loss_type, args.adj_type, final_perf))
    f.close()

    endTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('end ', endTime)
