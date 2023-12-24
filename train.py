from tqdm import tqdm
from test import valid, test
import torch
from collections import deque
import numpy as np
import torch.nn as nn
from loss import loss_jocor, pseudo_loss, loss_record
import torch.nn.functional as F
from utils import ndarray_memory, normalize_array, loss_display
from correction import positive_correction, negative_correction
import os
import time
import math
from sklearn.mixture import GaussianMixture as GMM

def train(model, optimizer, model2, optimizer2, train_loader, val_loader, test_loader, args):
    # os.environ['CUDA_VISIBLE_DEVICES'] ='%d'%args.gpuid
    mlb = None
    num_stop_dropping = 0
    best_valid_result = 0    
    loss_overall = []
    rank_overall = []
    label_overall = []
    for epoch in range(args.epochs):
        if args.pretrained == True:
            break

        args.epoch = epoch
        # args.current_rate = args.rate_schedule[epoch]       
        
        #formal training
        model.train()
        for i, batch in enumerate(tqdm(train_loader), 0):
            batch_loss = train_one_batch(epoch, batch, model, optimizer, args)

        # early stop
        valid_result = valid(model, val_loader, mlb, args)[-1]
        if valid_result > best_valid_result:
            best_valid_result = valid_result
            num_stop_dropping = 0
            torch.save(model.state_dict(),args.model_path)
        else:
            num_stop_dropping += 1

        if args.test_each_epoch:
            valid_result = valid(model, val_loader, mlb, args)
            test_result = valid(model, test_loader, mlb, args)
            print(f'Epoch: {epoch} | Loss: {batch_loss: .4f} | Stop: {num_stop_dropping} | Valid: {valid_result} | Test: {test_result} ')
        else:
            valid_result = valid(model, val_loader, mlb, args)
            print(f'Epochs: {epoch} | Train Loss: {batch_loss: .4f} | Early Stop: {num_stop_dropping} | Valid Result: {valid_result}')

        if num_stop_dropping >= args.early_stop_tolerance:
            print('Have not increased for %d check points, early stop training' % num_stop_dropping)
            break

        #record
        if epoch >= 0:
            loss_epoch = []
            rank_epoch = []
            label_epoch = []
            tp_epoch = []
            fp_epoch = []
            tn_epoch = []
            fn_epoch = []
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(train_loader), 0):
                    batch_loss, batch_rank, batch_label, tp, fp, tn, fn = record_one_batch(epoch, batch, model, optimizer, args)
                    loss_epoch.append(batch_loss)
                    rank_epoch.append(batch_rank)
                    label_epoch.append(batch_label)
                    tp_epoch.append(tp)
                    fp_epoch.append(fp)
                    tn_epoch.append(tn)
                    fn_epoch.append(fn)      
            # loss_display2(clean_current, noise_current, epoch, args)              
            loss_overall.append(np.array(loss_epoch))
            rank_overall.append(np.array(rank_epoch))
            if epoch==0:
                label_overall = np.array(label_epoch)
                tp_overall = np.array(tp_epoch)
                fp_overall = np.array(fp_epoch)
                tn_overall = np.array(tn_epoch)
                fn_overall = np.array(fn_epoch)

    # pseudo label
    if args.pretrained ==True:

        #load the model 
        model.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))

        #replace the pretrained_path to history_path 
        args.pt_history_path = args.pretrained_path.replace('lstm', 'history')
        args.pt_history_path = args.pt_history_path.replace('pth', 'npy')

        #load the history info
        loss_name = args.pt_history_path.replace('history', 'loss_history')
        rank_name = args.pt_history_path.replace('history', 'rank_history')
        label_name = args.pt_history_path.replace('history', 'label_history')
        tn_name = args.pt_history_path.replace('history', 'tn_history')

        #load the npy file
        loss_overall = np.load(loss_name)
        rank_overall = np.load(rank_name)
        label_overall = np.load(label_name)
        tn_overall = np.load(tn_name)

        #print the load info
        print('load the history info')
        print(loss_name)
        print(rank_name)
        print(label_name)
        print(tn_name)
        print(args.pretrained_path)

    else:
        loss_overall = np.array(loss_overall)
        rank_overall = np.array(rank_overall)
        label_overall = np.array(label_overall)
        tp_overall = np.array(tp_overall)
        fp_overall = np.array(fp_overall)
        tn_overall = np.array(tn_overall)
        fn_overall = np.array(fn_overall)

        # print the memory
        ndarray_memory('loss_overall', loss_overall)
        ndarray_memory('rank_overall', rank_overall)
        ndarray_memory('label_overall', label_overall)
        ndarray_memory('tp_overall', tp_overall)
        ndarray_memory('fp_overall', fp_overall)
        ndarray_memory('tn_overall', tn_overall)
        ndarray_memory('fn_overall', fn_overall)

        # save the history info
        loss_name = args.history_path.replace('history', 'loss_history')
        rank_name = args.history_path.replace('history', 'rank_history')
        label_name = args.history_path.replace('history', 'label_history')
        tp_name = args.history_path.replace('history', 'tp_history')
        fp_name = args.history_path.replace('history', 'fp_history')
        tn_name = args.history_path.replace('history', 'tn_history')
        fn_name = args.history_path.replace('history', 'fn_history')

        # save the npy file
        np.save(loss_name, loss_overall)
        np.save(rank_name, rank_overall)
        np.save(label_name, label_overall)
        np.save(tp_name, tp_overall)
        np.save(fp_name, fp_overall)
        np.save(tn_name, tn_overall)
        np.save(fn_name, fn_overall)

        # print the save info
        print('save the history info')
        print(loss_name)
        print(rank_name)
        print(label_name)
        print(tp_name)
        print(fp_name)
        print(tn_name)
        print(fn_name)
        print(args.model_path)
    
    # average the training history
    loss_overall = loss_overall.reshape(loss_overall.shape[0], -1)
    rank_overall = rank_overall.reshape(rank_overall.shape[0], -1)    
    # label_overall = label_overall[-1]
    loss_overall = np.mean(loss_overall, axis=0) + loss_overall[-1]
    rank_overall = np.mean(rank_overall, axis=0) + rank_overall[1]

    # positive and negative pairs
    label_overall_line = label_overall.reshape(-1, )  
    tn_overall_line = tn_overall.reshape(-1, )
    pos_loss_pairs = loss_overall[label_overall_line==1]
    pos_rank_pairs = rank_overall[label_overall_line==1]
    neg_loss_pairs = loss_overall[tn_overall_line==1]
    neg_rank_pairs = rank_overall[tn_overall_line==1]

    # normalization 
    pos_loss_pairs = normalize_array(pos_loss_pairs)
    pos_rank_pairs = normalize_array(pos_rank_pairs)
    neg_loss_pairs = normalize_array(neg_loss_pairs)
    neg_rank_pairs = normalize_array(neg_rank_pairs)

    # HSM
    com_coef = 0.7
    pos_hsm_pairs = pos_loss_pairs*com_coef+(1-com_coef)*pos_rank_pairs
    neg_hsm_pairs = neg_loss_pairs*com_coef+(1-com_coef)*neg_rank_pairs
    loss_display(pos_hsm_pairs, 'pos_hsm_pairs', args)
    loss_display(neg_hsm_pairs, 'neg_hsm_pairs', args)
    pos_hsm_pairs = pos_hsm_pairs.reshape(-1, 1)
    neg_hsm_pairs = neg_hsm_pairs.reshape(-1, 1)

    # use GMM to get the noise rate
    gmm = GMM(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
    gmm.fit(pos_hsm_pairs)
    prob = gmm.predict_proba(pos_hsm_pairs)
    prob = prob[:, gmm.means_.argmax()]
    pos_noise_rate = np.mean(prob)
    
    gmm = GMM(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
    gmm.fit(neg_hsm_pairs)
    prob = gmm.predict_proba(neg_hsm_pairs)
    prob = prob[:, gmm.means_.argmax()]
    neg_noise_rate = np.mean(prob)
    print('pos_noise_rate', pos_noise_rate)  
    print('neg_noise_rate', neg_noise_rate)   

    # get threshold of both metrics
    pos_thres_1 = np.percentile(pos_hsm_pairs, int((1-pos_noise_rate)*100)) 
    pos_thres_2 = np.percentile(pos_hsm_pairs, int((1-(pos_noise_rate/2))*100)) 
    neg_thres_1 = np.percentile(neg_hsm_pairs, int((1-neg_noise_rate)*100))
    neg_thres_2 = np.percentile(neg_hsm_pairs, int((1-(neg_noise_rate/2))*100))
    print('pos_threshold', pos_thres_1, pos_thres_2)
    print('neg_threshold', neg_thres_1, neg_thres_2)

    # hsm to pseudo label
    pos_hsm_pairs = pos_hsm_pairs.reshape(-1)
    neg_hsm_pairs = neg_hsm_pairs.reshape(-1)

    pos_pseudo = positive_correction(pos_hsm_pairs, pos_thres_1, pos_thres_2)
    neg_pseudo = negative_correction(neg_hsm_pairs, neg_thres_1, neg_thres_2)
    
    # plot the pseudo label
    loss_display(pos_pseudo, 'pos_pseudo', args)
    loss_display(neg_pseudo, 'neg_pseudo', args)

    # get pseudo label
    pseudo_overall = []
    pos_count = 0 
    neg_count = 0
    label_overall = label_overall.reshape(-1, 881)
    tn_overall = tn_overall.reshape(-1, 881)*-2
    overall = label_overall + tn_overall

    for inst in tqdm(overall):
        pseudo_inst = []
        for pair in inst:
            if pair == 1:
                pseudo_inst.append(pos_pseudo[pos_count])
                pos_count += 1
            elif pair == -2:
                pseudo_inst.append(neg_pseudo[neg_count])
                neg_count += 1
            elif pair == -1:
                print('error')
            else:
                pseudo_inst.append(0)
        pseudo_overall.append(pseudo_inst)
    pseudo_overall = np.array(pseudo_overall)
    print('pseudo_overall.shape', pseudo_overall.shape)

    # re training
    num_stop_dropping = 0
    best_valid_result = 0
    for epoch in range(args.epochs):
        args.epoch = epoch  
        #formal training
        for i, batch in enumerate(tqdm(train_loader), 0):
            pseudo_tensor = pseudo_overall[i*args.batch_size: (i+1)*args.batch_size]
            batch_loss = pseudo_one_batch(epoch, batch, pseudo_tensor,  model2, optimizer2, args) 

        valid_result = valid(model2, val_loader, mlb, args)[-1]
        if valid_result > best_valid_result:
            best_valid_result = valid_result
            num_stop_dropping = 0
            torch.save(model2.state_dict(), args.model_path)
        else:
            num_stop_dropping += 1

        if args.test_each_epoch:
            valid_result = valid(model2, val_loader, mlb, args)
            test_result = valid(model2, test_loader, mlb, args) 
            print(f'Epoch: {epoch} | Loss: {batch_loss: .4f} | Stop: {num_stop_dropping} | Valid: {valid_result} | Test: {test_result} ')
        else:
            valid_result = valid(model2, val_loader, mlb, args)
            print(f'Epochs: {epoch} | Train Loss: {batch_loss: .4f} | Early Stop: {num_stop_dropping} | Valid Result: {valid_result}')

        if num_stop_dropping >= args.early_stop_tolerance:
            print('Have not increased for %d check points, early stop training' % num_stop_dropping)
            break


def train_one_batch(epoch, batch, model, optimizer, args):
    # train for one batch
    model.to(args.device)
    model.train()

    src, trg, fn, pf, tp, can = batch
    input_id = src.to(args.device)
    labels = trg.to(args.device) 
    
    # Forward + Backward + Optimize
    logits1 = model(input_id)
    logits2 = None

    loss = loss_jocor(logits1, logits2, labels, args)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(closure=None)
    return loss.item()

def record_one_batch(epoch, batch, model, optimizer, args):
    # train for one batch
    model.to(args.device)
    model.eval()

    src, trg, fn, fp, tp, can = batch
    input_id = src.to(args.device)
    labels = trg.to(args.device)
    fn = fn.to(args.device)
    fp = fp.to(args.device)
    tp = tp.to(args.device)
    can = can.to(args.device)
    
    # Forward + Backward + Optimize
    logits1 = model(input_id)
    logits2 = None

    batch_loss, batch_rank, label_rank, tp, fp, tn, fn = loss_record(logits1, logits2, labels, fn, fp, tp, can, args)
    return batch_loss, batch_rank, label_rank, tp, fp, tn, fn

def pseudo_one_batch(epoch, batch, pseudo_tensor,  model, optimizer, args):

    # train for one batch
    model.to(args.device)
    model.train()

    src, trg, fn, pf, tp, can = batch
    input_id = src.to(args.device)
    labels = trg.to(args.device)  
    
    # Forward + Backward + Optimize
    logits1 = model(input_id)
    logits2 = None

    loss = pseudo_loss(logits1, logits2, labels, pseudo_tensor, args)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(closure=None)
    return loss.item()

def thres2pseudo(loss_batch, rank_batch, label_batch, args):
    # train for one batch
    loss_batch = torch.Tensor(loss_batch).to(args.device)
    rank_batch = torch.Tensor(rank_batch).to(args.device)
    label_batch = torch.Tensor(label_batch).to(args.device)

    pseudo_labels = label_batch.clone()
    mask_index = loss_batch > args.loss_threshold
    pseudo_labels[mask_index] = 0    
    mask_index = rank_batch > args.rank_threshold
    pseudo_labels[mask_index] = 0     
    pseudo_tensor = pseudo_labels.detach().cpu().numpy()

    return pseudo_tensor



    