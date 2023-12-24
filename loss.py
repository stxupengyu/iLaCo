import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

def pseudo_loss(y_1, y_2, t, pseudo, args):
    if 'weight' in args.mode: 
        loss_origin = F.binary_cross_entropy(y_1, t, reduce = False)
        weight = torch.Tensor(pseudo).to(args.device)
        loss = torch.mean(loss_origin*weight)
    elif 'combine' in args.mode:
        pseudo_labels = torch.Tensor(pseudo).to(args.device)
        coef = args.epoch/20.0 if args.epoch < 20 else 1
        combine_labels = pseudo_labels * coef + t * (1-coef)
        loss = F.binary_cross_entropy(y_1, combine_labels, reduce = False)
        loss = torch.mean(loss)
        # loss = F.binary_cross_entropy(y_1, t, reduce = False)
        # loss = torch.mean(loss)
    else:
        pseudo_labels = torch.Tensor(pseudo).to(args.device)
        loss = F.binary_cross_entropy(y_1, pseudo_labels, reduce = False)
        loss = torch.mean(loss)
    return loss

def loss_record(y_1, y_2, t, fn, fp, tp, can, args):
    
    # label
    tp_label = tp
    fp_label = fp
    tn_label = can
    fn_label = fn

    label_numpy = t.detach().cpu().numpy()
    tp_label = tp_label.detach().cpu().numpy()
    fp_label = fp_label.detach().cpu().numpy()
    tn_label = tn_label.detach().cpu().numpy()
    fn_label = fn_label.detach().cpu().numpy()

    # metirc
    loss_numpy = F.binary_cross_entropy(y_1, t, reduce = False).detach().cpu().numpy()
    rank = torch.argsort(y_1, descending= True) + torch.ones_like(y_1)
    rank_numpy = rank.detach().cpu().numpy()
    
    return loss_numpy, rank_numpy, label_numpy, tp_label, fp_label, tn_label, fn_label

def loss_jocor(y_1, y_2, t, args):
    loss_origin = F.binary_cross_entropy(y_1, t, reduce = False)
    loss = torch.mean(loss_origin)
    return loss

def bce_loss(y_pred, y_true):
    criteria = nn.BCEWithLogitsLoss()
    loss = criteria(y_pred, y_true)
    return loss

def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)
    

