import os
import numpy as np
import torch.utils.data as data_utils
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import logging
from scipy.sparse import csr_matrix
import json
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_data_loader2(args):
    with open(os.path.join(args.data_dir, "word2idx.json"), "r") as file:
        word2idx = json.load(file)
    with open(os.path.join(args.data_dir, "tag2idx.json"), "r") as file:
        tag2idx = json.load(file)
    
    src, fn, fp, tp, can = load_txt_data2(args)
    trg = [i+j for i, j in zip(tp, fn)]
    test_src, test_trg = load_txt_data3(args)

    tokenized_pairs = list(zip(src, trg))
    train, valid = dataSplit2(tokenized_pairs, args.seed)
    test = list(zip(test_src, test_trg))

    X_train, y_train = list2numpy(train, word2idx, tag2idx, args.max_len, args.vocab_size)
    X_valid, y_valid = list2numpy(valid, word2idx, tag2idx, args.max_len, args.vocab_size)
    X_test, y_test = list2numpy(test, word2idx, tag2idx, args.max_len, args.vocab_size)

    index_pairs = list(zip(fn, fp, tp, can))
    train_index, valid_index, test_index = dataSplit2(index_pairs, args.seed)
    train_fn_idx, train_fp_idx, train_tp_idx, train_can_idx = list2numpy_label(train_index, tag2idx)

    if args.mode == 'rcn':
        trg = [i+j for i, j in zip(tp, fp)]
        tokenized_pairs = list(zip(src, trg))
        train, valid, test = dataSplit(tokenized_pairs, args.seed)
        X_train, y_train = list2numpy(train, word2idx, tag2idx, args.max_len, args.vocab_size)
        X_valid, y_valid = list2numpy(valid, word2idx, tag2idx, args.max_len, args.vocab_size)

    args.label_size = y_train.shape[-1]

    logger.info(F'Size of Training Set: {len(X_train)}')
    logger.info(F'Size of Validation Set: {len(X_valid)}')
    logger.info(F'Size of Test Set: {len(X_test)}')

    train_data = data_utils.TensorDataset(torch.from_numpy(X_train).type(torch.LongTensor),
                                          torch.from_numpy(y_train).type(torch.float32),
                                            torch.from_numpy(train_fn_idx).type(torch.float32),
                                            torch.from_numpy(train_fp_idx).type(torch.float32),
                                            torch.from_numpy(train_tp_idx).type(torch.float32),
                                            torch.from_numpy(train_can_idx).type(torch.float32)
                                          )
    val_data = data_utils.TensorDataset(torch.from_numpy(X_valid).type(torch.LongTensor),
                                          torch.from_numpy(y_valid).type(torch.float32))
    test_data = data_utils.TensorDataset(torch.from_numpy(X_test).type(torch.LongTensor),
                                          torch.from_numpy(y_test).type(torch.float32))
    
    train_loader = data_utils.DataLoader(train_data, args.batch_size, shuffle=False, drop_last=True, num_workers=4)
    val_loader = data_utils.DataLoader(val_data, args.batch_size, shuffle=False, drop_last=True, num_workers=4)
    test_loader = data_utils.DataLoader(test_data, args.batch_size, drop_last=False)

    return train_loader, val_loader, test_loader, args

def get_data_loader(args):
    with open(os.path.join(args.data_dir, "word2idx.json"), "r") as file:
        word2idx = json.load(file)
    with open(os.path.join(args.data_dir, "tag2idx.json"), "r") as file:
        tag2idx = json.load(file)

    src, fn, fp, tp, can = load_txt_data2(args)
    trg = [i+j for i, j in zip(tp, fn)]

    tokenized_pairs = list(zip(src, trg))
    train, valid, test = dataSplit(tokenized_pairs, args.seed)

    X_train, y_train = list2numpy(train, word2idx, tag2idx, args.max_len, args.vocab_size)
    X_valid, y_valid = list2numpy(valid, word2idx, tag2idx, args.max_len, args.vocab_size)
    X_test, y_test = list2numpy(test, word2idx, tag2idx, args.max_len, args.vocab_size)

    index_pairs = list(zip(fn, fp, tp, can))
    train_index, valid_index, test_index = dataSplit(index_pairs, args.seed)
    train_fn_idx, train_fp_idx, train_tp_idx, train_can_idx = list2numpy_label(train_index, tag2idx)

    if args.mode == 'rcn':
        trg = [i+j for i, j in zip(tp, fp)]
        tokenized_pairs = list(zip(src, trg))
        train, valid, test = dataSplit(tokenized_pairs, args.seed)
        X_train, y_train = list2numpy(train, word2idx, tag2idx, args.max_len, args.vocab_size)
        X_valid, y_valid = list2numpy(valid, word2idx, tag2idx, args.max_len, args.vocab_size)

    args.label_size = y_train.shape[-1]

    logger.info(F'Size of Training Set: {len(X_train)}')
    logger.info(F'Size of Validation Set: {len(X_valid)}')
    logger.info(F'Size of Test Set: {len(X_test)}')

    train_data = data_utils.TensorDataset(torch.from_numpy(X_train).type(torch.LongTensor),
                                          torch.from_numpy(y_train).type(torch.float32),
                                            torch.from_numpy(train_fn_idx).type(torch.float32),
                                            torch.from_numpy(train_fp_idx).type(torch.float32),
                                            torch.from_numpy(train_tp_idx).type(torch.float32),
                                            torch.from_numpy(train_can_idx).type(torch.float32)
                                          )
    val_data = data_utils.TensorDataset(torch.from_numpy(X_valid).type(torch.LongTensor),
                                          torch.from_numpy(y_valid).type(torch.float32))
    test_data = data_utils.TensorDataset(torch.from_numpy(X_test).type(torch.LongTensor),
                                          torch.from_numpy(y_test).type(torch.float32))
    
    train_loader = data_utils.DataLoader(train_data, args.batch_size, shuffle=False, drop_last=True, num_workers=4)
    val_loader = data_utils.DataLoader(val_data, args.batch_size, shuffle=False, drop_last=True, num_workers=4)
    test_loader = data_utils.DataLoader(test_data, args.batch_size, drop_last=False)

    return train_loader, val_loader, test_loader, args

def load_txt_data2(args):
    max_src_len = 0
    max_trg_len = 0
    max_pos_len = 0
    max_neg_len = 0
    max_ran_len = 0
    src = []
    trg = []
    pos = []
    neg = []
    ran = []
    i = 0
    f=open(os.path.join(args.data_dir, args.data_name), 'r')
    for line in f.readlines():
        # process src and trg line 
        lineVec = line.strip().split(args.split_token)#split by 
        src_line = lineVec[0]
        trg_line = lineVec[1]   
        pos_line = lineVec[2]   
        neg_line = lineVec[3]  
        ran_line = lineVec[4]  
        src_word_list = src_line.strip().split(' ') 
        trg_word_list = trg_line.strip().split(';') 
        pos_word_list = pos_line.strip().split(';') 
        neg_word_list = neg_line.strip().split(';') 
        ran_word_list = ran_line.strip().split(';') 
        if len(src_word_list)>max_src_len:
            max_src_len = len(src_word_list)
        if len(trg_word_list)>max_trg_len:
            max_trg_len = len(trg_word_list)
        if len(pos_word_list)>max_pos_len:
            max_pos_len = len(pos_word_list)
        if len(neg_word_list)>max_neg_len:
            max_neg_len = len(neg_word_list)
        if len(ran_word_list)>max_ran_len:
            max_ran_len = len(ran_word_list)

        src.append(src_line)
        trg.append(trg_word_list)
        pos.append(pos_word_list)
        neg.append(neg_word_list)
        ran.append(ran_word_list)
        i+=1
        if i>= args.data_size:
            break

    assert len(src) == len(trg), \
        'the number of records in source and target are not the same'
    assert len(pos) == len(trg), \
        'the number of records in source and target are not the same'
    assert len(neg) == len(trg), \
        'the number of records in source and target are not the same'
    assert len(ran) == len(trg), \
        'the number of records in source and target are not the same'
    
    print('max_src_len', max_src_len)
    print('max_trg_len', max_trg_len)
    print('max_pos_len', max_pos_len)
    print('max_neg_len', max_neg_len)
    print('max_ran_len', max_ran_len)
    
    print("Finish reading %d lines" % len(src))
    return src, trg, pos, neg, ran

def load_txt_data3(args):
    max_src_len = 0
    max_trg_len = 0
    src = []
    trg = []
    i = 0
    f=open(os.path.join(args.data_dir, args.test_name), 'r')
    for line in f.readlines():
        # process src and trg line 
        lineVec = line.strip().split(args.split_token)#split by 
        src_line = lineVec[0]
        trg_line = lineVec[1]   
        src_word_list = src_line.strip().split(' ') 
        trg_word_list = trg_line.strip().split(';') 
        if len(src_word_list)>max_src_len:
            max_src_len = len(src_word_list)
        if len(trg_word_list)>max_trg_len:
            max_trg_len = len(trg_word_list)

        src.append(src_line)
        trg.append(trg_word_list)
        i+=1
        if i>= args.data_size:
            break

    assert len(src) == len(trg), \
        'the number of records in source and target are not the same'
    
    print('max_src_len', max_src_len)
    print('max_trg_len', max_trg_len)
    
    print("Finish reading %d lines" % len(src))
    return src, trg

def dataSplit(tokenized_pairs, random_seed):
    random.seed(random_seed)
    random.shuffle(tokenized_pairs)
    data_length = len(tokenized_pairs)
    train_length = int(data_length*.8)
    valid_length = int(data_length*.9)
    train, valid, test = tokenized_pairs[:train_length], tokenized_pairs[train_length:valid_length],\
                                     tokenized_pairs[valid_length:]  
    return train, valid, test

def dataSplit2(tokenized_pairs, random_seed):
    random.seed(random_seed)
    random.shuffle(tokenized_pairs)
    data_length = len(tokenized_pairs)
    valid_length = int(data_length*.9)
    train, valid = tokenized_pairs[:valid_length],\
                                     tokenized_pairs[valid_length:]  
    return train, valid

def encode_one_hot(inst, vocab_size, label_from):
    '''
    one hot for a value x, int, x>=1
    '''
    one_hots = np.zeros(vocab_size, dtype=np.float32)
    for value in inst:
        one_hots[value-label_from]=1
    return one_hots

def padding(input_list, max_seq_len, word2idx):
    padded_batch = word2idx['<pad>'] * np.ones((len(input_list), max_seq_len), dtype=np.int)
    for j in range(len(input_list)):
        current_len = len(input_list[j])
        if current_len <= max_seq_len:
            padded_batch[j][:current_len] = input_list[j]
        else:
            padded_batch[j] = input_list[j][:max_seq_len]
    return padded_batch

def list2numpy(src_trgs_pairs, word2idx, tag2idx, max_seq_len, vocab_size):
    '''
    word2id + padding + onehot
    '''
    text = []
    label = []
    for idx, (source, targets) in enumerate(src_trgs_pairs):
        src = [word2idx[w] if w in word2idx and word2idx[w] < vocab_size
               else word2idx['<unk>'] for w in source]
        trg = [tag2idx[w] for w in targets if w in tag2idx]
        text.append(src)
        label.append(trg)
    text = padding(text, max_seq_len, word2idx)
    label =  [encode_one_hot(inst, len(tag2idx), label_from=0) for inst in label] 
    text = np.array(text)
    label= np.array(label)
    print('text.shape', text.shape)
    print('label.shape', label.shape)
    return text, label

def list2numpy_label(src_trgs_pairs, tag2idx):
    pos = []
    neg = []
    ran = []
    ddd = []
    for idx, (pos_idx, neg_idx, ran_idx, ddd_idx) in enumerate(src_trgs_pairs):
        # print(pos_idx, neg_idx, ran_idx)
        pos.append([tag2idx[w] for w in pos_idx if w in tag2idx])
        neg.append([tag2idx[w] for w in neg_idx if w in tag2idx])
        ran.append([tag2idx[w] for w in ran_idx if w in tag2idx])
        ddd.append([tag2idx[w] for w in ddd_idx if w in tag2idx])
    pos = [encode_one_hot(inst, len(tag2idx), label_from=0) for inst in pos]
    neg = [encode_one_hot(inst, len(tag2idx), label_from=0) for inst in neg]
    ran = [encode_one_hot(inst, len(tag2idx), label_from=0) for inst in ran]
    ddd = [encode_one_hot(inst, len(tag2idx), label_from=0) for inst in ddd]
    pos = np.array(pos)
    neg = np.array(neg)
    ran = np.array(ran)
    ddd = np.array(ddd)
    return pos, neg, ran, ddd

def list2numpy_label_solo(trgs, tag2idx):
    pos = []
    for idx, pos_idx in enumerate(trgs):
        # print(pos_idx, neg_idx, ran_idx)
        pos.append([tag2idx[w] for w in pos_idx if w in tag2idx])
    pos = [encode_one_hot(inst, len(tag2idx), label_from=0) for inst in pos]
    pos = np.array(pos)
    return pos

def generate_lack(trg, neg):
    lack = []
    for target, observe in zip(trg, neg):
        temp = []
        for word in target:
            if word not in observe:
                temp.append(word)
        lack.append(temp)
    return lack