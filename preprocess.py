import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import re
from nltk.tokenize import word_tokenize
import os
from collections import Counter
import random


def preprocessing(args):

    args.txt_path = args.ori_data_path
    src, trg = load_txt_data(args.txt_path, args.data_size, args.split_token)
    tokenized_pairs = list(zip(src, trg))
    word2idx, idx2word, token_freq_counter, tag2idx, idx2tag = build_vocab(tokenized_pairs, args.vocab_size)
    tag2idx, idx2tag = get_label_dictionary(trg)
    label = list2numpy(tag2idx, trg)
    co_mat = np.dot(label.T, label)

    #
    len_list = [ len(temp) for temp in trg]
    Lavg = np.mean(len_list)
    L = len(tag2idx)
    print('Lavg, L', Lavg, L)
    rho_01 = args.rho *Lavg / (L- Lavg)
    rho_10 = args.rho
    print('rho_01, rho_10', rho_01, rho_10)

    #fn
    fn_label = []
    for label_list in trg:
        fn_label.append(fn_noise(label_list, rho_10))

    #tp
    tp_label = []
    for truth, fn in zip(trg, fn_label):
        tp_label.append([item for item in truth if item not in fn])

    #fp
    fp_label = []
    for label_list in trg:
        fp_label.append(fp_noise(label_list, idx2tag, rho_01, L))

    #observed 
    obs = [i+j for i, j in zip(tp_label, fp_label)]

    #co mat
    label = list2numpy(tag2idx, obs)
    co_mat = np.dot(label.T, label)
    normalized_co_mat = normalized(co_mat)
    print(normalized_co_mat)

    #candidate label
    can_label = []
    for label_list in obs:
        can_label.append(candidate_label_gen(normalized_co_mat, label_list, tag2idx, idx2tag, args.beta))

    #save
    save_path = os.path.join(args.data_dir,'data_rho%s.txt'%str(args.rho))
    print(save_path)

    #save 
    with open(save_path, 'w') as f:
        for src_line, fn, fp, tp, can in zip(src, fn_label, fp_label, tp_label, can_label):
            fn =  ';'.join(fn)
            fp =  ';'.join(fp)
            tp =  ';'.join(tp)
            can =  ';'.join(can)
            all_line = args.split_token.join([src_line, fn, fp, tp, can])
            f.write(all_line + '\n')

def load_txt_data(txt_path, data_size, split_token):
    max_src_len = 0
    max_trg_len = 0
    src = []
    trg = []
    i = 0
    f=open(txt_path, 'r')
    for line in f.readlines():
        # process src and trg line 
        lineVec = line.strip().split(split_token)#split by 
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
        if i>= data_size:
            break

    assert len(src) == len(trg), \
        'the number of records in source and target are not the same'
    
    print('max_src_len', max_src_len)
    print('max_trg_len', max_trg_len)
    
    print("Finish reading %d lines" % len(src))
    return src, trg


def build_vocab(tokenized_src_trg_pairs, vocab_size):
    '''
    Build the vocabulary from the training (src, trg) pairs
    :param tokenized_src_trg_pairs: list of (src, trg) pairs
    :return: word2idx, idx2word, token_freq_counter
    '''
    # Build vocabulary from training src and trg
    print("Building vocabulary from training data")
    token_freq_counter = Counter()
    token_freq_counter_tag = Counter()
    for src_word_list, trg_word_lists in tokenized_src_trg_pairs:
        token_freq_counter.update(src_word_list)
        token_freq_counter_tag.update(trg_word_lists)

    # Discard special tokens if already present
    special_tokens = ['<pad>', '<unk>']
    num_special_tokens = len(special_tokens)

    for s_t in special_tokens:
        if s_t in token_freq_counter:
            del token_freq_counter[s_t]

    word2idx = dict()
    idx2word = dict()
    for idx, word in enumerate(special_tokens):
        word2idx[word] = idx
        idx2word[idx] = word

    sorted_word2idx = sorted(token_freq_counter.items(), key=lambda x: x[1], reverse=True)

    sorted_words = [x[0] for x in sorted_word2idx]

    for idx, word in enumerate(sorted_words):
        word2idx[word] = idx + num_special_tokens

    for idx, word in enumerate(sorted_words):
        idx2word[idx + num_special_tokens] = word

    tag2idx = dict()
    idx2tag = dict()

    sorted_tag2idx = sorted(token_freq_counter_tag.items(), key=lambda x: x[1], reverse=True)

    sorted_tags = [x[0] for x in sorted_tag2idx]

    for idx, tag in enumerate(sorted_tags):
        tag2idx[tag] = idx

    for idx, tag in enumerate(sorted_tags):
        idx2tag[idx] = tag       
        
    print("Total vocab_size: %d, predefined vocab_size: %d" % (len(word2idx), vocab_size))
    print("Total tag_size: %d" %len(tag2idx))   
    
    return word2idx, idx2word, token_freq_counter, tag2idx, idx2tag


def get_label_dictionary(trg):
    tag2idx = dict()
    idx2tag = dict()
    token_freq_counter = Counter()
    for label_list in trg:
        token_freq_counter.update(label_list)
    sorted_tag2idx = sorted(token_freq_counter.items(), key=lambda x: x[1], reverse=True)

    sorted_tags = [x[0] for x in sorted_tag2idx]

    for idx, tag in enumerate(sorted_tags):
        tag2idx[tag] = idx

    for idx, tag in enumerate(sorted_tags):
        idx2tag[idx] = tag       
        
    print("Total tag_size: %d" %len(tag2idx))       
    return tag2idx, idx2tag

def encode_one_hot(inst, vocab_size):
    '''
    one hot for a value x, int, x>=1
    '''
    one_hots = np.zeros(vocab_size, dtype=np.float32)
    for value in inst:
        one_hots[value]=1
    return one_hots

def list2numpy(tag2idx, trg):
    label = []
    for idx, targets in enumerate(trg):
        label_list = [tag2idx[w] for w in targets if w in tag2idx]
        label.append(label_list)
    label =  [encode_one_hot(inst, len(tag2idx)) for inst in label] 
    label= np.array(label)
    print('label.shape', label.shape)
    return label

def normalized(co_mat):
    for i, row in enumerate(co_mat):
        co_mat[i,i] = 0
        if sum(row)<=0:
            continue
        co_mat[i] = row/sum(row)
    return co_mat


def fn_noise(label_list, rho_10):
    fn_label = []
    for label in label_list:
        if np.random.binomial(1, rho_10)==1:
            fn_label.append(label)
    return fn_label

def fp_noise(label_list, idx2tag, rho_01, L):
    fp_num = np.random.binomial(L, rho_01)
    #sample fp_num indices from 0 to L-1
    fp_idx = np.random.choice(L, fp_num, replace=False)
    fp_label = [idx2tag[i] for i in fp_idx]
    # remove the ground truth label in fp_label 
    for label in label_list:
        if label in fp_label:
            fp_label.remove(label)
    return fp_label

def candidate_label_gen(normalized_co_mat, label_list, tag2idx, idx2tag, beta):  
    # instacne level correaltion
    instance_specific_corr = np.zeros(len(tag2idx))
    for label in label_list:
        idx = tag2idx[label]
        label_num = len(label_list)
        if label_num<1: 
            continue
        instance_specific_corr += normalized_co_mat[idx] / label_num
    
    can_label = [] 
    # find the index that the corr > beta of instance_specific_corr
    for i, corr in enumerate(instance_specific_corr):
        if corr > beta:
            label = idx2tag[i]
            if label not in label_list:
                can_label.append(label)
    return can_label
