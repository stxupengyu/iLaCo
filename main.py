import os
import logging
import argparse
import time
import random
import numpy as np
import dataset, train, test, preprocess
from optimizers import *
from model import LSFL
from utils import time_since, cprint
import torch
from torch.optim import Adam
import datetime

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    #noise generation  
    parser.add_argument('--rho', type=float, 
                        help='noise rate', default=0.2)
    parser.add_argument('--preprocess', type=bool, default=False,   #True False
                        help="pre-process")

    #laco
    parser.add_argument('--alpha', type=float, 
                        help='alpha', default=0.7)
    parser.add_argument('--beta', type=float, 
                        help='beta', default=0.01)    

    #dataset
    parser.add_argument("--mode", default="rcn", type=str,
                        help="mode")
    parser.add_argument("--dataset", default="XXXXXX", type=str,
                        help="The input data directory")   
    parser.add_argument("--ori_data_dir", default="/data/XXXXXX/", type=str,
                        help="The input data directory")   
    parser.add_argument("--data_dir", default="/data/XXXXXX/NMLL", type=str,
                        help="The input data directory")
    parser.add_argument("--code_dir", default="/home/XXXXXX/code/nmll/ilaco", type=str,
                        help="The input data directory")    
    parser.add_argument("--data_name", default="train_rho0.6.txt", type=str,
                        help="The input data directory")
    parser.add_argument("--test_name", default="test.txt", type=str,
                        help="The input data directory")
    parser.add_argument("--output_name", default="output.txt", type=str,
                        help="The input data directory")
    parser.add_argument("--split_token", default="<Tags>:", type=str,
                        help="The input data directory")
    parser.add_argument('--max_len', type=int, default=500,
                        help="max length of document")
    parser.add_argument('--vocab_size', type=int, default=500000,
                        help="vocabulary size of dataset")
    parser.add_argument('--data_size', type=int, default=999999,
                        help="vocabulary size of dataset")     

    #model
    parser.add_argument('--emb_size', type=int, default=300,
                        help="embedding size")
    parser.add_argument('--hidden_size', type=int, default=256,
                        help="hideden size of LSTM")
    parser.add_argument('--feat_size', type=int, default=300,
                        help="feature size of LSFL")
    parser.add_argument("--dropout", default=0.5, required=False, type=float,
                        help="dropout of LSFL")
    parser.add_argument("--lr", default=1e-3, required=False, type=float,
                        help="learning rate of LSFL")

    #training
    parser.add_argument('--gpuid', type=int, default=7,
                        help="gpu id")
    parser.add_argument('--epochs', type=int, default=50,
                        help="epoch of LSFL")
    parser.add_argument('--early_stop_tolerance', type=int, default=5,
                        help="early stop of LSFL")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="batch size of LSFL")
    parser.add_argument('--swa_warmup', type=int, default=10,
                        help="begin epoch of swa")
    parser.add_argument('--swa_mode', type=bool, default=False,
                        help="use swa strategy")
    parser.add_argument('--gradient_clip_value', type=int, default=5.0,
                        help="gradient clip")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--test_each_epoch', type=bool, default=False,#True False
                        help="test performance on each epoch")


    #pretrained
    parser.add_argument('--pretrained', type=bool, default=False,   #True False
                        help="use pretrained LSFL model")
    parser.add_argument("--pretrained_path", default='xxx', type=str,
                        help="path of pretrained LSFL model")
    parser.add_argument('--debug', type=bool, default=False,   #True False
                        help="use pretrained LSFL model")
    args = parser.parse_args()


    #paths
    args.ori_data_path = os.path.join(os.path.join(args.ori_data_dir, args.dataset), '%s.txt'%args.dataset)
    args.model_dir = os.path.join(args.data_dir, 'model')
    args.data_dir = os.path.join(args.data_dir, args.dataset)
    args.save_dir = os.path.join(args.code_dir, 'output')
    args.plot_dir = os.path.join(args.code_dir, 'plot')
    args.output_txt = os.path.join(args.save_dir, args.mode+'_'+args.output_name)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    args.timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    args.model_path = os.path.join(args.model_dir, "lstm_%s_%s.pth" %(args.mode, args.timemark))
    args.history_path = os.path.join(args.model_dir, "history_%s_%s.npy" %(args.mode, args.timemark))
    args.data_name = 'train_rho%s.txt'%str(args.rho)


    #for debug
    if args.debug == True:
        args.data_size = 2000
        args.epochs = 15
        args.emb_size = 10
        args.hidden_size = 10
        args.feat_size = 10
        args.max_len = 200
        args.early_stop_tolerance = 15

    #log para
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in args.__dict__.items()]

    #for reproduce
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    pipeline(args)

def pipeline(args):

    #Noise Generate
    if args.preprocess == True:
        preprocess.preprocessing(args)
        # exit()

    #Dataset
    start_time = time.time()
    logger.info('Data Loading')
    train_loader, val_loader, test_loader, args = dataset.get_data_loader2(args)
    load_data_time = time_since(start_time)
    logger.info('Time for loading the data: %s' %load_data_time)

    #Model
    start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] ='%d'%args.gpuid
    args.device = torch.device('cuda:0')
    # model = JOCOR(args)
    model = LSFL(args)
    model = model.to(args.device)
    optimizer = DenseSparseAdam(model.parameters())

    model2 = LSFL(args)
    model2 = model2.to(args.device)
    optimizer2 = DenseSparseAdam(model2.parameters())

    #Training
    train.train(model, optimizer, model2, optimizer2, train_loader, val_loader, test_loader, args)
    training_time = time_since(start_time)
    logger.info('Time for training: %s' %training_time)
    logger.info(f'Best Model Path: {args.model_path}')
    model.load_state_dict(torch.load(args.model_path, map_location=args.device)) 

    #Predicting
    logger.info('Predicting')
    result = test.valid(model2, test_loader, None, args)
    logger.info(f'Final Test Result: {result}')
    with open(args.output_txt, 'a') as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        f.write(' |mode:'+args.mode)
        f.write(' |alpha:'+str(args.alpha))
        f.write(' |beta:'+str(args.beta))
        f.write(' |result:' +' '.join([str(i) for i in result])+'\n')
        f.close()

    # delete best model for saving memory    
    # os.remove(args.model_path)

if __name__ == '__main__':
    main()
