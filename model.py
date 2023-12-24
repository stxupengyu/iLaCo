import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from module import Encoder, Head
    
class JOCOR(nn.Module):

    def __init__(self, args):
        super(JOCOR, self).__init__()
        self.model1 = LSFL(args)
        self.model2 = LSFL(args)

class LSFL(nn.Module):

    def __init__(self, args):
        super(LSFL, self).__init__()
        self.emb = Embedding(vocab_size = args.vocab_size, emb_size=args.emb_size, emb_trainable=True)
        self.extractor = Extractor(args)
        self.clf = Classifier(args)

    def forward(self, input_id):
        emb_out, lengths, masks = self.emb(input_id)
        representation = self.extractor(emb_out, lengths, masks)
        logit = self.clf(representation)
        return logit

class Embedding(nn.Module):
    def __init__(self, vocab_size=None, emb_size=None, emb_init=None, emb_trainable=True, padding_idx=0, dropout=0.2):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx, sparse=True)
        self.emb.weight.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.padding_idx = padding_idx

    def forward(self, inputs):
        emb_out = self.dropout(self.emb(inputs))
        lengths, masks = (inputs != self.padding_idx).sum(dim=-1), inputs != self.padding_idx
        return emb_out[:, :lengths.max()], lengths, masks[:, :lengths.max()]

class Extractor(nn.Module):
    def __init__(self, args):
        super(Extractor,self).__init__()
        self.encoder = Encoder(args)
        self.head = Head(args)

    def forward(self, inputs, lengths, masks):
        representation = self.encoder(inputs, lengths, masks)
        representation = self.head(representation)
        return representation

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.output_layer = torch.nn.Linear(args.feat_size, 1, bias=False)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, representation): #N*L*D
        return torch.sigmoid(torch.squeeze(self.output_layer(representation),-1))



