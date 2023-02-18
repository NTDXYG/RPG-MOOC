import torch
from torchtext import data
import jieba
from Config import *

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def x_tokenize(x):
    return jieba.lcut(x)

TEXT = data.Field(sequential=True, tokenize=x_tokenize,fix_length=fix_length,
            use_vocab=True)

LABEL = data.Field(sequential=False,
            use_vocab=False)

train, dev, test = data.TabularDataset.splits(path=data_path,
                                              train=train_file,
                                              validation=valid_file,
                                              test=test_file,
                                              format='csv',
                                              skip_header=True,
                                              csv_reader_params={'delimiter':','},
                                              fields=[('text',TEXT),('label',LABEL)])

TEXT.build_vocab(train)

train_iter, val_iter, test_iter = data.BucketIterator.splits((train,dev,test),
                                                             batch_size = batch_size,
                                                             shuffle=True,
                                                             sort=False,
                                                             sort_within_batch=False,
                                                             repeat=False)

def getTEXT():
    return TEXT
def getLabel():
    return LABEL
def getIter():
    return train_iter, val_iter, test_iter