import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from tqdm import tqdm
import DataSet
import numpy as np
from sklearn.metrics import accuracy_score

from Config import *
from model.TextCNN import TextCNN
from model.TextRNN import TextRNN
from model.TextRNN_Attention import TextRNN_Attention
from model.Transformer import Transformer
from utils import set_seed


def eval_model(test_iter, name, device):
    model = torch.load('done_model/'+name+'_model.pkl')
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    accuracy = 0
    y_true = []
    y_pred = []
    total_test_num = len(test_iter.dataset)
    for batch in test_iter:
        feature = batch.text
        target = batch.label
        with torch.no_grad():
            feature = torch.t(feature)
        feature, target = feature.to(device), target.to(device)
        out = model(feature)
        loss = F.cross_entropy(out, target)
        total_loss += loss.item()
        accuracy += (torch.argmax(out, dim=1)==target).sum().item()
        y_true.extend(target.cpu().numpy())
        y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
    print('>>> Test loss:{}, Accuracy:{} \n'.format(total_loss/total_test_num, accuracy/total_test_num))
    score = accuracy_score(y_true, y_pred)
    print(score)
    from sklearn.metrics import classification_report
    print(name)
    print(classification_report(y_true, y_pred, target_names=label_list, digits=5))

def train_model(train_iter, dev_iter, model, name, early_stop, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)
    model.train()
    best_acc = 0
    count = 0
    # dice_loss = GHMC_Loss(bins=13, alpha=0.75)
    print('training...')
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        accuracy = 0
        total_train_num = len(train_iter.dataset)
        progress_bar = tqdm(enumerate(train_iter), total=len(train_iter))
        for i,batch in progress_bar:
            feature = batch.text
            target = batch.label
            with torch.no_grad():
                feature = torch.t(feature)
            feature, target = feature.to(device), target.to(device)
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            # loss = dice_loss(logit, target)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            total_loss += loss.item()
            accuracy += (torch.argmax(logit, dim=1) == target).sum().item()
            progress_bar.set_description(
            f'loss: {loss.item():.3f}')
        print('>>> Epoch_{}, Train loss is {}, Accuracy:{} \n'.format(epoch,loss.item()/total_train_num, accuracy/total_train_num))
        model.eval()
        total_loss = 0.0
        accuracy = 0
        total_valid_num = len(dev_iter.dataset)
        progress_bar = tqdm(enumerate(dev_iter), total=len(dev_iter))
        for i, batch in progress_bar:
            feature = batch.text
            target = batch.label
            with torch.no_grad():
                feature = torch.t(feature)
            feature, target = feature.to(device), target.to(device)
            out = model(feature)
            loss = F.cross_entropy(out, target)
            total_loss += loss.item()
            accuracy += (torch.argmax(out, dim=1)==target).sum().item()
        print('>>> Epoch_{}, Valid loss:{}, Accuracy:{} \n'.format(epoch, total_loss/total_valid_num, accuracy/total_valid_num))
        if(accuracy/total_valid_num > best_acc):
            print('save model...')
            count = 0
            best_acc = accuracy/total_valid_num
            saveModel(model, name=name)
        else:
            count += 1
            if(count==early_stop):
                print('early stop...')
                break

def saveModel(model,name):
    torch.save(model, 'done_model/'+name+'_model.pkl')

name = 'Transformer'
model = Transformer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
train_iter, val_iter, test_iter = DataSet.getIter()
early_stop = 3

if __name__ == '__main__':
    set_seed(1234)
    train_model(train_iter, val_iter, model, name, early_stop, device)
    eval_model(test_iter, name, device)