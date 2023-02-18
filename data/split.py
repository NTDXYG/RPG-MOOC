import random

import pandas as pd

from utils import set_seed

def read_txt_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data

set_seed(1234)
data_list = []
comment_list = read_txt_data('./mooc.txt')
comment_list = [comment.strip() for comment in comment_list]
for comment in comment_list:
    text = comment.split('\t')[0]
    label = comment.split('\t')[1]
    data_list.append([text, label])

# 按照8:1:1的比例划分训练集、验证集和测试集，保证每个类别的比例一致
def split_data(data_list, train_rate, val_rate):
    train_data = []
    val_data = []
    test_data = []
    for i in ['Confusion', 'Non-Confusion']:
        class_data = [data for data in data_list if data[1] == str(i)]
        train_num = int(len(class_data) * train_rate)
        val_num = int(len(class_data) * val_rate)
        train_data.extend(class_data[:train_num])
        val_data.extend(class_data[train_num:train_num + val_num])
        test_data.extend(class_data[train_num + val_num:])
    return train_data, val_data, test_data

train_data, val_data, test_data = split_data(data_list, 0.8, 0.1)

random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

df = pd.DataFrame(train_data, columns=['comment', 'label'])
df.to_csv('./train.csv', index=False)
df = pd.DataFrame(val_data, columns=['comment', 'label'])
df.to_csv('./val.csv', index=False)
df = pd.DataFrame(test_data, columns=['comment', 'label'])
df.to_csv('./test.csv', index=False)