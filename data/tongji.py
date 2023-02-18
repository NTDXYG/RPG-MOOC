import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby

from transformers import RobertaTokenizer
import jieba

from utils import set_seed

set_seed(1234)
# jieba分词，去除停用词
def cut_word(text):
    text = str(text)
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    text = jieba.lcut(text)
    return text
# tokenizer = RobertaTokenizer.from_pretrained("D:\\new_idea\\SmartContract2NL\\model\\codebert_sc")


df = pd.read_csv("train.csv",)
code_list = df['comment'].tolist()
print(len(code_list))
# NL_RX : NL 20; REGEX 50; AST 50
# ast_len_list = [len(tokenizer.tokenize(ast)) for ast in ast_list]
ast_len_list = [len(cut_word(ast)) for ast in code_list]

commutes = pd.Series(ast_len_list)


commutes.plot.hist(grid=True, bins=25, rwidth=0.9,
                   color='#607c8e')
# plt.title('Commute Times for 1,000 Commuters')
plt.xlabel('Feedback Sequence Length')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.75)
plt.savefig('data_desc.png', dpi = 800)
plt.show()


def getBili(num, demo_list):
    s = 0
    for i in range(len(demo_list)):
        if(demo_list[i] < num):
            s += 1
    print('<'+str(num)+'比例为'+str(s/len(demo_list)))

from numpy import *
code_len_list = ast_len_list
b = mean(code_len_list)
c = median(code_len_list)
counts = np.bincount(code_len_list)
d = np.argmax(counts)
print('平均值'+str(b))
print('众数'+str(d))
print('中位数'+str(c))

getBili(32,code_len_list)
getBili(64,code_len_list)
getBili(100,code_len_list)