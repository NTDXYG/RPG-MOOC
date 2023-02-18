import os
import csv
import pandas as pd
import numpy as np
import re
import jieba

from utils import set_seed

set_seed(1234)
# jieba分词，去除停用词
def cut_word(text, stop_words_file='data/stopword.txt'):
    stop_words = [line.strip() for line in open(stop_words_file, 'r', encoding='utf-8').readlines()]
    text = str(text)
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    text = jieba.lcut(text)
    # text = [word for word in text if word not in stop_words]
    return ' '.join(text)

label_2_category = {0: 'Confusion', 1: 'Non-Confusion'}
category_2_label = {v:k for k, v in label_2_category.items()}

data_raw = pd.read_csv('data/train.csv')
train_X = data_raw['comment'].tolist()
train_X = [cut_word(x) for x in train_X]
train_y = data_raw['label'].tolist()

data_raw = pd.read_csv('data/test.csv')
test_X = data_raw['comment'].tolist()
test_X = [cut_word(x) for x in test_X]
test_y = data_raw['label'].tolist()

from sklearn.base import TransformerMixin
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

count_vec = CountVectorizer(max_df=0.9, min_df=2)

tfidf_vec = TfidfTransformer()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# 导入逻辑回归模型
from sklearn.linear_model import LogisticRegression
# 导入支持向量机模型
from sklearn.svm import SVC
# 导入随机森林模型
from sklearn.ensemble import RandomForestClassifier

def Classifier():
    return Pipeline([
        ('count_vec', count_vec),
        ('tfidf_vec', tfidf_vec),
        ('lr', LogisticRegression())
    ])

mnbc_clf = Classifier()
mnbc_clf.fit(train_X, train_y)
print('logistic regression: ')
print(classification_report(test_y, mnbc_clf.predict(test_X), target_names=category_2_label.keys(), digits=5))


def Classifier():
    return Pipeline([
        ('count_vec', count_vec),
        ('tfidf_vec', tfidf_vec),
        ('svc', SVC())
    ])

mnbc_clf = Classifier()
mnbc_clf.fit(train_X, train_y)
print('svm: ')
print(classification_report(test_y, mnbc_clf.predict(test_X), target_names=category_2_label.keys(), digits=5))


def Classifier():
    return Pipeline([
        ('count_vec', count_vec),
        ('tfidf_vec', tfidf_vec),
        ('dt', DecisionTreeClassifier())
    ])

mnbc_clf = Classifier()
mnbc_clf.fit(train_X, train_y)
print('Decision Tree: ')
print(classification_report(test_y, mnbc_clf.predict(test_X), target_names=category_2_label.keys(), digits=5))

def Classifier():
    return Pipeline([
        ('count_vec', count_vec),
        ('tfidf_vec', tfidf_vec),
        ('rf', RandomForestClassifier())
    ])

mnbc_clf = Classifier()
mnbc_clf.fit(train_X, train_y)
print('RandomForest: ')
print(classification_report(test_y, mnbc_clf.predict(test_X), target_names=category_2_label.keys(), digits=5))
