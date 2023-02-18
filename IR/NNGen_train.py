import re

import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

label_2_category = {0: 'Confusion', 1: 'Non-Confusion'}
category_2_label = {v:k for k, v in label_2_category.items()}

def jaccard_similarity(list1, list2):
    """Calculate jaccard similarity"""
    a = set(list1)
    b = set(list2)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def clean(text):
    text = str(text)
    return ' '.join(jieba.lcut(text))

def find_mixed_nn(simi, diffs, test_diff, idx, bleu_thre):
    """Find the nearest neighbor using cosine simialrity and bleu score"""
    candidates = simi.argsort()[-bleu_thre:][::-1]
    max_score = 0
    max_idx = 0
    for j in candidates:
        if (j != idx):
            score = jaccard_similarity(diffs[j].split(), test_diff.split())
            if score > max_score:
                max_score = score
                max_idx = j
    return max_idx

def nngen(train_codes, start, test_codes):
    counter = TfidfVectorizer()
    train_matrix = counter.fit_transform(train_codes)
    test_matrix = counter.transform(test_codes)
    similarities = cosine_similarity(test_matrix, train_matrix)
    test_nls = []
    for idx, test_simi in enumerate(similarities):
        max_idx = find_mixed_nn(test_simi, train_codes, test_codes[idx], start+idx, bleu_thre=5)
        test_nls.append(max_idx)
    return test_nls

df = pd.read_csv('../data/train.csv')
train_comments = df['comment'].tolist()
train_categories = df['label'].tolist()
raw_train_comments = train_comments
train_comments = [clean(text) for text in train_comments]

df = pd.read_csv('../data/train.csv')
test_comments = df['comment'].tolist()
test_categories = df['label'].tolist()
raw_test_comments = test_comments
test_comments = [clean(text) for text in test_comments]

data_list = []
out_categories = []
for start in tqdm(range(0,len(test_comments), 1000), total=len(test_comments)/1000):
    out_categories.extend(nngen(train_comments, start, test_comments[start: start + 1000]))
for i,idx in enumerate(out_categories):
    data_list.append(["<E>{'评论':"+str(raw_train_comments[idx]) +";'标签':"+ str(train_categories[idx])+"}</E> "+raw_test_comments[i], test_categories[i]])
out_categories = [train_categories[x] for x in out_categories]
df = pd.DataFrame(data_list, columns=['comment', 'label'])
df.to_csv('../data/train_ir.csv', index=False)
print(classification_report(test_categories, out_categories, target_names=category_2_label.keys(), digits=3))