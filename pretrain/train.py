import os
import random
import pandas as pd
from tqdm import tqdm

from pretrain.model import BertClassifier
from utils import set_seed

set_seed(1234)

train_df = pd.read_csv('../data/train_ir.csv')
eval_df = pd.read_csv('../data/val_ir.csv')
test_df = pd.read_csv('../data/test_ir.csv')

train_df.columns = ["input_text", "target_text"]
eval_df.columns = ["input_text", "target_text"]
test_df.columns = ["input_text", "target_text"]

# first fine-tune CodeBert
classifier = BertClassifier(
        model_path='./bert-base-chinese',
        tokenizer_path='./bert-base-chinese',
        max_len=300,
        n_classes=2,
        epochs=15,
        model_save_path='./models/rgb_bert.pt',
        batch_size=4,
        learning_rate=2e-5
)
#
# classifier.preparation(
#         X_train=list(train_df['input_text']),
#         y_train=list(train_df['target_text']),
#         X_valid=list(eval_df['input_text']),
#         y_valid=list(eval_df['target_text'])
#     )
#
# classifier.train()

texts = list(test_df['input_text'])
labels = list(test_df['target_text'])

predictions = []
for i in tqdm(range(len(texts))):
        predictions.append(classifier.predict(texts[i]))

label_2_category = {0: 'Confusion', 1: 'Non-Confusion'}
category_2_label = {v:k for k, v in label_2_category.items()}

label_list = category_2_label.keys()
from sklearn.metrics import classification_report
print('rgb-bert')
print(classification_report(labels, predictions, target_names=label_list, digits=5))