# data preprocess
data_path = '../data/'
train_file = 'train.csv'
valid_file = 'val.csv'
test_file = 'test.csv'
fix_length = 128
batch_size = 32
# data label list
label_2_category = {0: 'Confusion', 1: 'Non-Confusion'}
category_2_label = {v:k for k, v in label_2_category.items()}

label_list = category_2_label.keys()
class_number = len(label_list)
# train details
epochs = 30
learning_rate = 1e-3