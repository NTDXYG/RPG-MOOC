import torch

from DL.Config import class_number

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
import DL.DataSet
import torch.nn as nn

class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        Vocab = len(DL.DataSet.getTEXT().vocab)  ## 已知词的数量
        Dim = 300  ##每个词向量长度
        dropout = 0.1
        hidden_size = 256 #隐藏层数量
        num_layers = 2 ##双层LSTM

        self.embedding = nn.Embedding(Vocab, Dim)  ## 词向量，这里直接随机
        self.lstm = nn.LSTM(Dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, class_number)

    def forward(self, x):
        # [batch len, text size]
        x = self.embedding(x)
        # [batch size, text size, embedding]
        output, (hidden, cell) = self.lstm(x)
        # output = [batch size, text size, num_directions * hidden_size]
        output = self.fc(output[:, -1, :])  # 句子最后时刻的 hidden state
        # output = [batch size, num_classes]
        return output