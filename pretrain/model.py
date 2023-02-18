import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizerFast, BertConfig, AdamW, get_linear_schedule_with_warmup

from pretrain.dataset import CustomDataset


class BertClassifier:
    def __init__(self, model_path, tokenizer_path, max_len=256, n_classes=2, epochs=15,
                 model_save_path='/save/codebert.pt', batch_size = 64, learning_rate = 2e-5):
        self.config = BertConfig.from_pretrained(model_path, num_labels = n_classes)
        self.model = BertForSequenceClassification.from_pretrained(model_path, config=self.config)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        if os.path.exists(self.model_save_path):
            self.model.load_state_dict(torch.load(self.model_save_path))
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)

    def preparation(self, X_train, y_train, X_valid, y_valid):
        # create datasets
        self.train_set = CustomDataset(X_train, y_train, self.tokenizer, max_len=self.max_len)
        self.valid_set = CustomDataset(X_valid, y_valid, self.tokenizer, max_len=self.max_len)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        t_total = len(self.train_loader) // self.epochs
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=1e-8)
        num_train_optimization_steps = self.epochs * len(self.train_loader)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=num_train_optimization_steps)


    def fit(self):
        self.model = self.model.train()
        losses = []
        correct_predictions = 0
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Training')
        for i, data in progress_bar:
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )
            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            progress_bar.set_description(
                f'loss: {loss.item():.3f}')
        train_acc = correct_predictions.double() / len(self.train_set)
        train_loss = np.mean(losses)
        return train_acc, train_loss

    def eval(self):
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for data in self.valid_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=targets
                )

                preds = torch.argmax(outputs.logits, dim=1)
                loss = outputs.loss
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        val_acc = correct_predictions.double() / len(self.valid_set)
        val_loss = np.mean(losses)
        return val_acc, val_loss

    def train(self):
        best_accuracy = 0
        count = 0
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_acc, train_loss = self.fit()
            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval()
            print(f'Val loss {val_loss} accuracy {val_acc}')
            print('-' * 10)

            if val_acc > best_accuracy:
                torch.save(self.model.state_dict(), self.model_save_path)
                best_accuracy = val_acc
                count = 0
            else:
                count += 1
                if count == 3:
                    print('Early stopping')
                    break

        self.model.load_state_dict(torch.load(self.model_save_path))

    def predict(self, text):

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )

        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        return prediction