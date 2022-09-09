################################ IMPORTS #######################################
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

#
from torch.utils.data import Dataset, DataLoader, Subset
from torch import tensor
from constants import test_path, train_path, dev_path

import pytorch_lightning as pl

from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    'model_name': 'roberta-base',
    'dataset': 'PAWS' ,  # ['MRPC', 'Quora', 'PAWS']
    'batch_size': 64,
    'epochs': 100,
    'num_classes': 1,
    'lr': 1.5e-6,
    'warmup_percentage': 0.2,
    'w_decay': 0.001,
}


####################################### LOAD DATASET #######################################
class Pairs_Dataset(Dataset):
    def __init__(self, data_path, tokenizer,
                 y_label_column, first_message_column, second_message_column,
                 max_token_length=128):

        # init variables
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.y_label_column = y_label_column
        self.first_message_column = first_message_column
        self.second_message_column = second_message_column
        self.max_token_length = max_token_length

        # prepare data at obj initialization
        if 'msr-paraphrase-corpus' or 'paws' in self.data_path:
            self.df = pd.read_csv(data_path, sep='\t', quoting=csv.QUOTE_NONE)
        else:
            self.df = pd.read_csv(data_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        current_row = self.df.iloc[index]

        first_message = str(current_row[self.first_message_column])
        second_message = str(current_row[self.second_message_column])

        y = current_row[self.y_label_column]
        label = torch.tensor(y, dtype=torch.float32)

        # {'input_ids': tensor([[101, 487, 1663, 111, 102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]])}
        tokens_dict = self.tokenizer(first_message, second_message,
                                     padding='max_length',
                                     truncation=True,
                                     max_length=self.max_token_length,
                                     return_tensors='pt',
                                     add_special_tokens=True)

        return {'input_ids': tokens_dict['input_ids'].flatten(),
                'attention_mask': tokens_dict['attention_mask'].flatten(),
                'labels': label}


############################## CREATING A DATASET ##################################
from transformers import AutoTokenizer
model_name = config['model_name']

tokenizer = AutoTokenizer.from_pretrained(model_name)


# dataset_train = Pairs_Dataset('../data/paws/train.tsv', tokenizer, 'label', 'sentence1', 'sentence2')
# dataset_test = Pairs_Dataset('../data/paws/test.tsv', tokenizer, 'label', 'sentence1', 'sentence2')
# dataset_val = Pairs_Dataset('../data/paws/dev.tsv', tokenizer, 'label', 'sentence1', 'sentence2')
#
path = 'https://raw.githubusercontent.com/wasiahmad/paraphrase_identification/' \
       'master/dataset/msr-paraphrase-corpus/msr_paraphrase_test.txt'
dataset_test = Pairs_Dataset(path, tokenizer, 'Quality', '#1 String', '#2 String')
BATCH_SIZE = config['batch_size']
# train_loader = DataLoader(dataset_train, BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset_test, BATCH_SIZE, shuffle=False)
# val_loader = DataLoader(dataset_val, BATCH_SIZE, shuffle=False)

####################################### ML MODEL CLASS #######################################
import pytorch_lightning as pl
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup
import torch.nn as nn
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F

############################ NEURAL NETWORK ###################################

class Classifier_Model(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict=True)

        # dropout layer
        self.dropout_layer = nn.Dropout()

        # hidden and output layers
        self.hidden_layer = nn.Linear(self.pretrained_model.config.hidden_size,
                                      256)

        self.output_layer = nn.Linear(256,
                                      1)

        # initialize weights
        torch.nn.init.xavier_uniform(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform(self.output_layer.weight)

        # loss function
        self.loss_function = nn.BCEWithLogitsLoss()


    # training loop
    def forward(self, input_ids, attention_mask, labels):
        # pass to RoBerta model
        out = self.pretrained_model(input_ids, attention_mask)

        pulled_output = torch.mean(out.last_hidden_state, 1)

        # nn
        res = self.hidden_layer(pulled_output)
        res = self.dropout_layer(res)
        res = F.relu(res)
        logits = self.output_layer(res)

        # loss
        loss = 0
        if labels is not None:
            loss = self.loss_function(logits.view(-1, self.config['num_classes']),
                                      labels.view(-1, self.config['num_classes']))

        return loss, logits


    def training_step(self, batch, batch_idx):
        # unpack dictionary
        # call forward func
        loss, logits = self(**batch)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return {'loss': loss, 'predictions': logits, 'labels': batch['labels']}

    def validation_step(self, batch, batch_idx):
        # unpack dictionary
        loss, logits = self(**batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return {'val_loss': loss, 'predictions': logits, 'labels': batch['labels']}

    def test_step(self, batch, batch_idx):
        # unpack dictionary
        _, logits = self(**batch)
        return logits

    def configure_optimizers (self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['lr'],
                                      weight_decay=self.config['w_decay'])
        total_steps = self.config['train_size'] / self.config['batch_size']
        warmup_steps = np.floor(total_steps * self.config['warmup_percentage'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]


# model = Classifier_Model.load_from_checkpoint('weights-50-epochs.ckpt', config=config)
# #
# # #%%
# model.to(device)

def predict(loader, model):
    predictions = np.array([])

    with torch.no_grad():
        model.eval()

        for batch in loader:
            ids = batch['input_ids']
            mask = batch['attention_mask']
            labels = batch['labels']

            ids = ids.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            _, outputs = model(ids, mask, labels)

            pred = torch.sigmoid(outputs).detach().cpu().numpy()

            predictions = np.concatenate((predictions, pred), axis = None)

            predictions_int = np.where(predictions > 0.5, 1, 0)

    return predictions, predictions_int

# preditions_train_raw, predictions_train_int = predict(train_loader)

# train_df = pd.DataFrame({
#     'predictions_raw': preditions_train_raw,
#     'predictions': predictions_train_int,
#     'id': list(range(1, len(preditions_train_raw) + 1))
# }).to_csv('distances/train_roberta.csv', index=False)
#
# #%%%


# preditions_test_raw, predictions_test_int = predict(test_loader, model)
#
# dev_df = pd.DataFrame({
#     'predictions_raw': preditions_test_raw,
#     'predictions': predictions_test_int,
#     'id': list(range(1, len(preditions_test_raw) + 1))
# }).to_csv('distances/roberta_mrpc.csv', index=False)