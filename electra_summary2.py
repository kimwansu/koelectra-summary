import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraModel, ElectraForSequenceClassification, AdamW
from tqdm import tqdm

from config import corpus_dir

cuda = torch.device('cuda:0')
cpu = torch.device('cpu')
device = cuda if torch.cuda.is_available() else cpu

model_name = 'monologg/koelectra-small-v2-discriminator'

#model = ElectraForSequenceClassification.from_pretrained(model_name).to(device)
model = ElectraModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

i = 0

# 문서 데이터, 정답 라벨 파일 로드
with open(corpus_dir + 'doc_summary4.json', 'r', encoding='utf-8') as f:
    j_doc = json.load(f)

# 문서 id 목록 로드
with open(corpus_dir + 'splitted_doclist.json', 'r', encoding='utf-8') as f:
    j_list = json.load(f)

train_sets = j_doc

# 테스트용 문서 id
test_doc_ids = j_list['test']

# 학습용 문서 id
train_doc_ids = []
for ids in j_list['train']:
    train_doc_ids += ids


def load_dataset(doc_ids):
    dataset = []
    for i, doc_id in enumerate(tqdm(doc_ids)):
        sents = j_doc[doc_id]['sents']
        encoded_sents = []
        max_in_sent = 0
        minibatch = []
        y_answers = j_doc[doc_id]['labels']
        assert(len(sents) == len(y_answers))
        for sent, y in zip(sents, y_answers):
            inputs = tokenizer(
                sent,
                return_tensors='pt',
                truncation=True,
                max_length=128,
                pad_to_max_length=True,
                add_special_tokens=True
            )

            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]
            segment_emb = torch.LongTensor([i % 2] * len(input_ids))
            minibatch.append([input_ids, attention_mask, segment_emb, [1 if a == y else 0 for a in range(2)]])

        input_ids_batch = []
        attention_batch = []
        segment_batch = []
        y_answer_batch = []
        for iid, am, se, ya in minibatch:
            input_ids_batch.append(iid)
            attention_batch.append(am)
            segment_batch.append(se)
            y_answer_batch.append(ya)
        
        input_ids_batch = torch.stack(input_ids_batch)
        attention_batch = torch.stack(attention_batch)
        segment_batch = torch.stack(segment_batch)

        dataset.append([input_ids_batch, attention_batch, segment_batch, y_answer_batch])

    return dataset


train_dataset = load_dataset(train_doc_ids)
test_dataset = load_dataset(test_doc_ids)

print('load.')

n_epochs = 100

#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

lstm = torch.nn.LSTM(
    input_size=256,
    hidden_size=32,
    num_layers=1,
    batch_first=True,
    bidirectional=True
).to(device)

linear = torch.nn.Linear(64, 2).to(device)
torch.nn.init.kaiming_uniform_(linear.weight)

class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        self.n_layers = 1
        self.hidden_dim = 32
        self.lstm = torch.nn.LSTM(
            input_size=256,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        """
        self.layers = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(32*2, 2)
        )
        """
        self.linear = torch.nn.Linear(32*2, 2)
        torch.nn.init.kaiming_uniform_(self.linear.weight)

    
    def forward(self, x):
        y, _ = self.lstm(x)
        #y = self.layers(y)
        y = self.linear(y)
        return y

lstm = MyLSTM().to(device)

lr = 0.002
#lr = 1e-3
#optimizer = AdamW(model.parameters(), lr=lr)
optimizer = optim.SGD(lstm.parameters(), lr=lr, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)


def run_train(dataset):
    lstm.load_state_dict(torch.load('model_electra_j_epoch209.pt'))
    for i in range(n_epochs):
        avg_loss = 0

        correct = 0
        total = 0

        batches = 0

        lstm.train()
        random.shuffle(dataset)
        for iid_batch, am_batch, segment_emb, y_batch in tqdm(dataset):
            optimizer.zero_grad()

            y_batch = torch.LongTensor(y_batch).to(device)
            y_pred = model(iid_batch.to(device), attention_mask=am_batch.to(device), token_type_ids=segment_emb.to(device))[0]

            y_cls_head = y_pred[:, 0].unsqueeze(1)
            y_pred = lstm(y_cls_head)

            y_pred = F.softmax(y_pred)
            #y_pred = F.sigmoid(y_pred).squeeze()
            y_pred = y_pred.squeeze()

            loss = criterion(y_pred, y_batch.float())
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(dataset)

            y_batch = torch.max(y_batch, 1)[1]
            predicted = torch.max(y_pred, 1)[1]
            correct += (predicted == y_batch).sum()
            total += len(y_batch)

            del y_pred
            del y_batch
            torch.cuda.empty_cache()

            """
            batches += 1
            if batches % 100 == 0:
                print('\nbatch loss: ', avg_loss, '  acc: ', correct.float() / total)
            """

        scheduler.step(avg_loss)

        print(f'Epoch {i+1+209} : train_loss: {avg_loss}, acc: {correct.float() / total}')
        torch.save(lstm.state_dict(), f'model_electra_j_epoch{i+1+209}.pt')


def test_model(state_fname):
    lstm.load_state_dict(torch.load(state_fname))
    
    with torch.no_grad():
        lstm.eval()

        test_correct = 0
        test_total = 0


        right_total = 0
        pred_total = 0
        answ_total = 0

        dright_total = 0
        dpred_total = 0

        for iid_batch, am_batch, segment_emb, y_batch in tqdm(test_dataset):
            y_batch = torch.LongTensor(y_batch).to(device)
            output =  model(iid_batch.to(device), attention_mask=am_batch.to(device), token_type_ids=segment_emb.to(device))
            y_pred = output[0]

            y_cls_head = y_pred[:, 0].unsqueeze(1)
            y_pred = lstm(y_cls_head)
            #y_pred = F.sigmoid(y_pred).squeeze()
            y_pred = F.softmax(y_pred)
            y_pred = y_pred.squeeze()

            tmp_y0 = y_pred.T[0]
            tmp_y1 = y_pred.T[1]
            tmp_y = tmp_y1 - tmp_y0

            top_id = sorted(range(len(tmp_y)), key=lambda i:tmp_y[i])[-6:][::-1]
            double_pred = np.array([1 if i in top_id else 0 for i in range(len(y_pred))])
            

            y_batch = torch.max(y_batch, 1)[1].to(cpu)
            predicted = torch.max(y_pred, 1)[1].to(cpu)
            test_correct += (predicted == y_batch).sum()
            test_total += len(y_batch)

            right_total += np.sum(y_batch.numpy() * predicted.numpy() == 1)
            pred_total += np.sum(predicted.numpy())
            answ_total += np.sum(y_batch.numpy())

            dright_total += np.sum(y_batch.numpy() * double_pred == 1)
            dpred_total += np.sum(double_pred)
        
        print(f'Accuracy: {test_correct.float() / test_total}')
        prec = right_total / pred_total
        recall = right_total / answ_total
        f1_score = 2*prec*recall/(prec+recall)
        print(f'prec: {prec}, recall: {recall}, f1: {f1_score}')

        dprec = dright_total / dpred_total
        drecall = dright_total / answ_total
        df1_score = 2*dprec*recall/(dprec+recall)
        print(f'dprec: {dprec}, drecall: {drecall}, df1: {df1_score}')





run_train(train_dataset)
#test_model('model_electra_j_epoch73.pt')

print('Ok.')
