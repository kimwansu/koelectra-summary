import os
import sys
import json
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from tqdm import tqdm

from config import corpus_dir

cuda = torch.device('cuda:0')
cpu = torch.device('cpu')
device = cuda if torch.cuda.is_available() else cpu


model = ElectraForSequenceClassification.from_pretrained('monologg/koelectra-small-v2-discriminator').to(device)
tokenizer = AutoTokenizer.from_pretrained('monologg/koelectra-small-v2-discriminator')

#model.load_state_dict(torch.load('model.pt'))

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

# 입력 문서의 임베딩 로드
embed_path = corpus_dir + 'summary_embed2/'

train_dataset = []
for doc_id in tqdm(train_sets):
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
            max_length=256,
            pad_to_max_length=True,
            add_special_tokens=True
        )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        minibatch.append([input_ids, attention_mask, [1 if a == y else 0 for a in range(2)]])

    input_ids_batch = []
    attention_batch = []
    y_answer_batch = []
    for iid, am, ya in minibatch:
        input_ids_batch.append(iid)
        attention_batch.append(am)
        y_answer_batch.append(ya)
    
    input_ids_batch = torch.stack(input_ids_batch)
    attention_batch = torch.stack(attention_batch)

    train_dataset.append([input_ids_batch, attention_batch, y_answer_batch])


epochs = 3
batch_size = 16

optimizer = AdamW(model.parameters(), lr=1e-5)

loss_list = []
acc_list = []

criterion = nn.CrossEntropyLoss()

for i in range(1):
    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0

    model.train()
    print(len(train_dataset))
    print(len(train_dataset[0]))
    for iid_batch, am_batch, y_batch in tqdm(train_dataset):
        optimizer.zero_grad()
        
        y_batch = torch.LongTensor(y_batch).to(device)
        y_pred = model(iid_batch.to(device), attention_mask=am_batch.to(device))[0]
        y_pred = F.softmax(y_pred)
        y_batch = torch.max(y_batch, 1)[1]

        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predicted = torch.max(y_pred, 1)[1]
        correct += (predicted == y_batch).sum()
        total += len(y_batch)

        del y_pred
        del y_batch
        torch.cuda.empty_cache()

        batches += 1
        if batches % 100 == 0:
            print('batch loss: ', total_loss, 'acc: ', correct.float() / total)
    
    loss_list.append(total_loss)
    acc_list.append(correct.float() / total)
    print(f'Epoch {i} : train_loss: {total_loss}, acc: {correct.float() / total}')


# 모델 테스트

model.eval()

test_correct = 0
test_total = 0

for iid_batch, am_batch, y_batch in tqdm(test_dataset):
    y_batch = y_batch.to(device)
    y_pred = model(iid_batch.to(device), attention_mask=am_batch.to(device))[0]
    
    y_batch = torch.max(y_batch, 1)[1]
    predicted = torch.max(y_pred, 1)[1]
    test_correct += (predicted - y_batch)
    test_total += len(y_batch)

print(f'Accuracy: {test_correct.float() / test_total}')

torch.save(model.state_dict(), 'model_electra.pt')



















