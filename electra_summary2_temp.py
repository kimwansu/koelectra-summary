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

from sklearn.metrics.pairwise import cosine_similarity

import rouge_score as rs

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

        dataset.append([doc_id, input_ids_batch, attention_batch, segment_batch, y_answer_batch])

    return dataset


#train_dataset = load_dataset(train_doc_ids)
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

lr = 0.02
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
        for doc_id, iid_batch, am_batch, segment_emb, y_batch in tqdm(dataset):
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

        for doc_id, iid_batch, am_batch, segment_emb, y_batch in tqdm(test_dataset):
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


def test_model2(state_fname):
    lstm.load_state_dict(torch.load(state_fname))

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    doc_ids = []
    text_backup = dict()

    with torch.no_grad():
        lstm.eval()

        for doc_id, iid_batch, am_batch, segment_emb, y_batch in tqdm(test_dataset):
            y_batch = torch.LongTensor(y_batch).to(device)
            output =  model(iid_batch.to(device), attention_mask=am_batch.to(device), token_type_ids=segment_emb.to(device))
            y_pred = output[0]

            sent_embeds = output.last_hidden_state[::,0].to(cpu).numpy()

            y_cls_head = y_pred[:, 0].unsqueeze(1)
            y_pred = lstm(y_cls_head)
            #y_pred = F.sigmoid(y_pred).squeeze()
            y_pred = F.softmax(y_pred)
            y_pred = y_pred.squeeze()

            tmp_y0 = y_pred.T[0]
            tmp_y1 = y_pred.T[1]
            sim_arr = tmp_y1 - tmp_y0
            sim_arr = sim_arr.to(cpu).numpy()

            mmr = get_mmr(sent_embeds, sim_arr)
            mmr_predicted = [1 if i in mmr else 0 for i in range(len(sim_arr))]

            top_id = sorted(range(len(sim_arr)), key=lambda i:sim_arr[i])[-3:][::-1]
            top3_pred = sorted(np.array([1 if i in top_id else 0 for i in range(len(y_pred))]))

            y_batch = torch.max(y_batch, 1)[1].to(cpu)
            predicted = torch.max(y_pred, 1)[1].to(cpu)

            #text = ' '.join(extract_text(doc_id, predicted.to(cpu)))
            #text = ' '.join(extract_title(doc_id)) # 제목+첫줄
            #text = ' '.join(extract_text(doc_id, [1,1,1])) # 처음3줄
            #text = ' '.join(extract_text(doc_id, top3_pred)) # 유사도 상위3위까지
            text = ' '.join(extract_text(doc_id, mmr_predicted)) # MMR 상위3위까지
            text_backup[doc_id] = '\n'.join(extract_text(doc_id, mmr_predicted))

            #ans = ' '.join(extract_text(doc_id, y_batch.to(cpu))) # 정답 추출요약
            ans = ' '.join(extract_generative_summary_sents(doc_id)) # 정답 생성요약

            #print(doc_id)
            prec, recl, f1 = rs.rouge_n(text.split(), ans.split())
            #print(prec, recl, f1)
            rouge1_scores.append(f1)

            text_bi = rs.generate_bigram(text.split())
            ans_bi = rs.generate_bigram(ans.split())
            prec, recl, f1 = rs.rouge_n(text_bi, ans_bi)
            #print(prec, recl, f1)
            rouge2_scores.append(f1)

            prec, recl, f1 = rs.rouge_l(text.split(), ans.split())
            #print(prec, recl, f1)
            rougeL_scores.append(f1)

            doc_ids.append(doc_id)

    rouge1_score = sum(rouge1_scores) / len(rouge1_scores)
    rouge2_score = sum(rouge2_scores) / len(rouge2_scores)
    rougeL_score = sum(rougeL_scores) / len(rougeL_scores)

    print(rouge1_score, rouge2_score, rougeL_score)

    assert(len(doc_ids) == len(rougeL_scores))
    rL_score = np.array(rougeL_scores)
    rougeL_ranks = rL_score.argsort()

    rank_1q_idx = rougeL_ranks[len(doc_ids) // 4]
    rank_2q_idx = rougeL_ranks[len(doc_ids) // 4 * 2]
    rank_3q_idx = rougeL_ranks[len(doc_ids) // 4 * 3]

    rank_1q_docid = doc_ids[rank_1q_idx]
    rank_2q_docid = doc_ids[rank_2q_idx]
    rank_3q_docid = doc_ids[rank_3q_idx]

    rank_1q_answer = '\n'.join(extract_generative_summary_sents(rank_1q_docid)) # 정답 생성요약
    rank_2q_answer = '\n'.join(extract_generative_summary_sents(rank_2q_docid)) # 정답 생성요약
    rank_3q_answer = '\n'.join(extract_generative_summary_sents(rank_3q_docid)) # 정답 생성요약

    rank_1q_result = text_backup[rank_1q_docid]
    rank_2q_result = text_backup[rank_2q_docid]
    rank_3q_result = text_backup[rank_3q_docid]

    print(rank_3q_docid)
    print('정확도 상위 25% 문장  F1: ', rougeL_scores[rank_3q_idx])
    print('결과:')
    print(rank_3q_result)
    print('------------')
    print('정답:')
    print(rank_3q_answer)

    print(rank_2q_docid)
    print('정확도 상위 50% 문장  F1: ', rougeL_scores[rank_2q_idx])
    print('결과:')
    print(rank_2q_result)
    print('------------')
    print('정답:')
    print(rank_2q_answer)

    print(rank_1q_docid)
    print('정확도 상위 75% 문장  F1: ', rougeL_scores[rank_1q_idx])
    print('결과:')
    print(rank_1q_result)
    print('------------')
    print('정답:')
    print(rank_1q_answer)
    


def get_cosine_similarity(x1, x2):
    return (x1 * x2).sum() / ((x1**2).sum()**.5 * (x2**2).sum()**.5)


def get_mmr(sent_embeds, sim_arr):
    # sim_arr : 각 문장의 문서 주제에 대한 유사도
    '''
    1. 현재 랭킹 1위 문장 선택, 결과에 추가
    2. 1위 제외한 나머지 문장을 대상으로 유사도 계산
    3. 유사도 점수가 높으면서 이미 추가한 문장과 유사도 낮은 문장 선택, 결과에 추가
    '''
    top_sent_idx = sim_arr.argmax() # 최고 점수 문장의 인덱스
    max_sim_score = sim_arr[top_sent_idx]
    # 점수 동률이 없다고 가정 --> OK
    import sys
    if max_sim_score in sim_arr[top_sent_idx+1:]:
        print('동률 점수 발생: ', sent_embeds, sim_arr)
        sys.exit(1)
    
    keywords_idx = [top_sent_idx]
    candidates_idx = [i for i in range(len(sim_arr)) if i != keywords_idx[0]]

    inter_sent_similarity = []
    for i in range(len(sim_arr)):
        inter_sent_similarity.append([])
        for j in range(len(sim_arr)):
            if i == j:
                inter_sent_similarity[-1].append(1.0)
            else:
                #sim = get_cosine_similarity(sent_embeds[i], sent_embeds[j])
                sim = cosine_similarity(sent_embeds[i].reshape(1,-1), sent_embeds[j].reshape(1,-1))
                inter_sent_similarity[-1].append(sim)

    top_n = 3
    alpha = 0.5 # 문서-문장간 유사도의 비중.
    for _ in range(top_n - 1):
        # cand_sim --> sim_arr에 이미 계산되어 있는 수 사용
        max_c_idx = -1
        max_c_val = -9999
        for c in candidates_idx:
            for k in keywords_idx:
                ck_sents_similarity = get_cosine_similarity(sent_embeds[c], sent_embeds[k])
                H = alpha * sim_arr[c] - (1-alpha) * ck_sents_similarity
                #print(f'{c} {k} {H}')
                if H > max_c_val:
                    max_c_val = H
                    max_c_idx = c
            
        mmr_idx = -1
        if max_c_idx != -1:
            mmr_idx = max_c_idx
        
        if mmr_idx >= 0:
            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)
        else:
            import sys
            print('최대 후보 못찾음')
            sys.exit(1)
    
    return sorted(keywords_idx)
    


def extract_text(doc_id, index):
    '''
    index: 문서의 문장 수와 같은 길이의 배열. 추출할 문장에는 1, 그외에는 0표시.
    임의로 처음 3문장만 추출할 때는 [1,1,1]을 넣으면 됨
    '''
    text = []
    for i, (sent, x) in enumerate(zip(j_doc[doc_id]['sents'], index)):
        if x != 0:
            text.append(sent)

    return text

def extract_title(doc_id):
    return [
        j_doc[doc_id]['title'],
        j_doc[doc_id]['sents'][0]
    ]

def extract_generative_summary_sents(doc_id):
    return j_doc[doc_id]['summaries']





#run_train(train_dataset)
#test_model('model_electra_j_epoch73.pt')
test_model2('model_electra_j_epoch265.pt')

print('Ok.')
