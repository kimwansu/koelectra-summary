import os
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

from config import corpus_dir


MAX_SENT_COUNT = 78
MAX_TOKEN_COUNT = 202

EMBED_SIZE = 768

OUTPUT_COUNT = 2  # 출력 레이블 종류(0, 1)

cuda = torch.device('cuda:1')
cpu = torch.device('cpu')
device = cuda if torch.cuda.is_available() else cpu


class RNN(nn.Module):
    def __init__(self, n_layers, hidden_dim, embed_dim, n_classes):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )

    def forward(self, x):
        h_0 = self._init_state(batch_size=x.size(0))
        x, _ = self.gru(x, h_0)


def main():
    start = time.time()

    cuda = torch.device('cuda:0')
    cpu = torch.device('cpu')
    device = cuda if torch.cuda.is_available() else cpu

    model = torch.nn.LSTM(
        input_size=EMBED_SIZE,
        hidden_size=1,
        num_layers=2,
        batch_first=True,
        bidirectional=True).to(device)

    model_out = nn.Linear(2, 1).to(device)

    lr = 0.01
    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # 정답 라벨 데이터 로드
    with open(corpus_dir + 'doc_summary4.json', 'r', encoding='utf-8') as f:
        j_doc = json.load(f)

    # 문서 id 목록 로드
    with open(corpus_dir + 'splitted_doclist.json', 'r', encoding='utf-8') as f:
        j_list = json.load(f)

    # 테스트용 문서 id
    test_doc_ids = j_list['test']

    # 검증용 문서 id
    val_doc_ids = j_list['train'][0]

    # 학습용 문서 id
    train_doc_ids = []
    for ids in j_list['train'][1:]:
        train_doc_ids += ids

    # 입력 문서의 임베딩 로드
    embed_path = corpus_dir + 'summary_embed/'

    n_epochs = 1000
    max_patience = 10  # 이 횟수만큼 성능 향상 없으면 학습 중단
    patience = 0

    best_loss_val = float('inf')
    best_epoch = 0
    best_state = None

    stop_by_val_data = False

    for epoch in range(n_epochs):
        avg_loss_train = 0.0
        avg_loss_val = 0.0
        data_count = len(train_doc_ids)
        random.shuffle(train_doc_ids)
        for doc_id in tqdm(train_doc_ids):
            fname = doc_id + '.embed.npy'
            full_fname = os.path.join(embed_path, fname)
            if not os.path.exists(full_fname):
                data_count -= 1
                continue

            optimizer.zero_grad()

            # 한 문서의 모든 문장의 KoELECTRA 임베딩
            embeds = np.load(full_fname)
            embed_tensors = torch.Tensor(embeds).to(device)
            output, status = model(embed_tensors)
            label_H = torch.sigmoid(model_out(output))
            label_H = label_H.squeeze(2)  # 마지막 괄호 제거
            label_H = label_H[:, -1].to(cpu)  # 마지막 상태만 추출

            label_Y = torch.Tensor(j_doc[doc_id]['labels'])

            loss_train = criterion(label_H, label_Y)
            loss_train.backward()
            optimizer.step()

            avg_loss_train += loss_train.item() / data_count

        # 여기쯤에서 검증 말뭉치로 성능 측정 시도
        # 일정 횟수 이상 성능 개선이 되지 않으면 마지막 최고성능 가중치로 복원 후 종료
        # 중간중간 상태 기록 필요
        with torch.no_grad():
            v_data_count = len(val_doc_ids)
            for v_doc_id in tqdm(val_doc_ids):
                fname = v_doc_id + '.embed.npy'
                v_full_fname = os.path.join(embed_path, fname)
                if not os.path.exists(v_full_fname):
                    v_data_count -= 1
                    continue

                v_embeds = np.load(v_full_fname)
                v_embed_tensors = torch.Tensor(v_embeds).to(device)
                v_output, v_status = model(v_embed_tensors)
                v_label_H = torch.sigmoid(model_out(v_output))
                v_label_H = v_label_H.squeeze(2)
                v_label_H = v_label_H[:, -1].to(cpu)

                v_label_Y = torch.Tensor(j_doc[v_doc_id]['labels'])

                loss_val = criterion(v_label_H, v_label_Y)
                avg_loss_val += loss_val.item() / v_data_count

        if avg_loss_val < best_loss_val:
            best_loss_val = avg_loss_val
            best_epoch = epoch
            best_state = model.state_dict()
            patience = max_patience
        else:
            patience -= 1
            stop_by_val_data = True
            if patience == 0:
                break

        scheduler.step(avg_loss_val)

        msg = 'NEW BEST' if patience == max_patience else f'patience={patience}'
        print(f'[Epoch {epoch+1:>4}] train_cost = {avg_loss_train:>.9}, val_cost = {avg_loss_val:>.9} {msg} lr = {scheduler._last_lr}')

    if stop_by_val_data:
        print(f'Epoch {best_epoch:>4} stop by validation data.')
        model.load_state_dict(best_state)

    model_state_fname = 'extract_summary.model'
    torch.save(model.state_dict(), model_state_fname)

    print('Ok.')

    end = time.time()
    elapsed = end - start
    print(elapsed)


if __name__ == '__main__':
    main()
