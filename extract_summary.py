import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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

    cuda = torch.device('cuda:1')
    cpu = torch.device('cpu')
    device = cuda if torch.cuda.is_available() else cpu

    model = torch.nn.LSTM(
        input_size=EMBED_SIZE,
        hidden_size=1,
        num_layers=2,
        batch_first=True,
        bidirectional=True).to(device)

    model_out = nn.Linear(2, 1)

    lr = 0.1
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # 정답 라벨 데이터 로드
    with open(corpus_dir + 'doc_summary4.json', 'r', encoding='utf-8') as f:
        j_doc = json.load(f)

    # 입력 문서의 임베딩 로드
    embed_path = corpus_dir + 'summary_embed/'

    n_epochs = 1000

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for i, fname in enumerate(os.listdir(embed_path)):
            if '.embed.npy' not in fname:
                continue

            # TODO: cross-validation 넣기
            

            doc_id = fname.replace('.embed.npy', '')

            full_fname = os.path.join(embed_path, fname)

            optimizer.zero_grad()

            # 한 문서의 모든 문장의 KoELECTRA 임베딩
            embeds = np.load(full_fname)
            embed_tensors = torch.Tensor(embeds)
            output, status = model(embed_tensors)
            label_H = torch.sigmoid(model_out(output))
            label_H = label_H.squeeze(2)  # 마지막 괄호 제거
            label_H = label_H[:,-1]  # 마지막 상태만 추출

            label_Y = torch.Tensor(j_doc[doc_id]['labels'])

            loss = criterion(label_H, label_Y)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'== Epoch {epoch+1} ==')
            print(f'loss: {epoch_loss}') 


    print('Ok.')

    end = time.time()
    elapsed = end - start
    print(elapsed)


if __name__ == '__main__':
    main()
