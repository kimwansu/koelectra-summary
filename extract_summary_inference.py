import os
import json

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from config import corpus_dir

EMBED_SIZE = 768

def main():
    cuda = torch.device('cuda:0')
    cpu = torch.device('cpu')
    device = cuda if torch.cuda.is_available() else cpu

    model = torch.nn.LSTM(
        input_size=EMBED_SIZE,
        hidden_size=32,
        num_layers=1,
        batch_first=True,
        bidirectional=True).to(device)

    linear_out = nn.Linear(64, 1).to(device)
    torch.nn.init.kaiming_uniform_(linear_out.weight)

    lr = 2e-3
    criterion = nn.BCELoss().to(device)

    # 정답 라벨 데이터 로드
    with open(corpus_dir + 'doc_summary4.json', 'r', encoding='utf-8') as f:
        j_doc = json.load(f)

    # 문서 id 목록 로드
    with open(corpus_dir + 'splitted_doclist.json', 'r', encoding='utf-8') as f:
        j_list = json.load(f)

    # 테스트용 문서 id
    test_doc_ids = j_list['test']

    avg_loss_test = 0.0

    # 입력 문서의 임베딩 로드
    embed_path = corpus_dir + 'summary_embed/'

    model.load_state_dict(torch.load('extract_summary.model'))
    model.eval()

    with torch.no_grad():
        t_data_count = len(test_doc_ids)
        for t_doc_id in tqdm(test_doc_ids):
            fname = t_doc_id + '.embed.npy'
            t_full_fname = os.path.join(embed_path, fname)
            if not os.path.exists(t_full_fname):
                t_data_count -= 1
                continue
            
            t_embeds = np.load(t_full_fname)
            t_embed_tensors = torch.Tensor(t_embeds).to(device)
            t_output, t_status = model(t_embed_tensors)

            t_label_H = torch.sigmoid(linear_out(t_output))
            t_label_H = t_label_H.squeeze(2)
            t_label_H = t_label_H[:, -1].to(cpu)

            t_label_Y = torch.Tensor(j_doc[t_doc_id]['labels'])

            loss_test = criterion(t_label_H, t_label_Y)
            avg_loss_test += loss_test.item() / t_data_count
        
        print(avg_loss_test)


    
    print('Ok.')


if __name__ == '__main__':
    main()
