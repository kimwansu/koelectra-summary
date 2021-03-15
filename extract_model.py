
'''
ELECTRA v2 BASE 모델로 문장을 인코드해서 파일로 출력하기
'''

import json

import torch
from transformers import ElectraModel, ElectraTokenizer

import time
from tqdm import tqdm
import numpy as np

from config import corpus_dir


def main():
    start = time.time()

    cuda = torch.device('cuda:0')
    cpu = torch.device('cpu')
    device = cuda if torch.cuda.is_available() else cpu

    electra = 'monologg/koelectra-base-v2-discriminator'

    # KoELECTRA 모델 로드(v2)
    model = ElectraModel.from_pretrained(electra).cuda()
    tokenizer = ElectraTokenizer.from_pretrained(electra)

    # 문서 데이터 파일 로드
    with open(corpus_dir + 'doc_summary4.json', 'r', encoding='utf-8') as f:
        j_doc = json.load(f)

    train_sets = j_doc
    
    for doc_id in tqdm(train_sets):
        sents = j_doc[doc_id]['sents']
        a = tokenizer.encode(sents, padding=True, return_tensors='pt').cuda()
        o = model(a)
        o_np = o[0].cpu().detach().numpy()
        np.save(f'summary_embed/{doc_id}.embed', o_np)

    # CPU로 실행 시간 테스트
    end = time.time()
    elapsed = end - start
    print(elapsed)

    print('Ok.')


if __name__ == '__main__':
    main()
