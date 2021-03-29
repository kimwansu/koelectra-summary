import json

from rouge_score import *
from config import *


# 문서 데이터, 정답 라벨 파일 로드
with open(corpus_dir + 'doc_summary4.json', 'r', encoding='utf-8') as f:
    j_doc = json.load(f)

# 문서 id 목록 로드
with open(corpus_dir + 'splitted_doclist.json', 'r', encoding='utf-8') as f:
    j_list = json.load(f)

train_sets = j_doc

# 테스트용 문서 id
test_doc_ids = j_list['test']


def main():
    print('hello')
    for doc_id in test_doc_ids[:10]:
        print(doc_id)
        summ_sent = get_extract_summary(doc_id)
        title_sent = get_titles(doc_id)
        line3_sent = get_3lines(doc_id)

        # 정답으로 사용할 문장
        gene_sent = get_generate_summary(doc_id)

        metric(summ_sent, gene_sent)
        print('--------')


# 제목+부제 추출. 부제는 없어서 임의로 첫번째 문장으로
def get_titles(doc_id):
    out_sents = [
        j_doc[doc_id]['title'],
        j_doc[doc_id]['sents'][0]
    ]
    return out_sents


# 처음 3문장 추출
def get_3lines(doc_id):
    out_sents = j_doc[doc_id]['sents'][:3]
    return out_sents


# 정답 추출 요약 문장 추출
def get_extract_summary(doc_id):
    out_sents = j_doc[doc_id]['topics']
    return out_sents


# 정답 생성 요약 문장 추출
def get_generate_summary(doc_id):
    out_sents = j_doc[doc_id]['summaries']
    return out_sents


def sentid2txt(doc_id, sent_idx):
    out_sents = []
    sent_idx = sorted(sent_idx)
    for i in sent_idx:
        if i < 0 or i > len(j_doc[doc_id]['sents']):
            return None
        
        out_sents.append(j_doc[doc_id]['sents'][i])

    return out_sents


def metric(test_sent, answer_sent):
    result = dict()
    test_sent = ' '.join(test_sent).split(' ')
    answer_sent = ' '.join(answer_sent).split(' ')

    # rouge-1
    prec, recl, f1 = rouge_n(test_sent, answer_sent)
    print(f'rouge-1: {prec} / {recl}   {f1}')

    # rouge-2
    test_bigram = generate_bigram(test_sent)
    answer_bigram = generate_bigram(answer_sent)

    prec, recl, f1 = rouge_n(test_bigram, answer_bigram)
    print(f'rouge-2: {prec} / {recl}   {f1}')

    # rouge-L
    prec, recl, f1 = rouge_l(test_sent, answer_sent)
    print(f'roue-L: {prec} / {recl}  {f1}')
    
    return result


def asdf():
    for doc_id in test_doc_ids:
        title = get_titles(doc_id)
        line3 = get_3lines(doc_id)

        answer_extract_summary = get_extract_summary(doc_id)
        answer_generate_summary = get_generate_summary(doc_id)

        # 정답 추출 요약 문장과의 rouge 스코어 계산
        


# 추출 요약 결과 불러오기
## 단순 결과
## 


if __name__ == '__main__':
    main()
