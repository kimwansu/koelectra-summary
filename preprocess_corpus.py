
'''
최초 작성일: 2021-3-9
최종 수정일: 2021-3-9

모두의 말뭉치 문서 요약 말뭉치 전처리
'''

import json
import numpy as np

from rouge_score import rouge_n, calc_f1_score, calc_lcs, lcs_subseq, generate_bigram

from config import corpus_dir

'''
전처리용으로 정답 문장과 원문과 rouge 스코어로 추출할 문장 결정?

출력: 원본 json + 원본 문장에 대한 요약문 레이블(0/1)

rogue-1 스코어 기반으로 해서 정답과 일치하는지 확인
'''


in_corpus_path = corpus_dir + 'doc_summary3.json'
out_corpus_path = corpus_dir + 'doc_summary4.json'

def main():
    j = load_corpus()
    for doc_id, data in j.items():
        sents = data['sents']
        topics = data['topics']
        # 유효성 검사(추출 요약 문장이 온전하게 원본에 일치하는 것이 있는지 확인)
        for ts in topics:
            if ts not in sents:
                print(doc_id, ts)
        
        answer_table = make_answer_table_simple(sents, topics)

        rouge_answer_table = make_answer_table_rouge(sents, topics)

        # 추출 요약 정답과 실제로 테스트한 것과 일치하지 않는 것이 있는지 검사
        if answer_table != rouge_answer_table:
            print(doc_id)

        labels = make_extact_sent_idx_list(sents, topics)

        # 원본 문서 id와 레이블을 기록
        j[doc_id]['labels'] = labels
    
    with open(out_corpus_path, 'w', encoding='utf-8') as f:
        json.dump(j, f, ensure_ascii=False, indent=2)


    print('Ok.')



def load_corpus():
    with open(in_corpus_path, 'r', encoding='utf-8') as f:
        j = json.load(f)
        return j


def make_answer_table_simple(sents, topics):
    # 토픽과 전체 일치하는 문장이 있으면 정답으로 간주하고 테이블 작성
    answer_table = [0] * len(sents)
    for i, sent in enumerate(sents):
        if sent in topics:
            answer_table[i] = 1

    return answer_table


def make_answer_table_rouge(sents, topics):
    # 문장 단위 rogue-1로 테이블 작성
    answer_table = [0] * len(sents)
    answer_sent = ' '.join(topics)

    current_score = 0.0
    current_sentence = []

    while True:
        # 1. 정답 문장과 원본의 모든 문장과 rouge 스코어 계산
        score_table = [0.0] * len(sents)
        for i, sent in enumerate(sents):
            # 이미 결정된 문장은 비교 대상에서 제외한다.
            if answer_table[i] != 0:
                continue

            test_sent = sent
            if current_sentence:
                test_sent = ' '.join(current_sentence) + ' ' + sent
            _, _, recl = rouge_n(answer_sent.split(' '), test_sent.split(' '))
            _, _, prec = rouge_n(test_sent.split(' '), answer_sent.split(' '))

            f1 = calc_f1_score(prec, recl)
            score_table[i] = f1
        
        # 2. 스코어가 가장 높은 문장을 선택, 1 레이블링
        max_score = max(score_table)
        max_index = np.argmax(score_table)

        if max_score > current_score:
            answer_table[max_index] = 1
            current_score = max_score
            current_sentence.append(sents[max_index])
        else:
            break

    return answer_table


def make_extact_sent_idx_list(sents, topics):
    # 추출 정답 문장의 인덱스 찾기
    # 문서 내에 문장이 중복으로 있지 않다고 가정
    labels = [0] * len(sents)
    for topic in topics:
        assert topic in sents
        idx = sents.index(topic)
        labels[idx] = 1

    return labels


if __name__ == '__main__':
    main()
