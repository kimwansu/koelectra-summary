
'''
최초 작성일: 2021-3-9
최종 수정일: 2021-3-9

모두의 말뭉치 문서 요약 말뭉치 유효성 검증(수량 위주로)
'''

import json

from config import corpus_dir

corpus_path = corpus_dir + '/doc_summary3.json'


def main():
    j = load_corpus()
    max_sents = 0
    max_words = 0
    max_words2 = 0
    for doc_id, data in j.items():
        sents = data['sents']
        topics = data['topics']
        summ = data['summaries']

        if len(sents) > max_sents:
            max_sents = len(sents)
        
        if len(summ) != 3:
            print(f'{doc_id}: {len(summ)} topics.')

        for sent in sents:
            words = sent.split(' ')
            if len(words) > max_words:
                max_words = len(words)

        for sent in summ:
            words = sent.split(' ')
            if len(words) > max_words2:
                max_words2 = len(words)
    
    print(f'max_sents: {max_sents}')
    print(f'max_words: {max_words}')
    print(f'max_summ_words: {max_words2}')
    print('Ok.')



def load_corpus():
    with open(corpus_path, 'r', encoding='utf-8') as f:
        j = json.load(f)
        return j


if __name__ == '__main__':
    main()
    