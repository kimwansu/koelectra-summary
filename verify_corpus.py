
'''
최초 작성일: 2021-3-9
최종 수정일: 2021-3-9

모두의 말뭉치 문서 요약 말뭉치 유효성 검사 및 수정
'''

import json

from rouge_score import calc_f1_score, calc_lcs, lcs_subseq, generate_bigram

from config import corpus_dir

corpus_path = corpus_dir + 'doc_summary3.json'

def main():
    j = load_corpus()
    #return
    for doc_id, data in j.items():
        topic_sents = data['topics']
        all_sents = data['sents']
        for ts in topic_sents:
            if ts not in all_sents:
                print(doc_id, ts)
        
        # 1. 원본 문장과 정답(추출) 문장과 rouge 스코어 계산
        # --> 추출 문장은 원본에서 그대로 골라낸 것

    print('Ok.')


def load_corpus():
    with open(corpus_path, 'r', encoding='utf-8') as f:
        j = json.load(f)
        return j
    """
    for doc_id, data in j.items():
        sents = data['sents']
        new_sents = []
        for sent in sents:
            sent = sent.replace('. .', '.')
            if '. ' in sent:
                split_sents = sent.split('. ')
                for s in split_sents[:-1]:
                    new_sents.append(s + '.')
                new_sents.append(split_sents[-1])
            else:
                new_sents.append(sent)
        
        new_sents2 = []
        for sent in new_sents:
            if '.” ' in sent:
                split_sents = sent.split('.” ')
                for s in split_sents[:-1]:
                    new_sents2.append(s + '.”')
                new_sents2.append(split_sents[-1])
            else:
                new_sents2.append(sent)
            
        new_sents3 = []
        for sent in new_sents2:
            if '? ' in sent:
                split_sents = sent.split('? ')
                for s in split_sents[:-1]:
                    new_sents3.append(s + '?')
                new_sents3.append(split_sents[-1])
            else:
                new_sents3.append(sent)
        
        j[doc_id]['sents'] = new_sents3

        new_topics = []
        for t in data['topics']:
            if '. ' in t:
                split_sents = t.split('. ')
                for s in split_sents[:-1]:
                    new_topics.append(s + '.')
                new_topics.append(split_sents[-1])
            else:
                new_topics.append(t)
        data['topics'] = new_topics

        new_summ = []
        for s in data['summaries']:
            if '. .' in s:
                s = s.replace('. .', '.')
            
            if '. ' in s:
                split_sents = s.split('. ')
                for s in split_sents[:-1]:
                    new_summ.append(s + '.')
                new_summ.append(split_sents[-1])
            else:
                new_summ.append(s)
        data['summaries'] = new_summ
        
    
    corpus2_path = 'C:/Users/Kim Wansu/Documents/corpus/doc_summary3.json'
    with open(corpus2_path, 'w', encoding='utf-8') as f:
        json.dump(j, f, ensure_ascii=False, indent=2)
    #"""

    return ''





if __name__ == '__main__':
    main()
