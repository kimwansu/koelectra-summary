
'''
최초 작성일: 2021-3-8
최종 수정일: 2021-3-8

모두의 말뭉치 문서 요약 말뭉치와 신문 기사 말뭉치 합치기
'''

import json
import os

from config import corpus_dir

summary_corpus_path = corpus_dir + '국립국어원 문서 요약 말뭉치(버전 1.0)/NIKL_SC.json'
news_corpus_dir = corpus_dir + '국립국어원 신문 말뭉치(버전 1.0)/'


def main():
    # 1. 요약 말뭉치 열기
    sum_titles, topic_sents, summary_sents = load_summary_corpus()
    #print(len(titles), len(topic_sents), len(summary_sents))

    # 2. 뉴스 말뭉치 열기
    doc_titles, doc_sents = load_news_corpus()
    
    # 3. 문서-요약 쌍 맞추기
    j = dict()
    for key in sorted(sum_titles.keys()):
        sum_title = sum_titles[key]
        if key not in doc_titles:
            print(f'{key} NOT FOUND')
            continue

        doc_title = doc_titles[key]
        assert sum_title == doc_title

        j[key] = {
            'title': sum_titles[key],
            'sents': doc_sents[key],
            'topics': topic_sents[key],
            'summaries': summary_sents[key]
        }

    # 4. 파일로 기록, 분할
    with open('C:/Users/Kim Wansu/Documents/corpus/doc_summary1.json', 'w', encoding='utf-8') as f:
        json.dump(j, f, ensure_ascii=False, indent=2)
    
    print('Ok.')
    


def load_summary_corpus():
    titles = dict()
    topic_sents = dict()
    summary_sents = dict()

    with open(summary_corpus_path, 'r', encoding='utf-8') as f:
        buf = f.read()
        j = json.loads(buf)
        for data in j['data']:
            doc_id = data['document_id']
            title = data['head']
            topics = data['topic_sentences']
            summaries = data['summary_sentences']

            titles[doc_id] = title
            topic_sents[doc_id] = topics
            summary_sents[doc_id] = summaries

    return titles, topic_sents, summary_sents


def load_news_corpus():
    titles = dict()
    doc_sents = dict()

    for fname in os.listdir(news_corpus_dir):
        _, ext = os.path.splitext(fname)
        if ext != '.json':
            continue

        full_path = os.path.join(news_corpus_dir, fname)
        with open(full_path, 'r', encoding='utf-8') as f:
            buf = f.read()
            j = json.loads(buf)
            for doc in j['document']:
                doc_id = doc['id']
                title = doc['paragraph'][0]['form']
                sents = [p['form'] for p in doc['paragraph'][1:]]

                titles[doc_id] = title
                doc_sents[doc_id] = sents
    
    return titles, doc_sents

            

if __name__ == '__main__':
    main()
