
'''
최초 작성일: 2021-3-11
최종 수정일: 2021-3-11

실제로 섞는 것은 번거로우니 문서의 key 값만 가져와서 순서를 섞고 분리한다.
'''

import json
import numpy as np

from config import corpus_dir


def main():
    # 1. 말뭉치 불러와서 문서 id만 불러온다.
    with open(corpus_dir + 'doc_summary4.json', 'r', encoding='utf-8') as f:
        j = json.load(f)
    
    doc_ids = list(j.keys())

    # 2. 순서를 섞는다.
    np.random.shuffle(doc_ids)

    # 3. 학습용, 검증용, 테스트용 분리
    ## 6등분으로 나눠서 1조각 + 나머지는 테스트용으로 사용
    ## 나머지 5조각은 5-fold cross validation 식으로 학습 및 검증
    splitted = np.array_split(doc_ids, 6)
    out_j = dict()
    for i, ids in enumerate(splitted):
        if i == 0:
            out_j['test'] = list(ids)
        else:
            if 'train' not in out_j:
                out_j['train'] = []
            
            out_j['train'].append(list(ids))
    
    # 4. 분리된 목록을 파일에 기록한다.
    with open(corpus_dir + 'splitted_doclist.json', 'w', encoding='utf-8') as f:
        json.dump(out_j, f, ensure_ascii=False, indent=2)


    print('Ok.')




if __name__ == '__main__':
    main()
