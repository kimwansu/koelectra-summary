---
modified: 2021-03-11T05:47:02.262Z
---

## 소스 파일
* verify_corpus.py : 모두의 말뭉치에서 한 줄에 두 문장 이상이 나오는 경우 분리해서 새로운 json 파일 작성. --> doc_summary3.json
* preprocess_corpus.py : 추출 요약 문장이 나오는 인덱스 번호 리스트 작성 --> doc_summary4.json
* split_shuffle_corpus.py : 문서 목록의 순서를 섞고 학습용, 테스트용 분리한다.

## 데이터 파일
* doc_summary3.json : 문장 분리 처리 완료된 말뭉치
* doc_summary4.json : 3 + 추출 요약 문장 인덱스 추가
* splitted_doclist.json : 4와 같이 사용. 학습, 테스트 데이터 로드 순서 저장.