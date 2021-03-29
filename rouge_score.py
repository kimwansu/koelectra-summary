# -*- coding: utf-8 -*-
'''
최초 작성일: 2021-3-8
최종 수정일: 2021-3-9

ROUGE 스코어 측정 테스트
'''

#test_answer = 'the cat was found under the bed'.split()
test_answer = "실업·물가폭등에 10건 발생 청년 노점상의 분신 자살이 기폭제가 돼 튀니지의 민중봉기가 ‘재스민 혁명’으로 성공한 이후 폭압정치와 빈곤에 허덕이는 북아프리카 국가들에서 18일 현재 분신 기도가 10건으로 느는 등 모방 분신이 크게 확산되고 있다. 호스니 무바라크 대통령이 29년째 집권하고 있는 이집트에선 18일 두 명의 젊은이가 분신을 기도했다고 <아에프페>통신이 이집트 보안관리를 인용해 보도했다.".split()
#test_answer = "'잔인한 4월'을 보낸 한국인 메이저리거 3인방이 5월 들어 움츠렸던 기지개를 켜고 있다. 올 시즌 MLB 무대에 입성한 강정호(28·피츠버그 파이리츠)는 4일(한국 시각) 미국 미주리주 부시스타디움에서 열린 세인트루이스 카디널스와의 원정 경기에서 메이저리그 데뷔 홈런을 신고했다. ◇드디어 터진 대포".split()
#test_answer = "2008년 대전지검장으로 재직하면서 부하 직원 250여 명에게 보내기 시작한 월요편지는 9년이 흘러 독자가 5000여 명으로 늘어났다. 화두도 조직 경영에서 인생 성찰로 바뀌었다. 독자는 '나이 저 정도 먹고 고검장까지 했다는 사람도 저런 고민을 하고 저런 약점이 있구나' 생각하며 인생을 돌아볼 계기가 되겠죠.".split()
#test_answer = 'police kill the gunman'.split()
#test_answer = 'the gunman kill police'.split()
#gold_answer = 'the cat was under the bed'.split()
gold_answer = "청년 노점상의 분신 자살이 기폭제가 된 튀니지의 민중봉기가 성공한 이후 북아프리카 국가들에서 모방 분신이 크게 확산되고 있다. 호스니 무바라크 대통령이 29년째 집권하고 있는 이집트와 격렬한 반정부 시위가 일어났던 알제리, 군부가 권력을 잡은 모리타니 등에서 분신이 이어지고 있다. 알제리 대학의 정치학 교수인 모하메드 라가브는 알자지라에 튀니지는 모든 아랍 사람들이 따를 모델이라며 독재자와 독재 정권의 시대는 끝나고 있다고 말했다.".split()
#gold_answer = "올 시즌 MLB 무대에 입성한 강정호는 4일 미국 미주리주 부시스타디움에서 열린 세인트루이스 카디널스와의 원정 경기에서 메이저리그 데뷔 홈런을 쳤다. 이날 7번 타자 겸 3루수로 선발 출전한 강정호는 9회초 카디널스 트레버 로젠탈의 초구 커브를 두들겨 좌중간 담장을 넘겨 홈런을 쳤다. 같은 날 추신수는 오클랜드 애슬레틱스와의 홈경기에서 4회 상대 선발 소니 그레이를 두들겨 2루타를 터뜨렸고 3타수 1안타 1볼넷을 기록했다.".split()
#gold_answer = "대전지검장으로 재직하면서 부하 직원들에게 월요편지를 보낸 것으로 유명한 조근호 변호사가 최근 인생의 목표에 대한 답을 찾아나가는 과정이라는 '당신과 행복을 이야기하고 싶습니다'를 펴냈다. 이번 책은 2011년 검찰에서 나온 뒤 쓴 월요편지 280여 편을 주제별로 묶어 정리한 것으로 인생 성찰을 화두로 삼고 있다. 그의 글의 바탕은 독서, 많이 읽어야 사소한 일상에서 삶의 의미를 찾을 수 있다는 게 그의 지론이다.".split()
#gold_answer = 'police killed the gunman'.split()


def main():
    prec, recl, f1 = rouge_n(gold_answer, test_answer)

    print(prec, recl, f1)

    test_bigram = generate_bigram(test_answer)
    gold_bigram = generate_bigram(gold_answer)
    #print(test_bigram)
    #print(gold_bigram)

    #print(test_bigram[0] == gold_bigram[0])

    prec, recl, f1 = rouge_n(gold_bigram, test_bigram)

    print(prec, recl, f1)

    #lcs = calc_lcs(gold_answer, test_answer)
    #print(lcs)
    #print(lcs_word(gold_answer, test_answer))
    print('----')
    #matched = lcs_subseq(gold_answer, test_answer)
    matched = lcs_subseq(test_answer, gold_answer)
    lcs_recl = float(len(matched)) / len(gold_answer)
    lcs_prec = float(len(matched)) / len(test_answer)
    print(lcs_prec, lcs_recl, calc_f1_score(lcs_recl, lcs_prec))

    matches = lcs_subseq(test_answer, gold_answer)
    print(matches)



    print('Ok.')


def rouge_n(pred, answer):
    tp = 0  # 있다고 판단했는데 진짜로 있다.
    fp = 0  # 있다고 판단했는데 없다
    fn = 0  # 없다고 판단했는데 있다.
    for w in pred:
        if w in answer:
            tp += 1
        else:
            fp += 1
    
    for w in answer:
        if w not in pred:
            fn += 1

    prec = tp / (tp+fp)
    recl = tp / (tp+fn)
    f1 = calc_f1_score(prec, recl)
    
    return [prec, recl, f1]

def rouge_l(pred, answer):
    matched = lcs_subseq(pred, answer)
    prec = float(len(matched) / len(pred))
    recl = float(len(matched) / len(answer))
    f1 = calc_f1_score(prec, recl)

    return [prec, recl, f1]


def calc_f1_score(a, b):
    if a == 0 and b == 0:
        return 0
        
    return 2 * (a * b) / (a + b)

def generate_bigram(x):
    bg = []
    for i in range(len(x) - 1):
        bg.append([x[i], x[i+1]])

    return bg

def calc_lcs(a, b):
    lcs = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i-1] == b[j-1]:
                lcs[i][j] = lcs[i-1][j-1] + 1
            else:
                lcs[i][j] = max(lcs[i][j-1], lcs[i-1][j])

    return lcs

# LCS 계산 + 일치 범위 찾기 --> 연속만 가능
def lcs_word(a, b):
    mx = 0
    idx = 0
    letters = [[0] * (len(b) + 1)] * (len(a) + 1)
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                letters[i+1][j+1] = letters[i][j] + 1
            
            if mx < letters[i+1][j+1]:
                mx = letters[i+1][j+1]
                idx = i + 1
    
    return a[idx-mx:idx]

# https://jay-ji.tistory.com/33
# LCS sub-sequence
def lcs_subseq(a, b):
    lcs = calc_lcs(a, b)
    i = len(a)
    j = len(b)
    matched = []
    while i > 0 and j > 0:
        if lcs[i-1][j] < lcs[i][j] and lcs[i][j-1] < lcs[i][j]:
            matched += [a[i-1]]
            i -= 1
            j -= 1
        elif lcs[i-1][j] < lcs[i][j]:
            j -= 1
        elif lcs[i][j-1] < lcs[i][j]:
            i -= 1
        else: # same
            i -= 1
            j -= 1

    matched.reverse()
    return matched


if __name__ == '__main__':
    main()

