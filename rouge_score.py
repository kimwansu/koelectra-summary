
'''
최초 작성일: 2021-3-8
최종 수정일: 2021-3-9

ROUGE 스코어 측정 테스트
'''

test_answer = 'the cat was found under the bed'.split()
#test_answer = 'police kill the gunman'.split()
#test_answer = 'the gunman kill police'.split()
gold_answer = 'the cat was under the bed'.split()
#gold_answer = 'police killed the gunman'.split()


def main():
    _, _, recl = rouge_n(gold_answer, test_answer)
    _, _, prec = rouge_n(test_answer, gold_answer)

    print(calc_f1_score(recl, prec))

    test_bigram = generate_bigram(test_answer)
    gold_bigram = generate_bigram(gold_answer)
    #print(test_bigram)
    #print(gold_bigram)

    #print(test_bigram[0] == gold_bigram[0])

    _, _, recl = rouge_n(gold_bigram, test_bigram)
    _, _, prec = rouge_n(test_bigram, gold_bigram)

    #print(recl, prec, f1_score(recl, prec))

    calc_lcs(gold_answer, test_answer)
    #print(lcs_word(gold_answer, test_answer))
    print('----')
    matched = lcs_subseq(gold_answer, test_answer)
    lcs_recl = float(len(matched)) / len(gold_answer)
    lcs_prec = float(len(matched)) / len(test_answer)
    print(calc_f1_score(lcs_recl, lcs_prec))



    print('Ok.')


def rouge_n(a, b):
    count = 0
    for w in a:
        if w in b:
            count += 1
    
    return [count, len(a), float(count) / len(a)]

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

