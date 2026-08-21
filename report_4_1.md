# [4장 1강 심화] - 비선형성 확인 실습 (1) : 선형층 합성의 수치 동등성, 합성 가능성 계산.
 - 연속된 두 Linear의 전체 계산을 하나의 Linear로 수학적으로 축약할 수 있는가?

## 실습 배경

루멘 팀은 상담 안전 필터의 분류 head를 `Linear → Linear → Linear`로 깊게 만들고 "층이 세 개니 복잡한 표현을 학습한다"고 보고했습니다. 그러나 중간에 활성화가 없으면 여러 선형 변환은 하나의 선형 변환으로 합쳐질 수 있습니다.

운영 데이터에는 두 신호가 서로 다를 때만 위험한 XOR형 패턴이 있습니다. 직선 하나로는 네 점을 완전히 나눌 수 없지만, 은닉층에 ReLU를 넣어 입력 공간을 꺾으면 표현할 수 있습니다.

이번 실습에서는 학습 성능을 막연히 비교하지 않고, 선형층 합성의 수치 동등성과 비선형 hidden feature가 XOR을 분리하는 과정을 직접 확인합니다.

## 실습 목표

- 활성화 없는 여러 Linear가 하나의 아핀 변환으로 합쳐짐을 검증한다.
- 같은 출력 shape가 같은 표현력을 뜻하지 않음을 설명한다.
- ReLU hidden feature로 XOR형 패턴을 표현한다.
- 검증 결과와 구조 근거를 함께 사용해 후보를 선택한다.

## 진행 방식

- Linear는 bias를 포함한 아핀 변환으로 계산한다.
- ReLU는 `max(0,x)`로 사용한다.
- 출력 label은 logit 0 이상을 class 1로 둔다.

## 오늘의 업무 흐름

구조 확인 → 합성 가능성 계산 → hidden 값 비교 → XOR 예측 → 후보 승인

## 상황 자료

```
XOR 입력: (0,0), (0,1), (1,0), (1,1)
정답:       0      1      1      0
```

## 문제 1. 세 층이라는 설명의 허점 진단

### 업무 요청

Linear 두 층의 연속 계산과 합친 한 층의 계산이 같은지 확인해, 왜 활성화 없는 깊이가 표현력을 늘리지 않는지 보고하세요.

### 수행해야 할 작업

1. 두 층 출력을 계산한다.
2. 합성 weight와 bias를 계산한다.
3. 한 번의 계산과 근사 비교한다.
4. 출력 shape만으로 구조를 평가하면 안 되는 이유를 쓴다.

tensor([[ 1.5000, -0.5000],
        [-0.5000, -3.0000]]) tensor([ 0.2500, -2.2500])
tensor([3.7500, 2.2500])
tensor([3.7500, 2.2500])
근사 비교: True
- 선형층 두 개는 활성화 함수가 없다면 하나의 선형층으로 합칠 수 있다.


**해설**

접근 순서는 실제 두 단계 출력을 만든 뒤, 행렬 항과 bias 항을 각각 합쳐 한 단계와 비교하는 것입니다. 수치가 같다는 결과는 활성화 없는 두 Linear가 하나의 아핀 변환으로 표현됨을 보여줍니다. 층 이름과 출력 shape만 보고 "더 깊어서 비선형"이라고 쓰는 답은 값이 어떤 함수로 바뀌는지를 확인하지 않은 것입니다.

중간 feature 수가 커져 계산량이 늘어도 최종 입력-출력 관계가 선형이라는 핵심은 달라지지 않습니다. 다만 학습 과정의 수치적 성질까지 이 두 샘플로 결론 낼 수는 없습니다. 여기서 일반화할 수 있는 범위는 활성화가 전혀 없는 연속 아핀 변환의 표현 형태입니다.

**합성 뒤 확인할 계약:** 제출 전에는 합성 전후 값이 같은지만 보지 말고 merged weight와 bias의 shape도 원래 입력·출력 계약과 맞는지 확인합니다. 활성화를 하나 끼운 뒤에는 같은 합성식이 성립하지 않는다는 반례까지 말로 설명하면 선형 합성의 적용 조건을 분명히 할 수 있습니다.

## 문제 2. XOR을 구분하는 ReLU hidden 코드 작성

### 업무 요청

두 입력이 서로 다를 때만 활성화되는 hidden feature 두 개를 만들고 XOR label을 출력하세요. 이 문제는 학습이 아니라 주어진 weight로 표현 가능성을 확인합니다.

### 수행해야 할 작업

1. 네 XOR 입력을 batch로 만든다.
2. 두 차이 방향의 Linear 결과에 ReLU를 적용한다.
3. hidden 합에 threshold를 적용한다.
4. 네 정답과 정확히 같은지 확인한다.

### 제출해야 할 결과

- hidden 값, preds, `matches=True`를 제출한다.

h1 = ReLU(x1 - x2) → weight [1, -1]
h2 = ReLU(x2 - x1) → weight [-1, 1]
bias는 둘 다 0으로 둘 수 있습니다.


문제 2. XOR을 구분하는 ReLU hidden 코드 작성
XOR 입력:
tensor([[0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]])
XOR 라벨:
tensor([0, 1, 1, 0])
[0.0, 0.0]  XOR 정답 → 0
[0.0, 1.0]  XOR 정답 → 1
[1.0, 0.0]  XOR 정답 → 1
[1.0, 1.0]  XOR 정답 → 0
Linear 결과:
tensor([[ 0.,  0.],
        [-1.,  1.],
        [ 1., -1.],
        [ 0.,  0.]])
ReLU 결과:
tensor([[0., 0.],
        [0., 1.],
        [1., 0.],
        [0., 0.]])

보고

h1 = ReLU(x1 - x2) → weight [1, -1]
h2 = ReLU(x2 - x1) → weight [-1, 1]
bias는 둘 다 0으로 둘 수 있습니다.
이렇게 하면 두 입력이 같을 때는 hidden 값이 모두 0이고, 다를 때는 둘 중 하나만 1이 됩니다. 이후 두 hidden 값을 더해 0.5를 기준으로 분류하면 XOR 정답인 [0, 1, 1, 0]이 나옵니다.
다만 핵심은 학습을 통해 최적의 파라미터를 찾는 것이 아니라, 사람이 정한 weight와 ReLU를 이용해 선형층만으로는 구분하지 못했던 XOR을 비선형 은닉층에서는 표현할 수 있다는 점을 확인하는 것입니다
# XOR을 구분할 수 있도록 은닉 노드 2개의 가중치를 구성하는 문제였던 것.

hidden: [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
preds: [0, 1, 1, 0]
matches: True

**해설**

먼저 두 입력 차이를 반대 방향으로 계산하고, ReLU로 양수 부분만 남깁니다. 입력이 같으면 두 hidden이 모두 0이고, 다르면 하나가 1이 됩니다. 이 중간 표현은 마지막 선형 점수 하나로 XOR을 나눌 수 있게 합니다. 단순히 `x1*x2` 같은 새 feature를 답에 적는 것만으로는 MLP 내부에서 비선형 변환이 어떻게 쓰였는지 검증하지 못합니다.

weight는 사람이 고정했으므로 이 코드가 일반 데이터에서 스스로 규칙을 학습했다는 뜻은 아닙니다. 또한 ReLU를 넣기만 하면 모든 문제를 해결하는 것도 아닙니다. 이 결론은 네 XOR 점에서 활성화가 표현력을 바꿀 수 있다는 구성적 예시에 한정됩니다.

**XOR를 설명할 때:** hidden의 두 열이 각각 어떤 입력 차이를 보존하는지 행별로 읽고, 마지막 합과 threshold가 네 label을 어떻게 만드는지 추적합니다. weight를 사람이 골랐다는 점과 학습으로 얻은 모델이 아니라는 점을 보고서 첫 문장에 남겨 과도한 성능 해석을 막습니다.

### 자주 하는 실수

- ReLU 없이 두 차이 feature를 그대로 합쳐 0으로 상쇄한다.
- 네 점 일치 결과를 모든 안전 문장 일반화 성능으로 해석한다.

## 문제 3. 알려진 XOR 정답과 별도 검증을 구분해 후보 선택


### 업무 요청

문제 2의 XOR 네 점은 구조가 패턴을 표현할 수 있는지 확인한 단위 테스트입니다. 운영 후보 세 개에 대해 별도 noisy validation, parameter, latency 결과가 추가로 도착했습니다. 단위 테스트 통과와 운영 검증을 구분해 다음 실험안을 고르세요.

### 수행해야 할 작업

1. XOR 단위 테스트 통과 여부를 확인한다.
2. noisy validation accuracy 0.85 이상을 적용한다.
3. parameter 20 이하, latency 0.8ms 이하를 함께 적용한다.
4. 통과 후보를 다음 실험안으로 선택하고 배포 승인과 구분한다.


  후보           구조  XOR acc  XOR test
0  L  Linear-only     0.75     False
1  R  ReLU hidden     1.00      True
2  W    ReLU wide     1.00      True
=== XOR TEST RESULT ===
통과 후보: ['R', 'W']
통과 수: 2 / 3
  후보           구조  noisy valid acc  Noisy validation test
0  L  Linear-only             0.72                  False
1  R  ReLU hidden             0.86                   True
2  W    ReLU wide             0.91                   True
=== NOISY VALIDATION TEST RESULT ===
통과 후보: ['R', 'W']
통과 수: 2 / 3
  후보           구조  params  latency  Efficiency test
0  L  Linear-only       3      0.2             True
1  R  ReLU hidden       9      0.4             True
2  W    ReLU wide      41      1.2            False
=== EFFICIENCY TEST RESULT ===
통과 후보: ['L', 'R']
통과 수: 2 / 3
=== NEXT EXPERIMENT CANDIDATES ===
  후보           구조
1  R  ReLU hidden

=== DEPLOYMENT APPROVAL ===
배포 승인: 별도 절차 필요  # R은 "다음 실험을 진행할 후보"이지, "프로덕션 배포 승인 완료"가 아니라는 관점.

## checks: {'L': {'xor': False, 'validation': False, 'budget': True}, 'R': {'xor': True, 'validation': True, 'budget': True}, 'W': {'xor': True, 'validation': True, 'budget': False}}
## selected: R

**해설**

L은 비용이 작아도 XOR 단위 테스트와 별도 validation을 통과하지 못합니다. W는 두 정확도가 가장 높지만 parameter와 latency 예산을 넘습니다. R만 표현 가능성, 별도 validation, 운영 예산을 모두 만족하므로 다음 학습 실험안으로 선택합니다. 문제 2에서 본 네 점을 그대로 재사용해 R을 고르는 것이 아니라 새 증거와 비용을 함께 적용한 결정입니다.

noisy validation도 제한된 표본이므로 이 결과는 다음 실험안 승인입니다. 실제 배포에는 여러 seed, 실제 문서 분포, 오판 비용을 포함한 평가가 더 필요합니다. 활성화가 있다는 구조 정보는 후보 자격을 설명하지만 accuracy의 원인을 단독으로 증명하지는 않습니다.

**리뷰 메모:** W를 줄여 예산 안의 새 후보를 만들 경우 R과 같은 split·seed·epoch로 다시 비교합니다. 구조와 학습 조건을 동시에 바꾸면 noisy validation 차이의 원인을 설명할 수 없으므로 변경 변수도 보고서에 남깁니다.



# [4장 2강 심화] - ReLU 적용 실습 (1) : 뉴런별 활성 비율/뉴런별 양수 비율_ 활성 비율은 `activation > 0`의 평균으로 계산
                                      중간 활성 감사
 - 한 batch에서 0이라고 바로 dead ReLU로 확정하지 않고, 여러 batch에서 반복되는지 추가 확인해야
 
## 실습 배경

루멘 팀의 상담 분류 MLP는 shape와 loss가 정상이지만 특정 hidden 뉴런이 모든 batch에서 0만 냅니다. Linear 출력이 계속 음수라 ReLU 뒤 신호가 사라진 것입니다. 오류 메시지는 없고 parameter 수도 정상이라 중간 값을 보지 않으면 발견하기 어렵습니다.

ReLU는 음수를 0으로, 양수를 그대로 두며 shape를 바꾸지 않습니다. 은닉층 뒤에 두어 비선형성을 만들지만 마지막 출력층 뒤에 무조건 붙이면 음수 logits를 없애 문제 유형별 출력 계약을 바꿀 수 있습니다.

이번 실습에서는 pre-activation과 activation을 따로 기록하고, 뉴런별 활성 비율로 조용한 dead ReLU 징후를 진단합니다.

## 실습 목표

- ReLU 전후 값과 shape를 비교한다.
- 은닉층에서 ReLU를 적용하는 코드를 작성한다.
- 뉴런별 활성 비율로 계속 0인 경로를 찾는다.
- 한 batch의 징후와 장기 dead ReLU 판정을 구분한다.

## 진행 방식

- 활성 비율은 `activation > 0`의 평균으로 계산한다.
- 출력층은 raw logits로 남긴다.
- 한 batch의 0 비율만으로 영구적인 상태를 단정하지 않는다.

## 오늘의 업무 흐름

Linear 중간값 수집 → ReLU 적용 → 활성 비율 계산 → 구조 수정 후보 → 여러 batch 재검증

## 상황 자료

```
pre_activation neuron0=[-3,-4]
pre_activation neuron1=[0.5,1.5]
```
## 문제 1. shape 정상인 0 출력 진단

### 업무 요청

두 뉴런의 ReLU 출력과 활성 비율을 계산하고 어느 뉴런을 우선 점검할지 보고하세요.

### 수행해야 할 작업

1. Linear pre-activation을 계산한다.
2. ReLU를 적용한다.
3. 뉴런별 양수 비율을 계산한다.
4. 한 batch 결과의 해석 한계를 적는다.

### 제출해야 할 결과

- pre, activation, active_ratio, 점검 뉴런을 제출한다.

neuron0 pre activation [-3, -4]
neuron1 pre activation [0.5, 1.5]
neuron0 activation tensor([0, 0])
neuron1 activation tensor([0.5000, 1.5000])
neuron0: 0.0 %
neuron1: 100.0 %


## 문제 2. 중간 활성 감사를 포함한 MLP 구현

### 업무 요청

feature 2개를 hidden 2개, class 2개로 바꾸는 모델을 작성하고 forward에서 logits와 hidden 활성값을 반환하세요.

### 수행해야 할 작업

1. Linear 두 개와 ReLU를 등록한다.
2. ReLU를 첫 Linear 뒤에만 둔다.
3. hidden과 logits shape를 assert한다.
4. 활성 비율을 출력한다.

### 제출해야 할 결과

- 모델 코드, 두 shape, 전체 active ratio를 제출한다.

문제 2. 중간 활성 감사를 포함한 MLP 구현
input shape: torch.Size([4, 2])
input min/max: -1.2653828859329224 1.3999584913253784
after fc1 shape: torch.Size([4, 2])
after fc1 min/max: -0.6908833384513855 1.2315603494644165
after relu shape: torch.Size([4, 2])
after relu min/max: 0.0 1.2315603494644165
logits shape: torch.Size([4, 2])
DebugActivationMLP(
  (fc1): Linear(in_features=2, out_features=2, bias=True)
  (relu1): ReLU()
  (fc2): Linear(in_features=2, out_features=2, bias=True)
)
positive_ratio: 37.5 %

## 문제 3. 활성 분포 경보와 탈락 조건 구분

### 업무 요청

같은 검증 batch에서 세 설정의 hidden 양수 비율이 주어졌습니다. 여러 대표 batch에서 ratio 0이 반복되면 hard block, `0.25~0.75` 밖이지만 0은 아니면 재점검 경보로 둡니다. validation accuracy 0.80 이상과 함께 `즉시 다음 실험`, `추가 점검`, `차단`으로 분류하세요.

### 수행해야 할 작업

1. 반복된 ratio 0 증거가 있는 후보를 차단한다.
2. accuracy 0.80 미만을 제외한다.
3. 임시 비율 구간 밖의 후보는 탈락이 아니라 추가 점검으로 보낸다.
4. 즉시 다음 실험안과 C를 재확인할 자료를 보고한다.

### 상황 자료

```
A ratio=0.00 acc=0.82 zero_ratio_batches=5
B ratio=0.48 acc=0.84 zero_ratio_batches=0
C ratio=0.92 acc=0.86 zero_ratio_batches=0
```

### 제출해야 할 결과

- ready, investigate, blocked와 다음 실험안을 제출한다.

  후보  ratio   acc  zero_ratio_batches status
0  A   0.00  0.82                   5     차단
1  B   0.48  0.84                   0  다음 실험
2  C   0.92  0.86                   0  추가 점검


## [4장 3강 심화] - 이진 분류 출력 실습 (1)

실습 배경

루멘 팀의 개인정보 포함 여부 필터는 샘플마다 logit 하나를 출력합니다. 최근 코드 리뷰에서 모델 마지막에 Sigmoid를 넣은 뒤 `BCEWithLogitsLoss`에도 전달하는 이중 적용이 발견됐습니다. 실행 오류는 없지만 loss가 기대한 raw logit 계약을 받지 못합니다.

학습 단계는 `raw logits → BCEWithLogitsLoss`, 해석 단계는 `logits → sigmoid → threshold → label`로 분리해야 합니다. 기본 확률 threshold 0.5는 logit 0과 대응하지만, 오판 비용이 다르면 validation 데이터에서 다른 threshold를 고를 수 있습니다.

이번 실습은 shape·dtype·Sigmoid 위치를 함께 점검하고 실제 비용으로 threshold를 선택합니다.

## 실습 목표

- 이진 분류의 `(B,1)` raw logit 계약을 설명한다.
- BCEWithLogitsLoss 앞의 이중 Sigmoid를 진단한다.
- 확률과 threshold로 label을 구현한다.
- FN/FP 비용을 반영해 운영 threshold를 선택한다.

## 진행 방식

- target은 logits와 같은 shape의 float다.
- loss에는 raw logits를 넣는다.
- 확률은 보고·추론 단계에서만 만든다.

## 오늘의 업무 흐름

출력 계약 검사 → loss 입력 확인 → sigmoid 해석 → threshold 예측 → 비용 비교

## 상황 자료

```
logits=[[0.0],[2.0]], target=[[0.0],[1.0]]
broken: loss_fn(sigmoid(logits), target)
```

## 문제 1. 이중 Sigmoid의 silent bug 진단

### 업무 요청

올바른 loss와 잘못된 loss를 같은 입력에서 비교하고, 왜 값이 달라지는지 설명하세요.

### 수행해야 할 작업

1. logits/target shape와 dtype을 확인한다.
2. raw logits로 loss를 계산한다.
3. Sigmoid 선적용 loss를 계산한다.
4. 모델 마지막 층의 수정안을 쓴다.

### 제출해야 할 결과

- correct/wrong loss와 수정 원칙을 제출한다.

logits shape: torch.Size([2, 1])
logits dtype: torch.float32
target shape: torch.Size([2, 1])
target dtype: torch.float32
loss: tensor(0.4100)
loss_sigmoid: tensor(0.6604)


**해설**

먼저 shape와 float target 계약이 맞는지 확인한 뒤 입력 변환만 달리해 loss를 비교합니다. BCEWithLogitsLoss는 내부에서 안정적인 Sigmoid+BCE 계산을 하므로 외부 Sigmoid를 통과한 0~1 값을 다시 logit처럼 처리하면 다른 목적함수가 됩니다. 실행이 된다는 이유로 정상이라고 판단하는 답은 API 입력 계약을 놓칩니다.

수정은 모델 마지막 Sigmoid를 제거해 raw logit을 loss에 전달하고, 확률이 필요할 때 별도 `torch.sigmoid`를 호출하는 것입니다. 표시된 두 loss의 크기만으로 모델 품질을 비교할 수는 없습니다. 동일한 parameter에서 계산식 차이를 확인한 단위 테스트에 한정됩니다.

**수정 뒤 볼 로그:** 모델 구조에서 Sigmoid가 빠졌는지와 loss 호출 직전 값 범위가 0~1로 제한되지 않았는지 함께 확인합니다. 추론 화면에서만 확률을 만들도록 함수 경계를 분리하면 이후 코드 리뷰에서 학습과 해석 경로가 다시 섞이는 일을 줄일 수 있습니다.

## 문제 2. 이진 추론 함수를 직접 작성

### 업무 요청

raw logits와 확률 threshold를 받아 확률·label을 반환하는 함수를 작성하세요. 입력은 반드시 `(B,1)`이어야 합니다.

### 수행해야 할 작업

1. 2차원 마지막 길이 1을 검사한다.
2. Sigmoid로 확률을 만든다.
3. threshold 이상을 1로 만든다.
4. 반환 shape를 입력과 같게 유지한다.

### 제출해야 할 결과
- 함수, 확률, preds를 제출한다.

logits shape: torch.Size([4, 1])
raw logits: tensor([[-1.],
        [ 0.],
        [ 1.],
        [ 2.]])
probs: tensor([[0.2689],
        [0.5000],
        [0.7311],
        [0.8808]])
preds: tensor([[0],
        [0],
        [1],
        [1]])
input shape : torch.Size([4, 1])
preds shape : torch.Size([4, 1])
shape test: PASS

**해설**
## threshold 경계: 0.5 포함 초과 똑바로 판단할 것
입력 계약을 먼저 검사하고, Sigmoid와 threshold를 순서대로 적용합니다. 0.7은 확률 기준이므로 logit 0.7과 직접 비교하면 다른 경계가 됩니다. 인자 없는 `squeeze()`를 함수 내부에서 써 반환하면 batch 1에서 축이 사라질 수 있으므로 `(B,1)`을 그대로 유지합니다.

이 함수의 Sigmoid 값은 0~1 범위라 확률처럼 읽을 수 있지만 실제 빈도와 잘 맞는다는 보장은 없습니다. threshold도 이 네 값으로 결정하지 않습니다. 대표 validation 데이터와 오판 비용으로 선택하고, 함수는 선택된 기준을 일관되게 적용하는 역할만 합니다.

**threshold 경계 테스트:** batch 1과 threshold 경계값인 확률 0.7 사례도 추가로 테스트해 이상과 초과 중 어느 쪽을 양성으로 정의했는지 고정합니다. 이 문제는 이상을 사용했으므로 정확히 0.7도 class 1이며, 서비스 정책이 바뀌면 비교 연산과 문서를 함께 갱신해야 합니다.


## 문제 3. 누락 비용을 반영한 threshold 선택

### 업무 요청

개인정보 문서를 놓치는 FN 비용은 5, 정상 문서를 추가 차단하는 FP 비용은 2입니다. threshold 0.5와 0.7을 같은 validation 네 건에서 비교하세요.

### 수행해야 할 작업

1. 후보별 label을 만든다.
2. FN과 FP를 센다.
3. `5*FN+2*FP`를 계산한다.
4. 비용 최소 threshold와 데이터 한계를 보고한다.

### 상황 자료

```
probs=[0.90,0.65,0.55,0.40]
target=[1,0,1,0]
```

### 제출해야 할 결과

- 후보별 비용과 selected threshold를 제출한다.

=== 결과 ===
   threshold  FN  FP  cost
0        0.5   0   1     2
1        0.7   1   0     5

비용 최소 threshold: 0.5
최소 비용: 2.0

=== 데이터 한계 ===
validation 데이터: 4 건
데이터가 4건뿐이므로 threshold 일반화 성능을 확정할 수 없음

**해설**

threshold 0.5는 FP 하나로 비용 2, 0.7은 FN 하나로 비용 5가 됩니다. 개인정보 누락 비용이 더 크다는 업무 조건 때문에 0.5를 선택합니다. 단순 정확도는 두 후보 모두 0.75라 구분하지 못하므로 오류 방향을 직접 세어야 합니다. 가장 높은 threshold가 더 안전하다는 직관도 이 비용 구조에서는 틀립니다.

네 건은 계산 연습용이라 운영 threshold를 확정하기에 너무 작습니다. 실제 선택은 대표 validation set에서 하고, 데이터 비율이나 비용 정책이 바뀌면 다시 검증해야 합니다. 이 결론을 다른 서비스나 다른 양성 정의에 그대로 일반화할 수 없습니다.

**선택 편향을 막으려면:** threshold를 고른 뒤 같은 validation 자료로 모델을 다시 학습하거나 후보를 무한히 늘리면 선택 편향이 커집니다. 가능한 후보와 비용식을 먼저 정하고 선택 후에는 별도 데이터에서 고정 threshold를 재검증해야 운영 보고의 의미가 유지됩니다.


# [4장 4강 심화] - 다중 분류 출력 실습 (1)

## 실습 배경

루멘의 문서 라우터는 인사·재무·법무 세 class 중 하나를 고릅니다. 모델은 `(B,3)` logits를 정상 출력하지만, 한 구현이 Softmax를 batch 축 `dim=0`에 적용했습니다. 모든 열의 합은 1이라 단순 검사에는 통과했지만 샘플별 확률 분포가 아니었습니다.

다중 분류 학습은 raw logits와 `(B,)` long class index를 `CrossEntropyLoss`에 전달합니다. Softmax는 결과를 해석할 때 class 축에 적용하고, class만 고를 때는 logits에서 바로 argmax해도 같습니다.

이번 실습에서는 축 계약·loss 입력·class 선택을 분리하고, 운영 후보의 확률 합과 target 형식까지 함께 검사합니다.

## 실습 목표

- `(B,C)` logits에서 class 축을 식별한다.
- 잘못된 Softmax dim을 row sum으로 진단한다.
- raw logits·long target의 CrossEntropy 계약을 구현한다.
- 구조와 검증 수치를 함께 사용해 후보를 선택한다.

## 진행 방식

- class는 마지막 차원이며 Softmax `dim=-1`을 사용한다.
- target은 one-hot이 아닌 기본 class index long이다.
- loss 앞에는 Softmax를 두지 않는다.

## 오늘의 업무 흐름

logits shape 확인 → class 축 Softmax → row sum 검사 → argmax → loss/후보 승인



## 상황 자료

```
logits=[[2,1,0],[0,1,2]]
expected: 각 행의 class 확률 합=1
```

## 문제 1. 합이 1인데도 잘못된 Softmax 진단

### 업무 요청

`dim=0`과 `dim=1` 결과의 행 합을 비교해 어떤 축이 잘못됐는지 증명하세요.

### 수행해야 할 작업

1. 두 방향 Softmax를 계산한다.
2. 각 행의 합을 구한다.
3. 샘플별 분포 조건을 적용한다.
4. 열 합만 확인한 검사가 왜 부족한지 쓴다.

### 제출해야 할 결과

- wrong/correct row sums와 올바른 dim을 제출한다.

tensor([[0.8808, 0.5000, 0.1192],
        [0.1192, 0.5000, 0.8808]])
tensor([[0.6652, 0.2447, 0.0900],
        [0.0900, 0.2447, 0.6652]])
각 행의 합: tensor([1.5000, 1.5000])    # wrong row sums (dim=0)
각 행의 합: tensor([1., 1.])            # correct row sums (dim=1)
pass

**해설**

먼저 logits 축 의미를 확인하고 두 Softmax 결과를 행 방향으로 합산합니다. `dim=0`은 같은 class 열을 서로 다른 샘플끼리 정규화해 각 행 합이 1.5가 됩니다. 전체 Tensor나 열 합만 1인지 보는 답은 업무 단위인 샘플별 분포를 검증하지 못합니다. 이 페이지에서는 class가 두 번째 축이라 dim=1이 맞습니다.

class가 항상 마지막 차원이라는 코드 계약에서는 `dim=-1`이 더 일반적입니다. 그러나 축 순서가 다른 데이터를 무조건 마지막 차원으로 가정하면 또 다른 오류가 됩니다. 따라서 shape 표기와 dim 선택을 함께 남겨야 하며, 합이 1이라는 조건만으로 확률 보정까지 보장되지는 않습니다.

**class 축 회귀 테스트:** dim을 수정한 뒤에는 행 합뿐 아니라 각 행의 argmax가 원래 logits argmax와 같은지도 확인할 수 있습니다. 열 합 1 검사를 삭제하는 데서 끝내지 말고 class 축이 마지막이라는 shape 계약을 함수 주석과 assert로 남겨 다음 차원 변경 때 다시 발견되도록 합니다.


## 문제 2. 다중 분류 학습·추론 계약 코드 작성

### 업무 요청

raw logits에서 loss, class 확률, 예측 class를 만드는 함수를 작성하세요. target 계약 위반은 loss 호출 전에 막아야 합니다.

### 수행해야 할 작업

1. logits `(B,C)`와 target `(B,)`를 검사한다.
2. target long과 class 범위를 검사한다.
3. raw logits로 CrossEntropyLoss를 계산한다.
4. 확률과 argmax 예측을 반환한다.

### 제출해야 할 결과

- 함수, loss scalar 여부, row sums, preds를 제출한다.

logits shape: torch.Size([2, 3])
target shape: torch.Size([2])
shape test: PASS                    #
target dtype: torch.int64
target range: 0 ~ 2
target test: PASS                   #
probs:
tensor([[0.6652, 0.2447, 0.0900],
        [0.0900, 0.2447, 0.6652]])
preds_from_logits: tensor([0, 2])   #

**해설**

공통 batch, target 차원, dtype, class 범위를 loss 전에 검사합니다. CrossEntropyLoss에는 raw logits를 전달하고, 사람에게 class별 값을 보여줄 때만 Softmax를 계산합니다. target을 `(B,1)`로 유지하거나 float로 바꾸는 답은 기본 class index 계약과 다릅니다. 모델 마지막 Softmax와 loss 내부 처리를 중복시키는 것도 같은 오류입니다.

`argmax(logits)`와 `argmax(probs)`는 순서가 같아 class 예측만 필요하면 Softmax를 생략할 수 있습니다. 이 함수는 샘플마다 하나의 class가 정답인 경우에 한정됩니다. 여러 label이 동시에 참인 업무에는 출력과 loss 계약을 다시 정의해야 합니다.

**label 사전까지 볼 이유:** target의 최솟값과 최댓값을 검사해 음수 label이나 class 수 이상의 index도 loss 전에 차단해야 합니다. class 이름 순서가 바뀌는 문제는 값 범위만으로 잡히지 않으므로 모델 출력 열과 label 사전의 버전도 함께 관리하는 것이 실제 운영에서 중요합니다.


## 문제 3. 라우터 후보의 운영 계약 승인

### 업무 요청

세 후보의 validation 기록을 비교합니다. 정확도 기준은 0.85 이상이고, 샘플별 확률 합 오차는 `1e-6` 이하여야 하며 target은 long class index여야 합니다. 안전 부서 class recall 0.75 이상도 승인 자료에 필요하며, 값이 아직 없으면 실패가 아니라 재측정으로 분류합니다.

### 수행해야 할 작업

1. Tensor 계약 위반 후보를 제외한다.
2. 필수 recall이 누락된 후보를 재측정으로 분리한다.
3. 모든 자료가 있는 후보에 accuracy와 recall 기준을 적용한다.
4. 승인·재측정·탈락 사유를 각각 보고한다.

### 상황 자료

```
A acc=0.88 row_sum_error=0.50 target=float safety_recall=0.81
B acc=0.86 row_sum_error=0.00 target=long  safety_recall=0.78
C acc=0.90 row_sum_error=0.00 target=long  safety_recall=미측정
```

### 제출해야 할 결과

- approved, remeasure, rejected, selected를 제출한다.

  candidate  row_sum_error  row_sum_ok
0         A            0.5       False
1         B            0.0        True
2         C            0.0        True
  candidate   acc  safety_recall     status
0         A  0.88           0.81   rejected
1         B  0.86           0.78   approved
2         C  0.90            NaN  remeasure

**해설**

A는 확률 합과 target dtype 계약이 틀려 metric 자체를 신뢰할 수 없습니다. B는 Tensor 계약, accuracy, 안전 부서 recall을 모두 만족해 현재 승인 후보입니다. C는 accuracy가 가장 높고 Tensor 계약도 맞지만 안전 recall이 미측정이라 재측정으로 보냅니다. 누락값을 0으로 간주해 탈락시키거나 0.90만 보고 승인하는 답은 모두 증거 상태를 잘못 해석합니다.

B 선택은 C의 안전 recall 측정이 끝날 때까지의 1차 승인입니다. C가 0.75를 통과하면 같은 고정 validation 기준에서 accuracy를 다시 비교할 수 있습니다. class 불균형과 부서별 오류 비용까지 이 표가 설명하지는 않으므로 전체 배포 확정과는 구분해야 합니다.

**승인 뒤 확인할 품질:** B를 승인한 뒤에는 전체 accuracy 외에 부서별 누락 사례를 별도로 확인합니다. class 비율이 크게 다르면 0.86이 빈도가 높은 한 부서 성능에 좌우될 수 있으므로, 여기서 정한 Tensor 계약 통과와 업무 품질 통과를 서로 다른 체크 항목으로 유지합니다.
