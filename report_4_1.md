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


## 문제 3. 알려진 XOR 정답과 별도 검증을 구분해 후보 선택


### 업무 요청

문제 2의 XOR 네 점은 구조가 패턴을 표현할 수 있는지 확인한 단위 테스트입니다. 운영 후보 세 개에 대해 별도 noisy validation, parameter, latency 결과가 추가로 도착했습니다. 단위 테스트 통과와 운영 검증을 구분해 다음 실험안을 고르세요.

### 수행해야 할 작업

1. XOR 단위 테스트 통과 여부를 확인한다.
2. noisy validation accuracy 0.85 이상을 적용한다.
3. parameter 20 이하, latency 0.8ms 이하를 함께 적용한다.
4. 통과 후보를 다음 실험안으로 선택하고 배포 승인과 구분한다.



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