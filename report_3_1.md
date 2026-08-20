# [3장 1강 심화] - 퍼셉트론 계산 실습 (1)

## 실습 배경

루멘 팀은 고비용 LLM 호출 전에 단순 안전 신호 두 개로 요청을 1차 분류하는 작은 점수기를 검토합니다. 입력은 `금칙어 일치도`와 `정상 업무 문구 일치도`, 출력은 확률이 아닌 가중합 점수 하나입니다. 점수가 0 이상이면 정밀 검토 큐로 보냅니다.

운영자는 가중치가 큰 feature가 무조건 중요하다고 보고했지만 두 feature의 단위가 다릅니다. 또 bias를 바꾸자 같은 가중치에서도 검토 건수가 달라졌습니다. 퍼셉트론의 가중치·편향·선형 결정 경계를 분리해 해석해야 합니다.

이번 실습에서는 학습 알고리즘 대신 이미 주어진 `w`, `b`로 batch 점수를 계산하고, 오판 비용에 따라 bias 후보를 선택합니다.

## 실습 목표

- `X @ w + b`의 batch shape와 값을 계산한다.
- logit 점수와 확률을 구분한다.
- 가중치 부호·크기와 편향의 역할을 설명한다.
- 선형 경계 안에서 오판 비용 기반 운영 결정을 내린다.

## 진행 방식

- 예측 규칙은 `logit >= 0`이면 1이다.
- feature 단위가 다르면 weight 절댓값만으로 중요도를 단정하지 않는다.
- 문제 3의 비용은 false negative 5, false positive 1로 고정한다.

## 오늘의 업무 흐름

입력/가중치 shape 확인 → 가중합 계산 → 경계 적용 → 오판 유형 집계 → bias 선택

## 상황 자료

```
X.shape=(3,2), w.shape=(2,), b=scalar
검토 큐 label: 1
```
## 문제 1. 점수를 확률로 오해한 운영 로그 진단

### 업무 요청

담당자가 logit `1.0`을 "위험 확률 100%"라고 보고했습니다. 실제 batch 점수와 예측을 계산하고 잘못된 해석을 바로잡으세요.

### 수행해야 할 작업

1. 세 샘플의 가중합을 계산한다.
2. 결과 shape를 확인한다.
3. 0 기준 label을 만든다.
4. logit을 확률로 부를 수 없는 이유를 설명한다.


```bash
z      : tensor([ 1.0000, -0.8000, -1.9000])
z shape     : (3,)
preds       : [1, 0, 0]
```

**해설**
먼저 행렬 곱의 shape를 확인한 뒤 bias와 경계를 적용합니다. `1.0`은 경계에서 양의 방향으로 떨어진 점수이지 100% 확률이 아닙니다. 값 범위에도 제한이 없습니다. 점수의 부호로 현재 규칙의 class만 판단할 수 있습니다. 이 결론은 주어진 선형 점수기에 대한 것이며 점수를 확률로 바꾸는 방법은 뒤 강의 범위입니다.

**오답 점검:** 각 feature와 weight를 원소별로 곱한 뒤 샘플 내부에서 합쳐야 합니다. batch 전체를 한 번에 합산하면 logits shape가 scalar가 되고, 점수 1.0을 확률이나 퍼센트로 보고하면 아직 적용하지 않은 변환을 가정한 것입니다.

**적용 범위와 한계:** 이 계산은 주어진 weight, bias와 threshold 0의 선형 점수기에만 적용됩니다. feature 단위가 다르면 weight 절댓값으로 중요도를 단순 비교할 수 없고, 세 샘플 결과만으로 실제 안전 필터의 일반화 성능이나 확률 신뢰도를 판단할 수 없습니다.


## 문제 2. shape 계약이 있는 퍼셉트론 함수 작성

### 업무 요청

입력 feature 수가 바뀌었을 때 조용히 잘못 계산되지 않도록 batch 전용 함수를 작성하세요.

### 수행해야 할 작업

1. X는 2차원, w는 1차원인지 검사한다.
2. 마지막 feature 수가 같은지 검사한다.
3. scalar bias만 허용한다.
4. logits와 preds를 함께 반환한다.

logits shape: torch.Size([3, 2])
preds shape: torch.Size([3, 2])

**해설**

연산 전에 차원 수와 feature 길이를 확인하면 오류 메시지를 모델 계약에 맞게 좁힐 수 있습니다. bias를 `(B,)`로 허용하면 샘플마다 다른 경계를 쓰게 되어 하나의 퍼셉트론이라는 해석이 깨질 수 있습니다. 함수는 binary threshold용이며 다중 class 출력에는 그대로 일반화되지 않습니다.

**오답 점검:** X의 첫 차원을 feature 수로 검사하면 batch 크기가 바뀔 때 오작동합니다. bias vector를 batch 길이로 허용하면 샘플마다 서로 다른 경계를 쓰게 되어 하나의 공유 퍼셉트론이라는 모델 정의가 달라집니다.

**적용 범위와 한계:** 이 함수는 표 형태의 batch 입력과 binary threshold를 위한 최소 계약입니다. class가 여러 개이거나 샘플마다 여러 출력이 필요하면 weight와 output shape를 새로 정의해야 하며, assert 통과가 feature 열의 순서와 값 스케일까지 보증하지는 않습니다.


## 문제 3. 오판 비용으로 bias 후보 선택

### 업무 요청

위험 요청을 놓치는 비용이 정상 요청을 추가 검토하는 비용보다 5배 큽니다. 같은 weight에서 bias 두 개를 비교해 총 비용이 낮은 후보를 고르세요.

### 수행해야 할 작업

1. 후보별 예측을 계산한다.
2. false negative와 false positive를 센다.
3. `5*FN + 1*FP`를 계산한다.
4. 선택과 선형 모델의 한계를 보고한다.


bias: 0.0 logits: tensor([ 1.0000, -0.6000,  0.1000])
bias: 0.0 FN: 1 FP: 1
bias: 0.7 logits: tensor([1.7000, 0.1000, 0.8000])
bias: 0.7 FN: 0 FP: 1
[{'bias': 0.0, 'FN': 1, 'FP': 1, 'cost': 6}, {'bias': 0.7, 'FN': 0, 'FP': 1, 'cost': 1}]
selected bias: 0.7
selected cost: 1

결과 보고
bias == 0.0 의 경우 cost가 6. 
bias == 0.7 의 경우 cost가 1.
selected bias: bias ==0.7

선형 보고의 한계(선형 결정 경계) : 비용 기준으로는 bias=0.7을 선택한다. FN 비용이 5배이므로 FN을 1→0으로 줄이는 것이 FP 1을 유지하는 것보다 총 비용을 크게 감소시킨다. 다만 현재 모델은 X @ w + b에 기반한 단일 선형 결정 경계만 사용할 수 있으므로, 데이터가 선형적으로 분리되지 않는 경우에는 bias 조정만으로 원하는 분류 성능을 얻는 데 한계가 있다.
## 주어진 weight와 비용 함수 안에서 가장 싼 선택일 뿐이다. ##

**해설**
  비용을 명시하지 않으면 단순 정확도가 운영 목표를 대신하게 됩니다. `b=0.7`은 양성 방향으로 경계를 이동해 FN을 없애고 FP 하나를 감수하므로 비용이 낮습니다. 이 선택은 주어진 세 샘플과 비용 비율에서만 유효합니다. 데이터가 직선 하나로 나뉘지 않으면 bias 조정만으로 해결할 수 없습니다.

**오답 점검:** 정확도만 비교하면 두 오류의 운영 비용 차이를 잃습니다. bias 0.7이 양성 예측을 늘렸다는 사실만 보고 안전하다고 결론 내리지 말고, false negative와 false positive를 별도로 세어 합의한 비용식에 넣어야 합니다.

**적용 범위와 한계:** 선택은 세 샘플과 수업용 비용 정책에 종속됩니다. 데이터 비율이나 비용이 바뀌면 최적 bias도 달라지고, 직선 하나로 나뉘지 않는 패턴은 bias 이동만으로 해결할 수 없으므로 더 복잡한 표현을 검토해야 합니다. 최종 보고에는 후보별 FN·FP 원자료도 남겨 비용식 변경 때 다시 계산할 수 있어야 합니다.


# [3장 2강 심화] - MLP 구조 실습 (1)

## 실습 배경

루멘 팀은 문서 임베딩 6개 feature를 세 업무 class 점수로 바꾸는 작은 MLP를 붙이려 합니다. 설계 문서에는 hidden size가 6, 코드에는 다음 층의 입력이 8로 적혀 있어 forward가 중간에서 멈췄습니다.

또한 "층을 깊게"와 "층을 넓게"라는 표현만 있고 parameter 예산이 없습니다. 모델 구조를 검토할 때는 각 층의 `(B, feature)` 흐름과 학습 가능한 Linear 층 수, parameter 수를 함께 써야 합니다.

이번 실습에서는 입력 데이터 자체는 층 수에서 제외하고 학습 가능한 Linear 층을 기준으로 셉니다.

## 실습 목표

- 입력층·은닉층·출력층 역할을 shape로 설명한다.
- batch 크기는 유지되고 feature 차원이 변함을 검증한다.
- width·depth 변화와 parameter 수를 연결한다.
- 예산 안에서 구조 후보를 근거 있게 선택한다.

## 진행 방식

- 입력 shape는 `(B,6)`, 출력은 `(B,3)`이다.
- Linear parameter는 `in*out + out`으로 센다.
- ReLU는 shape를 유지하는 활성화로만 사용한다.

## 오늘의 업무 흐름

구조도 읽기 → 층별 shape 추적 → 모델 구현 → parameter 계산 → 예산 내 후보 선택

## 상황 자료

```
설계: 6 -> 6 -> 3
코드: Linear(6,6) -> ReLU -> Linear(8,3)
```
+ 입력 shape는 (B,6), 출력은 (B,3)

## 문제 1. 중간 shape 로그로 첫 오류 찾기

### 업무 요청

설계와 코드 메타데이터를 비교해 forward가 실패하는 정확한 경계를 지정하세요.

### 수행해야 할 작업

1. 첫 Linear 출력 feature를 구한다.
2. ReLU 뒤 shape가 바뀌는지 판단한다.
3. 둘째 Linear의 `in_features`와 비교한다.
4. 최소 수정과 대안 수정의 차이를 설명한다.

### 제출해야 할 결과

- 오류 경계, actual/expected feature, 최소 수정안을 제출한다.

X shape     : torch.Size([5, 6])
hidden_linear   : torch.Size([5, 6])
hidden_activated: torch.Size([5, 6])
logits          : torch.Size([5, 3])

# 4. 최소 수정과 대안 수정의 차이를 설명한다.
## 설계와 코드 메타데이터를 비교해 forward가 실패하는 정확한 경계를 지정.
### 오류 경계, actual/expected feature, 최소 수정안을 제출??

# layer별 (B,F) shape를 순서대로 추적해 첫 in_features 불일치에서 진단을 멈춥니다.
# 마지막 RuntimeError만 보고 뒤 layer를 바꾸지 않고 앞 출력과 다음 입력의 계약을 대조합니다.

first_issue: fc2: expected 8, got 6
minimal_fix: Linear(6, 3)

**해설**

shape를 입력부터 한 층씩 전달하면 fc2 경계에서 처음 계약이 깨집니다. 설계가 `6→6→3`이므로 최소 수정은 fc2의 입력을 6으로 바꾸는 것입니다. fc1 출력을 8로 바꾸는 대안도 실행은 되지만 모델 설계를 함께 변경하므로 별도 승인 대상입니다. 출력 shape만 맞춘다고 두 구조가 같은 모델은 아닙니다.

**오답 점검:** 둘째 Linear의 오류를 피하려고 hidden Tensor를 8개로 임의 padding하면 설계 자체를 바꿉니다. ReLU가 shape를 바꿀 것이라고 추측하거나 output 3만 맞추면 된다고 보는 답도 첫 실패 경계를 설명하지 못합니다.

**적용 범위와 한계:** 이 추적은 순차적으로 연결된 단일 경로 MLP에 맞습니다. 여러 입력 분기처럼 경로가 합쳐지는 모델에서는 각 경로 shape와 결합 연산을 별도로 추적해야 하며, 실행 가능한 shape가 의미 있는 구조를 보장하지는 않습니다.


## 문제 2. 층별 shape를 반환하는 MLP 작성

### 업무 요청

입력 6, hidden 4, class 3인 MLP를 만들고 forward 검증 시 중간 shape도 반환하세요.

### 수행해야 할 작업

1. 두 Linear를 `__init__`에 등록한다.
2. 첫 Linear 뒤 ReLU를 적용한다.
3. logits에는 별도 후처리를 넣지 않는다.
4. batch 5개의 shape 흐름을 검증한다.



input  : torch.Size([5, 6])
flatten: torch.Size([5, 6])
fc1    : torch.Size([5, 4])
relu   : torch.Size([5, 4])
fc2    : torch.Size([5, 3])
parameters: 43

# 등록된 Linear와 activation을 순서대로 사용하고 hidden·logits를 함께 반환해 중간 계약을 노출합니다.
# 더미 forward는 shape 연결 검증일 뿐 학습 성능 증거가 아니라는 범위를 유지합니다.


**해설**
`fc1`이 feature 6을 4로 바꾸고 ReLU는 shape를 유지합니다. `fc2`가 4를 class 점수 3개로 바꿉니다. parameter는 `6*4+4 + 4*3+3 = 43`입니다. hidden을 반환하는 방식은 디버깅용이며 운영 API가 항상 중간 Tensor를 노출해야 한다는 뜻은 아닙니다.

**오답 점검:** bias를 parameter 수에서 제외하면 실제 비용을 놓칩니다. forward 안에서 Linear를 생성하면 호출마다 새 parameter가 생기고 optimizer에 안정적으로 등록되지 않으므로 생성자 등록 원칙을 지켜야 합니다.

**적용 범위와 한계:** hidden을 반환하는 것은 진단 편의를 위한 설계이며 운영 API의 필수 형식은 아닙니다. 이 더미 forward는 shape와 parameter 연결만 확인하고, **loss 감소와 분류 정확도** 는 후속 학습과 validation에서 별도로 검증해야 합니다.


## 문제 3. 65개 parameter 예산의 구조 선택

### 업무 요청

세 후보 중 parameter가 65개 이하이면서 학습 가능한 Linear 층 수가 가장 많은 구조를 고르세요. 동률이면 parameter가 적은 것을 선택합니다.

### 수행해야 할 작업

1. 각 Linear의 weight와 bias를 센다.
2. 후보 총 parameter를 계산한다.
3. 예산 초과 후보를 제외한다.
4. 선택 기준과 표현력 판단의 한계를 쓴다.


### 상황 자료

```
A: 6 -> 4 -> 3
B: 6 -> 8 -> 3
C: 6 -> 4 -> 4 -> 3

결과 보고

# 모든 weight와 bias의 numel을 합쳐 parameter 예산을 먼저 적용합니다.
# 예산 통과 구조 안에서만 설계 조건을 비교하고 layer 수를 원소 수로 오해하지 않습니다.

```
문제 3. 65개 parameter 예산의 구조 선택
model_A
0.weight torch.Size([4, 6]) numel= 24
model_A
0.bias torch.Size([4]) numel= 4
model_A
2.weight torch.Size([3, 4]) numel= 12
model_A
2.bias torch.Size([3]) numel= 3
model_B
0.weight torch.Size([8, 6]) numel= 48
model_B
0.bias torch.Size([8]) numel= 8
model_B
2.weight torch.Size([3, 8]) numel= 24
model_B
2.bias torch.Size([3]) numel= 3
model_C
0.weight torch.Size([4, 6]) numel= 24
model_C
0.bias torch.Size([4]) numel= 4
model_C
2.weight torch.Size([4, 4]) numel= 16
model_C
2.bias torch.Size([4]) numel= 4
model_C
4.weight torch.Size([3, 4]) numel= 12
model_C
4.bias torch.Size([3]) numel= 3
model_A total_params=43
model_B total_params=83
model_C total_params=63
model_A: 43 → 선택
model_B: 83 → 제외
model_C: 63 → 선택

**해설**

B는 넓지만 예산을 넘습니다. A와 C 중 C가 Linear 층이 하나 더 많아 주어진 선택 규칙에서 승인됩니다. parameter 수와 깊이만으로 실제 성능을 확정할 수 없으며, 활성화와 학습 결과를 같은 검증 조건에서 비교해야 합니다. 여기서는 구조 검토 규칙만 적용합니다.

**오답 점검:** width만 큰 B를 표현력이 높다고 즉시 고르면 예산 65를 위반합니다. 반대로 parameter가 가장 적은 A를 자동 선택하면 팀이 정한 Linear 층 수 우선 규칙을 무시하므로 필터와 정렬 기준을 순서대로 적용해야 합니다.

**적용 범위와 한계:** C 선택은 parameter 예산과 구조 규칙에 따른 설계 검토 결과입니다. 실제 성능은 활성화, 초기값, optimizer와 데이터에 따라 달라지며, 깊이가 하나 늘었다는 사실만으로 validation 우위를 보장할 수 없습니다. 선택된 구조는 더미 forward로 최종 shape까지 다시 확인한 뒤 학습 실험으로 넘깁니다.


# [3장 3강 심화] - Linear layer shape 실습

## 실습 배경

루멘 팀의 분류 head를 다른 코드로 옮기는 과정에서 weight 행렬 방향이 뒤집혔습니다. 문서에는 `Linear(3,4)`라고 적었지만 수동 계산 코드는 `x @ weight`를 호출했고, PyTorch가 저장하는 `(out,in)` 계약을 놓쳤습니다.

또한 모델 요약에는 weight만 집계되어 bias가 빠졌습니다. 작은 차이는 여러 층에서 누적되어 parameter 예산과 checkpoint 검토를 어긋나게 합니다.

이번 실습은 `nn.Linear(in_features,out_features)`가 `x @ weight.T + bias`로 계산된다는 사실을 수동 계산과 parameter 감사로 검증합니다.

## 실습 목표

- Linear weight `(out,in)`과 bias `(out,)`를 계산한다.
- 마지막 입력 차원이 `in_features`와 같아야 함을 검증한다.
- 모듈 출력과 수동 행렬 계산을 비교한다.
- parameter 예산에 weight와 bias를 모두 반영한다.

## 진행 방식

- batch 입력은 `(B,in)`으로 둔다.
- 수동 계산은 `x @ weight.T + bias`를 사용한다.
- 출력값 비교는 허용 오차를 고려한다.

## 오늘의 업무 흐름

Linear 선언 읽기 → parameter shape 예측 → 수동 계산 → 모듈 비교 → 예산 승인

## 상황 자료

```
layer=Linear(3,4)
x.shape=(2,3)
stored weight.shape=(4,3)
```

## 문제 1. 전치 누락 원인 진단

### 업무 요청

`x @ weight`가 실패한 이유를 shape만으로 설명하고, 올바른 곱의 shape를 출력하세요.

### 수행해야 할 작업

1. x와 weight의 행렬 곱 내부 차원을 비교한다.
2. 저장 weight를 전치해야 하는 이유를 쓴다.
3. 올바른 출력 shape를 계산한다.
4. weight shape를 `(in,out)`로 저장한다고 오해한 지점을 지적한다.

# Linear weight의 (out,in) 축과 수기 행렬곱 방향을 대조해 전치 누락을 찾습니다.
# batch 차원을 in_features로 사용하지 않고 입력 마지막 축만 layer 계약에 연결합니다.

x shape: torch.Size([2, 3])
weight shape: torch.Size([4, 3])
x 내부 차원: 3
weight 내부 차원: 4
일치: False
correct output shape: (2, 4)

**해설**

PyTorch Linear의 weight는 출력 뉴런별 행을 가지므로 `(out,in)`입니다. 입력 `(B,in)`과 곱하려면 `(in,out)`으로 전치합니다. batch 2는 유지되고 마지막 차원이 4가 됩니다. weight를 외부 포맷으로 내보낼 때 저장 규칙이 다를 수 있으므로 이름만 믿지 말고 실제 shape를 확인해야 합니다.

**오답 점검:** weight를 입력 차원과 출력 차원의 순서로 잘못 외워 그대로 곱하면 내부 차원이 맞지 않습니다. 전치 뒤 출력 shape를 반대로 쓰는 답도 batch가 첫 축에 유지된다는 Linear 계약을 놓친 것입니다.

**적용 범위와 한계:** PyTorch Linear 저장 규칙에는 이 설명이 맞지만 외부 라이브러리나 파일 포맷은 다른 방향으로 저장할 수 있습니다. 모델을 옮길 때는 이름보다 실제 parameter shape와 작은 수동 계산을 함께 확인해야 합니다. bias shape도 출력 차원과 같은지 함께 감사해야 전치 오류와 별도 문제를 구분할 수 있습니다.



## 문제 2. 수동 계산과 모듈 결과 검증

### 업무 요청

고정한 weight·bias로 모듈 출력과 수동 출력이 같은지 확인하는 회귀 테스트를 작성하세요.

### 수행해야 할 작업

1. Linear(3,2)를 만든다.
2. gradient 추적 없이 parameter 값을 복사한다.
3. 모듈과 수동 출력을 각각 계산한다.
4. shape와 근사 동일성을 검증한다.

module output:
tensor([[-1.8000,  2.5000],
        [-1.8000,  0.0000]], grad_fn=<AddmmBackward0>)
manual output:
tensor([[-1.8000,  2.5000],
        [-1.8000,  0.0000]], grad_fn=<AddBackward0>)
same? True

**해설**

# 고정 weight·bias를 no_grad에서 복사해 모듈 출력과 x @ W.T + b를 같은 값으로 비교합니다.
# shape 일치와 allclose를 함께 검사해 우연히 같은 요약값만 맞는 답을 통과시키지 않습니다.

고정 parameter를 복사한 뒤 동일 입력을 두 경로로 계산합니다. `allclose`는 부동소수점의 작은 표현 차이를 허용합니다. `torch.equal`만 사용하면 수학적으로 같은 계산도 미세한 반올림 차이로 실패할 수 있습니다. 이 테스트는 Linear 한 층의 계산 계약을 검증하며 전체 모델의 정확도는 평가하지 않습니다.

**오답 점검:** 수동식에서 bias를 누락하면 weight 곱만 검증하고 Linear 전체 계산을 비교한 것이 아닙니다. 부동소수점 결과를 exact equality만으로 비교하면 계산 순서의 미세한 반올림 차이를 의미 오류로 잘못 판정할 수 있습니다.

**적용 범위와 한계:** 고정 parameter의 두 샘플 테스트는 모듈과 수동식의 로컬 계약을 확인합니다. 다른 dtype과 device에서도 재검증할 수 있지만, 이 일치 결과가 학습된 weight의 품질이나 분류 성능을 말해 주지는 않습니다.


## 문제 3. 출력 head 예산 검토

### 업무 요청

입력 feature 12개인 문서 head 후보 중 parameter 80개 이하에서 class 수 요구를 만족하는 것을 고르세요.

### 수행해야 할 작업

1. 후보별 weight와 bias shape를 적는다.
2. 총 parameter 수를 계산한다.
3. 필요한 class 수 5를 만족하는지 확인한다.
4. 예산과 출력 계약을 모두 통과한 후보를 고른다.

### 상황 자료

```
A=Linear(12,4), B=Linear(12,5), C=Linear(12,7)
required_classes=5, budget=80
```

A weight shape: torch.Size([4, 12])
A bias shape  : torch.Size([4])
B weight shape: torch.Size([5, 12])
B bias shape  : torch.Size([5])
C weight shape: torch.Size([7, 12])
C bias shape  : torch.Size([7])
total params: 52
total params: 65
total params: 91
A: False
B: True
C: False
eligible: ['B']
selected: B

**해설**

A는 예산 안이지만 class logit이 하나 부족합니다. C는 출력 계약과 예산을 모두 벗어납니다. B의 weight `(5,12)` 60개와 bias 5개가 요구를 만족합니다. 출력 7개 중 5개만 쓰는 설계는 불필요한 parameter뿐 아니라 label 의미도 모호하게 만듭니다.

**오답 점검:** 출력 수가 필요한 class보다 많아도 남는 logits를 버리면 된다고 생각하기 쉽지만 label 의미와 loss 계약이 불명확해집니다. parameter를 셀 때 weight만 더하면 B를 실제보다 작게 기록하게 됩니다.

**적용 범위와 한계:** 정확히 class 5개라는 현재 라우터 정책에 따라 B가 선택됩니다. class 집합이 바뀌면 출력 head와 label mapping을 함께 갱신해야 하며, 예산 통과가 latency나 accuracy 기준 통과를 대신하지 않습니다. 구현 뒤에는 named parameters의 실제 합계도 수동 계산 65와 대조합니다.

