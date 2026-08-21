## ** 작성일: 2026_8_19 ** ##

## 딥러닝 기본 과정 실습 중 딥러닝 기본에 관한 문제를 풀염

# [1-2강] 딥러닝 적용 판단 - 실습

필수 1:예시 문제 접근 방식 분류

1.     {'case': '스팸 메일 분류', 'data': '텍스트와 라벨이 충분함', 'pattern': '단어 조합이 다양함'}, 딥러닝. 또는 머신러닝.
    {'case': '세금 계산', 'data': '명확한 공식이 있음', 'pattern': '규칙이 거의 변하지 않음'}, 규칙 기반.
    {'case': '제조 센서 불량 예측', 'data': '수치형 로그와 라벨이 있음', 'pattern': '몇 개 특징으로 설명 가능'}, 머신러닝
    {'case': '이미지 속 부품 결함 탐지', 'data': '이미지와 라벨이 많음', 'pattern': '사람이 특징을 직접 설계하기 어려움'}, 딥러닝.

필수 2: 특징 설계와 표현 학습 비교
 - 직접 만든 텍스트 특징을 계산
 - 사람이 특징을 설계하는 방식의 한계.

reviews = [
    '배송이 빠르고 품질이 좋아요',
    '가격은 괜찮지만 품질이 별로예요',
    '재구매하고 싶을 만큼 만족합니다',
]
positive_words = ['빠르고', '좋아요', '만족', '재구매']

# TODO: 각 문장에 positive_words가 몇 개 포함되어 있는지 세어보세요.
feature_counts = []
for text in reviews:
    count = 0
    count = sum(word in text for word in positive_words)# TODO: 힌트 - sum(word in text for word in positive_words)
    feature_counts.append(count)

print('직접 설계한 긍정 단어 특징:', feature_counts)
print('딥러닝에서는 이런 특징을 사람이 전부 정하지 않고 데이터로부터 표현을 학습합니다.')

심화 1:딥러닝 적용 판단 메모
 - 새로운 비즈니스 문제에 대해 데이터, 모델, 주의점

# 이 세 분류로 비즈니스 판단. 데이터, 모델, 주의점

ex)
데이터 관점: 문장 텍스트와 intent 라벨이 필요합니다.
모델 관점: 문의 표현이 다양하므로 텍스트 표현을 학습하는 딥러닝 또는 임베딩 기반 모델이 유리합니다.
주의점: 라벨 데이터가 적다면 규칙 기반 또는 간단한 머신러닝 baseline부터 비교해야 합니다.

# [1-3강] 데이터-모델-손실-최적화-평가 흐름 - 실습

## 필수 1: 학습 파이프라인 순서 정렬

**문제 설명**
DataLoader부터 평가까지 학습 파이프라인 카드를 올바른 순서로 배열

ordered_cards = ['DataLoader에서 batch 꺼내기', 'model(x) forward', 'loss 계산', 'loss.backward()', 'optimizer.step()', '평가 지표 계산']

## 필수 2: 한 batch 학습 흐름 주석 달기

**문제 설명**
forward, loss, zero_grad, backward, step의 역할을 코드 옆에 주석으로 적습니다.

pred = model(x)              # TODO 모델 forward: 입력을 예측값으로 변환
loss = loss_fn(pred, y)      # TODO loss 계산: 예측값과 정답의 차이를 하나의 값으로 계산.
optimizer.zero_grad()        # TODO gradian zero. : gradient 초기화.
loss.backward()              # TODO 역전파 : parameter gradient 계산.
optimizer.step()             # TODO 주어진 계수에 가중치 최적화. parameter 업데이트.

## 심화 1: train과 validation 구분

**문제 설명**
평가 단계에서 model.eval()과 torch.no_grad()가 들어가는 위치를 확인합니다.

- model.eval()과 with torch.no_grad(): 블록 안에서는 기울기(Gradient) 계산 및 가중치 업데이트를 하지 않으므로, optimizer.zero_grad()와 loss.backward()를 삭제

# [1-4강] 문제 유형별 입출력 매핑 - 실습

## 필수 1: 문제 유형별 출력 형태 매핑

**문제 설명**
회귀/이진/다중 분류 문제의 output dimension과 loss 후보를 매핑합니다.

# TODO: output_dim, loss 후보를 채워보세요.
mapping = {
    '회귀': {'output_dim': 1, 'loss': 'MSELoss'},
    '이진 분류': {'output_dim': 1, 'loss': 'BCEwithLogitsLoss'},
    '다중 분류': {'output_dim': 10, 'loss': 'CrossEntropyLoss'},
}

## 필수 2: 입력 데이터 shape 읽기

**문제 설명**
Tabular, Image, Sequence Tensor의 shape 의미를 문장으로 설명합니다.

    'tabular': '표인데 갯수 5개 특성값 4개씩 있는 2차원 텐서',
    'image': 'batch 5, channel 1이므로 흑백, height 28, width 28인 픽셀의 2차원 공간구조?',
    'sequence': 'batch =5, time =12, feature=8의 시퀀스 데이터'

## 심화 1: logits와 target으로 loss 판단

**문제 설명**
logits와 target 형태를 보고 적절한 손실 함수를 선택합니다.

# 심화 1. logits와 target을 보고 손실 함수 후보 판단하기

logits = torch.randn(6, 3)      # 3-class logits
target = torch.tensor([0, 2, 1, 1, 0, 2])

loss_fn = nn.CrossEntropyLoss()

# # [1장 5강] - 학습 코드 구조 해석 실습

## 필수 1 : **코드 블록 단계명 붙이기**

code_blocks = [
    'import torch, torch.nn as nn', 'import module'
    'train_loader = ...', 'dataset'
    'model = nn.Linear(4, 3)', 'model'
    'loss_fn = nn.CrossEntropyLoss()', 'loss'
    'optimizer = torch.optim.Adam(model.parameters())', 'optimizer'
    'for epoch in range(epochs): ...', 'loop'

## 필수 2 : **작은 학습 코드 구조 읽기**

# TODO: 아래 loop에서 forward/loss/backward/update 위치를 주석으로 표시해보세요.
logits = model(x) #forward
loss = loss_fn(logits, y) #loss 계산
optimizer.zero_grad() # gradient 초기화
loss.backward() #backward: gradient 계산
optimizer.step() # parameter update


## 심화 1 : **metric과 checkpoint 위치 표시**

history = {'loss': []}
checkpoint = {}

# TODO: loss 기록과 checkpoint 저장이 어느 시점에 들어가면 좋을지 생각해보세요.
history['loss'].append(float(loss.item()))
checkpoint['model_state_dict'] = model.state_dict()
print('history:', history) #loss 기록
print('checkpoint keys:', list(checkpoint.keys())) # checkpoint 저장
assert 'model_state_dict' in checkpoint

학습 루프(for epoch in range(...)) 내부에서
- history['loss'].append(...): 매 Epoch(또는 Batch)가 끝나는 시점에 오차 기록용으로 삽입.
- checkpoint['model_state_dict'] = ...: 보통 학습이 다 끝난 후(루프 밖)나, 성능이 이전보다 좋아졌을 때(최고 성능 갱신 시) 저장용으로 삽입.

# [2-1강] Tensor 생성과 dtype/shape 확인 - 실습 (학생용)

## 필수 1: Tensor 속성 확인
Tensor를 만들고 shape, dtype, ndim을 출력합니다.

# TODO: 리스트를 Tensor로 바꾸고 shape/dtype/ndim을 출력해보세요.
values = [[1, 2, 3], [4, 5, 6]]
tensor = torch.tensor(values)

print('tensor:', tensor)
print('shape:', tensor.shape)
print('dtype:', tensor.dtype)
print('ndim:', tensor.ndim)

tensor: tensor([[1, 2, 3],
        [4, 5, 6]])
shape: torch.Size([2, 3])
dtype: torch.int64
ndim: 2

## 필수 2: label dtype 구분

**문제 설명**
회귀 target과 분류 target의 dtype 차이를 확인합니다.

# 필수 2. label dtype 구분하기
regression_target = torch.tensor([1.2, 3.4, 5.6])
classification_target = torch.tensor([0, 2, 1])

# TODO: 회귀 target과 분류 target의 dtype이 왜 다른지 메모해보세요.
print('regression target dtype:', regression_target.dtype)
print('classification target dtype:', classification_target.dtype)

regression target dtype: torch.float32
classification target dtype: torch.int64

 - 회귀 target은 float32로 실수형. 회귀는 집값, 온도, 키처럼 연속적인 수치(소수점이 포함된 실수)를 예측하는 문제이기 때문에, 미세한 오차 계산 및 손실 함수(MSE 등) 연산을 위해 소수점을 표현할 수 있는 float32를 사용.
 - 분류 target은 int64로 (정수형/Long). 분류는 '0번 클래스(개)', '1번 클래스(고양이)', '2번 클래스(새)'처럼 각 데이터가 속한 범주(Class)의 인덱스 번호를 나타내기 때문에, 소수점이 없는 라벨 번호 형태인 int64를 사용.


## 심화 1: loss dtype 오류 예방
**문제 설명**
CrossEntropyLoss에 맞게 target dtype을 수정

# 심화 1. 잘못된 dtype으로 인한 loss 오류 예방하기
logits = torch.randn(3, 3)
wrong_target = torch.tensor([0.0, 1.0, 2.0])  # float입니다.

# TODO: CrossEntropyLoss에 맞게 target dtype을 고쳐보세요.
fixed_target = wrong_target.long()  # 힌트: wrong_target.long()
print('fixed_target dtype:', fixed_target.dtype)

fixed_target dtype: torch.int64

  - CrossEntropyLoss의 핵심 규칙:
    logits:모델의 출력 (3개 데이터 $\times$ 3개 클래스 점수 $\rightarrow$ float32)
    target: 각 데이터가 몇 번 클래스에 속하는지 나타내는 인덱스 번호 ($\rightarrow$  반드시 torch.int64 / long 타입이어야 함)
    [0.0, 1.0, 2.0] 처럼 소수점이 붙은 float 상태로 nn.CrossEntropyLoss()에 넣으면 RuntimeError: Expected object of scalar type Long... 에러 발생함.따라서 .long() (또는 .to(torch.long))을 붙여서 int64 정수형 텐서로 변환해 주는 것이 핵심임.

# [2-2강] Batch dimension과 broadcasting - 실습

# 필수 1. batch dimension 추가/제거하기
single_image = torch.randn(1, 28, 28)  # C,H,W

# TODO: 모델에 넣기 위해 batch 차원을 맨 앞에 추가해보세요.
batched_image = single_image.unsqueeze(0)  # 힌트: unsqueeze(0)
print('single_image shape:', single_image.shape)
print('batched_image shape:', batched_image.shape)

single_image shape: torch.Size([1, 28, 28])
batched_image shape: torch.Size([1, 1, 28, 28])

  PyTorch의 모든 신경망 모델(CNN, MLP 등)은 이미지 1장을 넣더라도 "반드시 맨 앞에 배치(Batch) 차원이 붙어 있는 4차원 형태 (B, C, H, W)"의 입력만 받도록 설계되어 있습니다.
  single_image: (1, 28, 28) $\rightarrow$ 3차원 (채널 1개, 세로 28, 가로 28)
  batched_image: (1, 1, 28, 28) $\rightarrow$ 4차원 (배치 1개, 채널 1개, 세로 28, 가로 28)0번째 인덱스 위치에 크기가 1인 차원을 새로 뚫어주기 때문에 unsqueeze(0)을 사용함.

## 필수 2: broadcasting 결과 예측

**문제 설명**
서로 다른 shape의 Tensor 연산 결과 shape을 예측

# 필수 2. broadcasting 결과 shape 예측하기
a = torch.randn(4, 3)
b = torch.randn(3)
result = a + b
# TODO: result.shape이 왜 이렇게 나오는지 적어보세요.
print('a shape:', a.shape)
print('b shape:', b.shape)
print('result shape:', result.shape)

a shape: torch.Size([4, 3])
b shape: torch.Size([3])
result shape: torch.Size([4, 3])

## 심화 1: 위험한 broadcasting 수정

**문제 설명**
pred와 target shape을 맞춰 의도치 않은 broadcasting을 예방
pred = torch.randn(4, 1)
target = torch.randn(4)  # shape이 다릅니다.
print('pred shape:', pred.shape, '| target shape:', target.shape)
# TODO: target을 pred와 같은 shape으로 바꿔보세요.

fixed_target = target.unsqueeze(1)
print('fixed_target shape:', fixed_target.shape)

pred shape: torch.Size([4, 1]) | target shape: torch.Size([4])
fixed_target shape: torch.Size([4, 1])

- nsqueeze는 단어 그대로 "쥐어짜다(squeeze)의 반대(un-)"라는 뜻.
- 차원을 쥐어짜서 누르는 squeeze()와 반대로, 차원을 '펼쳐서 1차원을 뚫어준다'는 의미로 기억.

## **## ** 작성일: 2026_8_20 ** ##

# [2-3강] CPU/GPU device와 .to(device) - 실습

## 필수 1: device 확인과 Tensor 이동

**문제 설명**
cuda 사용 가능 여부를 확인하고 Tensor를 device로 이동합니다.

# Tensor의 .to()는 이동된 Tensor를 반환하므로, 반환값을 다시 x에 저장
# 참고: nn.Module.to(device)는 모듈의 파라미터와 버퍼를 이동시키고 모듈 자신을 반환
##  둘 다 x = x.to(device), model = model.to(device)처럼 작성해도 좋음.

x_device = x.to(device)


```bash
    선택된 device: cuda
    x_device.device: cuda:0
```

## 필수 2: 모델과 입력 같은 device 맞추기

**문제 설명**
model과 입력 Tensor가 같은 device에 있도록 구성합니다.

# TODO: model과 x를 같은 device로 이동해보세요.
model = model.to(device)
x = x.to(device)
out = model(x)
print('out shape:', out.shape)

## 심화 1: batch 이동 helper 작성

**문제 설명**
입력과 라벨을 한 번에 device로 옮기는 함수를 작성합니다.

# 심화 1. 안전한 device 이동 함수 만들기

```bash
def move_batch_to_device(batch, device):
    x, y = batch
    # TODO: x와 y를 모두 device로 이동해서 반환하세요.
    return x.to(device), y.to(device)

batch = (torch.randn(2, 4), torch.tensor([0, 1]))
x_moved, y_moved = move_batch_to_device(batch, device)
print(x_moved.device, y_moved.device)
assert x_moved.device == device and y_moved.device == device
```

```bash
cuda:0 cuda:0
```
# [2-4강] Shape/Device 오류 디버깅 - 실습

## 필수 1: Linear 입력 차원 오류 수정

**문제 설명**
입력 feature 수에 맞게 nn.Linear의 in_features를 수정합니다.
# 필수 1. Linear 입력 차원 오류 수정하기

x = torch.randn(8, 5)
# TODO: x의 feature 수에 맞게 in_features를 수정해보세요.
model = nn.Linear(5, 2)
print('x shape:', x.shape)
print('model expects in_features:', model.in_features)

```bash
x shape: torch.Size([8, 5])
model expects in_features: 5
```

## 필수 2: batch 차원 누락 수정

**문제 설명**
단일 sample에 batch 차원을 추가합니다.

# 필수 2. batch 차원 누락 수정하기
single_sample = torch.randn(5)
# TODO: batch 차원을 추가해 model 입력으로 만들세요.
batch_sample = single_sample.unsqueeze(0)
out = model(batch_sample)
print('single shape:', single_sample.shape)
print('batch shape:', batch_sample.shape)

print('out shape:', out.shape)
assert batch_sample.shape == (1,5)

```bash
single shape: torch.Size([5])
batch shape: torch.Size([1, 5])
out shape: torch.Size([1, 2])
```

## 심화 1: 디버깅 체크리스트 작성

**문제 설명**
shape/device 오류를 만났을 때 확인할 항목을 정리합니다.

# 심화 1. 디버깅 체크리스트 만들기
checklist = [
    '입력 x.shape가 모델 in_features와 맞는지 확인합니다.',
    'target shape과 dtype이 loss 함수 요구사항에 맞는지 확인합니다.',
    'model과 Tensor가 같은 device에 있는지 확인합니다.',
]
# TODO: shape/device 오류를 만났을 때 확인할 항목 3개를 적어보세요.
for i, item in enumerate(checklist, 1):
    print(i, item)
if not checklist:
    print('TODO: x.shape, y.shape, model device 등을 체크리스트에 넣어보세요.')
assert len(checklist) == 3

# [3-1강] 퍼셉트론과 선형 결정 경계 - 실습

## 필수 1: 퍼셉트론 수식 계산

**문제 설명**
z=x@w+b를 직접 계산합니다.


# 필수 1. 퍼셉트론 수식 계산하기
x = torch.tensor([2.0, -1.0])
w = torch.tensor([0.5, 1.0])
b = torch.tensor(0.2)
# TODO: z = x @ w + b 를 계산해보세요.

z = x @ w + b
print('z:', z.item())

```bash
z: 0.20000000
```

## 필수 2: 선형 결정 경계 예측

**문제 설명**
여러 point에 대해 z>0 기준으로 class를 예측합니다.

```bash
# 필수 2. 선형 결정 경계 시각화 준비
points = torch.tensor([[1., 1.], [2., 1.], [-1., -1.], [-2., -1.]])
# TODO: 각 point에 대해 z를 계산하고 z > 0이면 1, 아니면 0으로 예측해보세요.
scores = torch.zeros(len(points))

w = torch.tensor([1., 1.])
b = 0.
scores = points @ w + b

pred = (scores > 0).long()

print('scores:', scores.tolist())
print('pred:', pred.tolist())
```

결과

```bash
scores: [2.0, 3.0, -2.0, -3.0]
pred: [1, 1, 0, 0]
```

## 심화 1: bias 변화 관찰

**문제 설명**
bias 값 변화가 예측 결과를 어떻게 바꾸는지 확인합니다.

# 심화 1. bias를 바꾸면 결정 경계가 어떻게 이동하는지 관찰하기
bias_values = [-1.0, 0.0, 1.0]
for bias in bias_values:
    score = points @ w + bias
    print('bias=', bias, 'pred=', (score > 0).long().tolist())

bias= -1.0 pred= [1, 1, 0, 0]
bias= 0.0 pred= [1, 1, 0, 0]
bias= 1.0 pred= [1, 1, 0, 0]


# 심화 1. bias를 바꾸면 결정 경계가 어떻게 이동하는지 관찰하기

bias가 바뀌면 결정 경계는 항상 이동한다.
다만 현재 데이터의 위치 때문에 prediction이 안 바뀔 수도 있다.

bias_values = [-3.0, 0.0, 3.0]
for bias in bias_values:
    score = points @ w + bias
    print('bias=', bias, 'pred=', (score > 0).long().tolist())

z: tensor(0.2000)
scores: [2.0, 3.0, -2.0, -3.0]
pred: [1, 1, 0, 0]
bias= -3.0 pred= [0, 0, 0, 0]
bias= 0.0 pred= [1, 1, 0, 0]
bias= 3.0 pred= [1, 1, 1, 0]


# [3-2강] MLP의 입력층/은닉층/출력층 - 실습

## 필수 1: MLP 층별 shape 추적

**문제 설명**
입력, hidden, logits의 shape을 확인합니다.

hidden = torch.relu(layer1(x))
## 이 Linear → ReLU 조합이 MLP의 기본적인 한 층을 구성

```bash
x: torch.Size([4, 6]) hidden: torch.Size([4, 3]) logits: torch.Size([4, 2])
```

## 필수 2: hidden size와 parameter 수

**문제 설명**
hidden size 변경 시 parameter 수가 어떻게 달라지는지 계산합니다.

# 필수 2. hidden size를 바꾸면 parameter 수가 어떻게 달라질까요?
def count_params(model):
    return sum(p.numel() for p in model.parameters()) 
## PyTorch 실제 파라미터 수 

# TODO: hidden_dim=4인 MLP를 만들어 parameter 수를 출력해보세요.
model = nn.Sequential(nn.Linear(6, 3), nn.ReLU(), nn.Linear(3, 2))
print('parameter count(before):', count_params(model))

model = nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, 2))
print('parameter count(after):', count_params(model))

'''bash
parameter count(before): 29
parameter count(after): 38
'''


## 심화 1: depth와 width 비교

**문제 설명**
은닉층 개수에 따른 parameter 수 차이를 비교합니다.

layer1 = nn.Linear(input_dim, hidden_dim) # (6,3)
layer2 = nn.Linear(hidden_dim, output_dim) # (3,2)

# TODO: hidden, logits를 계산해보세요.
hidden = layer1(x)
logits = layer2(hidden)

model_a = nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, 2))
model_b = nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
print('model_a params:', count_params(model_a))
print('model_b params:', count_params(model_b))

```bash
x: torch.Size([4, 6]) hidden: torch.Size([4, 3]) logits: torch.Size([4, 2])
parameter count(before): 29
parameter count(after): 38
model_a params: 38
model_b params: 58
```

# [3-3강] 가중치/편향과 nn.Linear - 실습

## 필수 1: nn.Linear weight/bias shape

**문제 설명**
Linear layer의 weight, bias shape과 parameter 수를 계산합니다.

# 필수 1. nn.Linear의 weight/bias shape 확인
linear = nn.Linear(in_features=5, out_features=3)
print('weight shape:', linear.weight.shape)
print('bias shape:', linear.bias.shape)

# TODO: parameter 수를 직접 계산해보세요.
manual_param_count = linear.weight.numel() + linear.bias.numel()

#  X.shape(3,5) @ weight.T shape(5,3) + bias shape (,3)

print('manual_param_count:', manual_param_count)

weight shape: torch.Size([3, 5])
bias shape: torch.Size([3])
manual_param_count: 18

## 필수 2: named_parameters 확인

**문제 설명**
parameter 이름, shape, 개수를 출력합니다.

for name, param in linear.named_parameters():
    print(name, param.shape, param.numel())

weight torch.Size([3, 5]) 15
bias torch.Size([3]) 3

## model.parameters() - parameter는 파라미터 값만, 이것은 (이름, 파라미터) 쌍.
## named_parameters()
## 모델 학습 자체에는 이름이 없어도 된다. 그러나 사람이 모델을 관리·분석·디버깅하려면 이름이 매우 유용함.

 0.weight torch.Size([3, 5])    0.weight → 첫 번째 Linear의 weight
 0.bias   torch.Size([3])       0.bias   → 첫 번째 Linear의 bias
 2.weight torch.Size([2, 3])    2.weight → 두 번째 Linear의 weight
 2.bias   torch.Size([2])       2.bias   → 두 번째 Linear의 bias

# 유용성
  예를 들어 모델의 특정 층만 보고 싶을 때 또는 모델 저장/불러오기(`model.state_dict()`)

for name, param in model.named_parameters():
    if "0.weight" in name:
        print(param)

## 심화 1: 수식과 nn.Linear 비교

**문제 설명**
x@W.T+b 계산이 nn.Linear와 같은 결과인지 확인합니다.

# 심화 1. 직접 계산한 Linear와 nn.Linear 비교하기
x = torch.randn(2, 5)
# TODO: y_manual = x @ W.T + b 형태를 완성해보세요.
y_layer = linear(x)
y_manual = x @ linear.weight.T + linear.bias
print('max difference:', (y_layer - y_manual).abs().max().item())

max difference: 0.0

# [3-4강] 입출력 차원 계산과 flatten - 실습

## 필수 1: 이미지 Tensor flatten

**문제 설명**
이미지 Tensor를 batch 차원을 유지한 채 flatten합니다.

# 필수 1. 이미지 Tensor flatten하기
images = torch.randn(8, 1, 28, 28)
# TODO: batch 차원을 유지하면서 flatten해보세요.
flat = torch.flatten(images, start_dim=1)
print('images shape:', images.shape)
print('flat shape:', flat.shape)

images shape: torch.Size([8, 1, 28, 28])
flat shape: torch.Size([8, 784])

## 필수 2: MLP 입력 차원 맞추기

**문제 설명**
flatten 결과에 맞게 Linear 입력 차원을 설정합니다.

# 필수 2. MLP 입력 차원 맞추기
# TODO: flat의 feature 수에 맞게 in_features를 설정하세요.
in_features = 1*28*28
model = nn.Linear(in_features, 4)
print('model.in_features:', model.in_features)

model.in_features: 784

## 심화 1: reshape/view 연습

**문제 설명**
3차원 Tensor를 batch-first 2차원 Tensor로 변환합니다.

# 심화 1. reshape/view 사용 시 batch 차원을 유지하기
small = torch.randn(2, 3, 4)
# TODO: small을 (2, 12)로 바꿔보세요.
small_flat = small.view(small.size(0), -1)
print('small_flat shape:', small_flat.shape)

small_flat2 = small.reshape(small.size(0), -1)
print('small_flat2 shape:', small_flat.shape)


small_flat shape: torch.Size([2, 12])
small_flat2 shape: torch.Size([2, 12])


2026_8_21

# [4-1강] 비선형성과 활성화 함수 필요성 - 실습

## 필수 1: ReLU 전후 출력 비교

**문제 설명**
선형 출력과 ReLU 적용 후 출력 차이를 확인합니다.

x: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
linear_only: [-1.4639999866485596, -0.6990000009536743, 0.06499999761581421, 0.8299999833106995, 1.5950000286102295, 2.3589999675750732, 3.124000072479248]
with_relu: [0.0, 0.0, 0.06499999761581421, 0.8299999833106995, 1.5950000286102295, 2.3589999675750732, 3.124000072479248]

## 필수 2: XOR 데이터 관찰

**문제 설명**
선형 경계로 어려운 XOR 구조를 데이터로 확인합니다.

XOR data:
[0.0, 0.0] -> 0
[0.0, 1.0] -> 1
[1.0, 0.0] -> 1
[1.0, 1.0] -> 0

## 심화 1: 비선형 feature 추가

**문제 설명**
x1*x2 같은 비선형 feature가 표현력을 높이는 이유를 관찰합니다.

expanded features: tensor([[0., 0., 0.],
        [0., 1., 0.],
        [1., 0., 0.],
        [1., 1., 1.]])

3차원 feature 공간에서는 XOR을 선형식으로 표현할 수 있게 되는 것.
feature space.
그리고 이게 중요한 이유는:
비선형 feature를 추가하면 원래 선형 분리 불가능했던 문제를 새로운 공간에서는 선형적으로 표현할 수 있다.

# [4-2강] ReLU의 역할과 사용 위치 - 실습 (학생용)

## 필수 1: ReLU 값 비교

**문제 설명**
음수, 0, 양수 입력에 ReLU를 적용합니다.

relu_values = torch.relu(values)

before: [-2.0, -0.5, 0.0, 1.0, 3.0]
after: [0.0, 0.0, 0.0, 1.0, 3.0]

## 필수 2: MLP에 ReLU 추가

**문제 설명**
은닉층 Linear 뒤에 ReLU를 연결합니다.

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(), # TODO: 여기에 ReLU를 추가해보세요.
    nn.Linear(8, 3)
)

## 심화 1: dead ReLU 직관 확인

**문제 설명**
음수 입력이 0으로 잘리는 현상을 확인합니다.

# 심화 1. dead ReLU 직관 확인하기
negative_input = torch.tensor([-5.0, -1.0, 0.5])
print('ReLU output:', torch.relu(negative_input).tolist())
print('음수 입력은 0으로 잘리므로 일부 뉴런이 계속 0만 출력할 수 있습니다.')

ReLU output: [0.0, 0.0, 0.5]
음수 입력은 0으로 잘리므로 일부 뉴런이 계속 0만 출력할 수 있습니다.


# [4-3강] Sigmoid와 이진 분류 출력층 - 실습 (학생용)

## 필수 1: Sigmoid 확률 변환

**문제 설명**
binary logits를 0~1 확률로 변환합니다.

# 필수 1. logits를 Sigmoid 확률로 바꾸기
logits = torch.tensor([-2.0, 0.0, 2.0])
# TODO: sigmoid를 적용해보세요.
probs = torch.sigmoid(logits)
print('logits:', logits.tolist())
print('probs:', probs.tolist())

logits: [-2.0, 0.0, 2.0]
probs: [0.11920291930437088, 0.5, 0.8807970285415649]

## 필수 2: threshold 기반 예측

**문제 설명**
확률 0.5 기준으로 이진 label을 만듭니다.

probs: [0.23100000619888306, 0.5989999771118164, 0.890999972820282, 0.3319999873638153]
pred_label: [0, 0, 0, 0]
preds: tensor([0, 1, 1, 0])


## 심화 1: BCEWithLogitsLoss 연결

**문제 설명**
Sigmoid 전 logits와 float target으로 loss를 계산합니다.

loss_fn = nn.BCEWithLogitsLoss()
if loss_fn is None:
    print('TODO: nn.BCEWithLogitsLoss()를 사용해보세요.')
else:
    print(loss_fn(logits, target).item())

0.3709379732608795

# [4-4강] Softmax와 다중 분류 출력층 - 실습 (학생용)

## 필수 1: Softmax 확률 변환

**문제 설명**
다중 분류 logits를 class 확률 분포로 변환합니다.
probs = torch.softmax(logits, dim=1)

probs: tensor([[0.6590, 0.2424, 0.0986]])
sum: tensor(1.0000)

## 필수 2: argmax 예측

**문제 설명**
logits에서 가장 큰 class index를 예측합니다.

# 필수 2. argmax로 예측 class 만들기
logits = torch.tensor([[0.1, 2.5, 0.3], [3.0, 1.0, 0.2]])
# TODO: 각 sample의 예측 class index를 구하세요.
pred = torch.zeros(2, dtype=torch.long)
pred = torch.argmax(logits, dim=1) 
print('pred:', pred.tolist())

pred: [1, 0]

## 심화 1: CrossEntropyLoss 연결

**문제 설명**
raw logits와 class index target으로 loss를 계산합니다.

# 심화 1. CrossEntropyLoss는 raw logits와 class index target을 받습니다.
logits = torch.randn(4, 3)
target = torch.tensor([0, 2, 1, 1])
# TODO: loss를 계산해보세요.
loss_fn = nn.CrossEntropyLoss()
if loss_fn is None:
    print('TODO: nn.CrossEntropyLoss()를 사용해보세요.')
else:
    print(loss_fn(logits, target).item())

1.5617684125900269
