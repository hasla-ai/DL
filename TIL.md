2026_8_19

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




