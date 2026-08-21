# [4장 1강 심화] - 비선형성 확인 실습 (1)
print("[4장 1강 심화] - 비선형성 확인 실습")

# 문제 1. 세 층이라는 설명의 허점 진단
print("문제 1. 세 층이라는 설명의 허점 진단")
# Linear 두 층의 연속 계산과 합친 한 층의 계산이 같은지 확인해, 왜 활성화 없는 깊이가 표현력을 늘리지 않는지 보고하세요.
## 두 output shape, same, 구조 진단 문장을 제출
import torch
x = torch.tensor([[1., 2.], [-1., 0.5]]) # shape : (2,2)
W1 = torch.tensor([[1., 2.], [0., -1.]]) # shape : (2,2)
b1 = torch.tensor([0.5, -0.5])           # shape : (1,2) 
W2 = torch.tensor([[2., -1.]])           # shape : (1,2)
b2 = torch.tensor([0.25])

# 1. 두 층 출력을 계산한다.
y1 = x @ W1 + b1
W2 = W2.squeeze(0)
y2 = x @ W2 + b2
print(y1,y2)

# 2. 합성 weight와 bias를 계산한다.
W = W1 @ W2
b = b1 @ W2 + b2
z1 = x @ W + b
print(z1)

# 3. 한 번의 계산과 근사 비교한다.
z2 = y1@W2 +b2
print(z2)
print("근사 비교:", torch.allclose(z1, z2))
# 4. 출력 shape만으로 구조를 평가하면 안 되는 이유를 쓴다.

# 문제 2. XOR을 구분하는 ReLU hidden 코드 작성
print("문제 2. XOR을 구분하는 ReLU hidden 코드 작성")
# 두 입력이 서로 다를 때만 활성화되는 hidden feature 두 개를 만들고 XOR label을 출력.
# 이 문제는 학습이 아니라 주어진 weight로 표현 가능성을 확인.

# h1 = ReLU(x1 - x2) → weight [1, -1]
# h2 = ReLU(x2 - x1) → weight [-1, 1]
# bias는 둘 다 0으로 둘 수 있습니다.

# 1. 네 XOR 입력을 batch로 만든다.
X_xor = torch.tensor(
    [[0.0, 0.0],
     [0.0, 1.0],
     [1.0, 0.0],
     [1.0, 1.0]],
    dtype=torch.float32
)
y_xor = torch.tensor([0, 1, 1, 0], dtype=torch.long)
print("XOR 입력:")
print(X_xor)
print("XOR 라벨:")
print(y_xor)
for x, y in zip(X_xor, y_xor):
    print(x.tolist(), " XOR 정답 →", y.item())

# 2. 두 차이 방향의 Linear 결과에 ReLU를 적용한다.
# Linear 1

# h1 = ReLU(x1 - x2) → weight [1, -1]
# h2 = ReLU(x2 - x1) → weight [-1, 1]
# bias는 둘 다 0으로 둘 수 있습니다.

W1 = torch.tensor([
    [1., -1.],   # h1: x1 - x2
    [-1., 1.]    # h2: x2 - x1
])
b1 = torch.tensor([0., 0.]) # bias

h = X_xor @ W1.T + b1   # Linear
print("Linear 결과:")
print(h)
h_relu = torch.relu(h)
print("ReLU 결과:")
print(h_relu) # 결과의 두 열이 각각 두 은닉노드의 출력

# 4. 네 정답과 정확히 같은지 확인한다.


# 문제 3. 알려진 XOR 정답과 별도 검증을 구분해 후보 선택
print("문제 3. 알려진 XOR 정답과 별도 검증을 구분해 후보 선택")
# 문제 2의 XOR 네 점은 구조가 패턴을 표현할 수 있는지 확인한 단위 테스트.
# 운영 후보 세 개에 대해 별도 noisy validation, parameter, latency 결과 추가.
# 단위 테스트 통과와 운영 검증을 구분해 실험안 선택하기.

#  후보  구조          XOR acc  noisy valid acc  params  latency
# L     Linear-only     0.75        0.72           3      0.2ms
# R     ReLU hidden     1.00        0.86           9      0.4ms
# W     ReLU wide       1.00        0.91          41      1.2ms

import pandas as pd

results = pd.DataFrame({
    "후보": ["L", "R", "W"],
    "구조": ["Linear-only", "ReLU hidden", "ReLU wide"],
    "XOR acc": [0.75, 1.00, 1.00],
    "noisy valid acc": [0.72, 0.86, 0.91],
    "params": [3, 9, 41],
    "latency": [0.2, 0.4, 1.2]
})
results["Efficiency test"] = (
    (results["params"] <= 20) &
    (results["latency"] <= 0.8)
)

# 후보별 계약 판정, selected, 아직 배포 승인할 수 없는 이유를 제출

# 1. XOR 단위 테스트 통과 여부를 확인한다.
results["XOR test"] = results["XOR acc"] == 1.00
print(results[["후보", "구조", "XOR acc", "XOR test"]])

passed = results[results["XOR test"]]

print("=== XOR TEST RESULT ===")
print(f"통과 후보: {passed['후보'].tolist()}")
print(f"통과 수: {len(passed)} / {len(results)}")

# 2. noisy validation accuracy 0.85 이상을 적용한다.
results["Noisy validation test"] = results["noisy valid acc"] >= 0.85
print(
    results[
        ["후보", "구조", "noisy valid acc", "Noisy validation test"]
    ]
)
passed = results[results["Noisy validation test"]]

print("=== NOISY VALIDATION TEST RESULT ===")
print(f"통과 후보: {passed['후보'].tolist()}")
print(f"통과 수: {len(passed)} / {len(results)}")

# 3. parameter 20 이하, latency 0.8ms 이하를 함께 적용한다.
results["Efficiency test"] = (
    (results["params"] <= 20) &
    (results["latency"] <= 0.8)
)
print(
    results[
        ["후보", "구조", "params", "latency", "Efficiency test"]
    ]
)
passed = results[results["Efficiency test"]]

print("=== EFFICIENCY TEST RESULT ===")
print(f"통과 후보: {passed['후보'].tolist()}")
print(f"통과 수: {len(passed)} / {len(results)}")

# 4. 통과 후보를 다음 실험안으로 선택하고 배포 승인과 구분한다.
# 모든 검증을 통과한 후보
results["All tests"] = (
    (results["XOR test"]) &
    (results["Noisy validation test"]) &
    (results["Efficiency test"])
)

# 다음 실험 후보
next_candidates = results[results["All tests"]]
print("=== NEXT EXPERIMENT CANDIDATES ===")
print(next_candidates[["후보", "구조"]])
print("\n=== DEPLOYMENT APPROVAL ===")
print("배포 승인: 별도 절차 필요")




# [4장 2강 심화] - ReLU 적용 실습
print("[4장 2강 심화] - ReLU 적용 실습")

# 문제 1. shape 정상인 0 출력 진단
print("문제 1. shape 정상인 0 출력 진단")
# 두 뉴런의 ReLU 출력과 활성 비율을 계산하고 어느 뉴런을 우선 점검할지 보고
# pre, activation, active_ratio, 점검 뉴런을 제출


# 1. Linear pre-activation을 계산한다.
neuron0=[-3,-4] # pre_activation
neuron1=[0.5,1.5] # pre_activation
print("neuron0 pre activation", neuron0)
print("neuron1 pre activation", neuron1)
# 2. ReLU를 적용한다.
a0 = torch.relu(torch.tensor(neuron0))
a1 = torch.relu(torch.tensor(neuron1))
print("neuron0 activation", a0)
print("neuron1 activation", a1)
# 3. 뉴런별 양수 비율을 계산한다.
positive_ratio_0 = (a0 > 0).float().mean()
positive_ratio_1 = (a1 > 0).float().mean()
print("neuron0:", positive_ratio_0.item() * 100, "%")
print("neuron1:", positive_ratio_1.item() * 100, "%")
# 4. 한 batch 결과의 해석 한계를 적는다.


# 문제 2. 중간 활성 감사를 포함한 MLP 구현
print("문제 2. 중간 활성 감사를 포함한 MLP 구현")
# feature 2개를 hidden 2개, class 2개로 바꾸는 모델을 작성하고 forward에서 logits와 hidden 활성값을 반환
# 모델 코드, 두 shape, 전체 active ratio를 제출

# 1. Linear 두 개와 ReLU를 등록한다.
import torch
import torch.nn as nn

class DebugActivationMLP(nn.Module):
    """
    forward 과정에서 중간 shape와 값 범위를 출력하는 디버깅용 MLP입니다.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2, 2)

    def forward(self, x):
        print("input shape:", x.shape)
        print("input min/max:", x.min().item(), x.max().item())

        x = self.fc1(x)
        print("after fc1 shape:", x.shape)
        print("after fc1 min/max:", x.min().item(), x.max().item())

        x = self.relu1(x)
        print("after relu shape:", x.shape)
        print("after relu min/max:", x.min().item(), x.max().item())

        logits = self.fc2(x)
        print("logits shape:", logits.shape)
        return logits

X = torch.randn(4, 2)
model = DebugActivationMLP()
logits = model(X)
print(model)
# 2. ReLU를 첫 Linear 뒤에만 둔다.

# 3. hidden과 logits shape를 assert한다.
assert model.relu1(X).shape == model.fc2(X).shape

# 4. 활성 비율을 출력한다.
positive_ratio_0 = (model.fc2(X) > 0).float().mean()
print("positive_ratio:", positive_ratio_0.item() * 100, "%")

# 문제 3. 활성 분포 경보와 탈락 조건 구분
print("문제 3. 활성 분포 경보와 탈락 조건 구분")
## 같은 검증 batch에서 세 설정의 hidden 양수 비율이 주어졌습니다.
### 여러 대표 batch에서 ratio 0이 반복되면 hard block, `0.25~0.75` 밖이지만 0은 아니면 재점검 경보로 둡니다.
### validation accuracy 0.80 이상과 함께 `즉시 다음 실험`, `추가 점검`, `차단`으로 분류.
#### ready, investigate, blocked와 다음 실험안을 제출

# A ratio=0.00 acc=0.82 zero_ratio_batches=5
# B ratio=0.48 acc=0.84 zero_ratio_batches=0
# C ratio=0.92 acc=0.86 zero_ratio_batches=0

results = pd.DataFrame({
    "후보": ["A", "B", "C"],
    "ratio": [0.00, 0.48, 0.92],
    "acc": [0.82, 0.84, 0.86],
    "zero_ratio_batches": [5, 0, 0]
})
# 1. 반복된 ratio 0 증거가 있는 후보를 차단한다.
results["blocked"] = results["zero_ratio_batches"] > 0

# 2. accuracy 0.80 미만을 제외한다.
results["accuracy_pass"] = results["acc"] >= 0.80

# 3. 임시 비율 구간 밖의 후보는 탈락이 아니라 추가 점검으로 보낸다.
ratio_min = 0.30
ratio_max = 0.80

results["ratio_check"] = (
    (results["ratio"] >= ratio_min) &
    (results["ratio"] <= ratio_max)
)

results["status"] = "추가 점검"

results.loc[
    results["blocked"],
    "status"
] = "차단"

results.loc[
    (~results["blocked"]) &
    (~results["accuracy_pass"]),
    "status"
] = "탈락"


results.loc[
    (~results["blocked"]) &
    (results["accuracy_pass"]) &
    (~results["ratio_check"]),
    "status"
] = "추가 점검"

results.loc[
    (~results["blocked"]) &
    (results["accuracy_pass"]) &
    (results["ratio_check"]),
    "status"
] = "다음 실험"

# 4. 즉시 다음 실험안과 C를 재확인할 자료를 보고한다.
print(
    results[
        ["후보", "ratio", "acc", "zero_ratio_batches", "status"]
    ]
)

## [4장 3강 심화] - 이진 분류 출력 실습 (1)
print("[4장 3강 심화] - 이진 분류 출력 실습")

logits = torch.tensor([[0.0], [2.0]])
target = torch.tensor([[0.0],[1.0]])
# broken: loss_fn(sigmoid(logits), target)

## 문제 1. 이중 Sigmoid의 silent bug 진단
print("문제 1. 이중 Sigmoid의 silent bug 진단")
# correct/wrong loss와 수정 원칙을 제출

# 1. logits/target shape와 dtype을 확인한다.
print("logits shape:", logits.shape)
print("logits dtype:", logits.dtype)
print("target shape:", target.shape)
print("target dtype:", target.dtype)

# 2. raw logits로 loss를 계산한다.
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, target)
print("loss:", loss)						# 학습 loss 계산

# 3. Sigmoid 선적용 loss를 계산한다.
probs = torch.sigmoid(logits)
loss_sigmoid = criterion(probs, target)
print("loss_sigmoid:", loss_sigmoid)

# 4. 모델 마지막 층의 수정안을 쓴다.


## 문제 2. 이진 추론 함수를 직접 작성
print("문제 2. 이진 추론 함수를 직접 작성")
# raw logits와 확률 threshold를 받아 확률·label을 반환하는 함수를 작성하세요. 입력은 반드시 `(B,1)`이어야.
# 함수, 확률, preds를 제출
logits = torch.tensor([[-1.0], [0.0], [1.0], [2.0]])
print("logits shape:", logits.shape)
print("raw logits:", logits)

threshold = 0.5

# 1. 2차원 마지막 길이 1을 검사한다.
assert logits.shape[-1] == 1

# 2. Sigmoid로 확률을 만든다.
probs = torch.sigmoid(logits)
print("probs:", probs)

# 3. threshold 이상을 1로 만든다.
preds = (probs > threshold).long()
print("preds:", preds)

# 4. 반환 shape를 입력과 같게 유지한다.
assert preds.shape == logits.shape
print("input shape :", logits.shape)
print("preds shape :", preds.shape)
print("shape test: PASS")


## 문제 3. 누락 비용을 반영한 threshold 선택
print("문제 3. 누락 비용을 반영한 threshold 선택")
# 개인정보 문서를 놓치는 FN 비용은 5, 정상 문서를 추가 차단하는 FP 비용은 2
# threshold 0.5와 0.7을 같은 validation 네 건에서 비교

# probs=[0.90,0.65,0.55,0.40]
# target=[1,0,1,0]

probs = torch.tensor([0.90, 0.65, 0.55, 0.40])
target = torch.tensor([1, 0, 1, 0])

# 1. 후보별 label을 만든다.

results =[]

for threshold in [0.5, 0.7]:
    preds = (probs >= threshold).long()
    accuracy = (preds == target).float().mean()

    # 2. FN과 FP를 센다.
    TP = ((preds == 1) & (target == 1)).sum()
    TN = ((preds == 0) & (target == 0)).sum()
    FP = ((preds == 1) & (target == 0)).sum()
    FN = ((preds == 0) & (target == 1)).sum()

    # 3. `5*FN+2*FP`를 계산한다.
    cost = 5 * FN + 2 * FP

    results.append({
        "threshold": threshold,
        "FN": FN.item(),
        "FP": FP.item(),
        "cost": cost.item()
    })

results = pd.DataFrame(results)
best = results.loc[results["cost"].idxmin()]

print("=== 결과 ===")
print(results)

print("\n비용 최소 threshold:", best["threshold"])
print("최소 비용:", best["cost"])

print("\n=== 데이터 한계 ===")
print("validation 데이터:", len(target), "건")
print("데이터가 4건뿐이므로 threshold 일반화 성능을 확정할 수 없음")

# 4. 비용 최소 threshold와 데이터 한계를 보고한다.



# [4장 4강 심화] - 다중 분류 출력 실습 (1)
print("# [4장 4강 심화] - 다중 분류 출력 실습 (1)")

## 문제 1. 합이 1인데도 잘못된 Softmax 진단
print("문제 1. 합이 1인데도 잘못된 Softmax 진단")
# `dim=0`과 `dim=1` 결과의 행 합을 비교해 어떤 축이 잘못됐는지 증명하세요.
# wrong/correct row sums와 올바른 dim을 제출

logits = torch.tensor([[2,1,0],[0,1,2]])  # expected: 각 행의 class 확률 합=1
logits = logits.float()

# 1. 두 방향 Softmax를 계산한다.
probs1 = torch.softmax(logits, dim=0)
probs2 = torch.softmax(logits, dim=1)  
print(probs1)
print(probs2)

# 2. 각 행의 합을 구한다.
print("각 행의 합:", probs1.sum(dim=1))
print("각 행의 합:", probs2.sum(dim=1))

# 3. 샘플별 분포 조건을 적용한다.
assert torch.allclose(probs2.sum(dim=1), torch.ones(probs2.size(0)))
print("pass")
# 4. 열 합만 확인한 검사가 왜 부족한지 쓴다.


## 문제 2. 다중 분류 학습·추론 계약 코드 작성
print("문제 2. 다중 분류 학습·추론 계약 코드 작성")

# raw logits에서 loss, class 확률, 예측 class를 만드는 함수를 작성
## target 계약 위반은 loss 호출 전에 막아야.
### 함수, loss scalar 여부, row sums, preds를 제출

logits = torch.tensor([[2,1,0],[0,1,2]])  # expected: 각 행의 class 확률 합=1
logits = logits.float()
target = torch.tensor([0, 2], dtype=torch.long)


# 1. logits `(B,C)`와 target `(B,)`를 검사한다.
assert logits.ndim == 2
assert target.ndim == 1
assert logits.size(0) == target.size(0) # logits의 batch 수와 target의 샘플 수가 같은지 확인
print("logits shape:", logits.shape)
print("target shape:", target.shape)
print("shape test: PASS")

# 2. target long과 class 범위를 검사한다.
assert target.dtype == torch.long
assert target.min() >= 0
assert target.max() < logits.size(1)
print("target dtype:", target.dtype)
print("target range:", target.min().item(), "~", target.max().item())
print("target test: PASS")

# 3. raw logits로 CrossEntropyLoss를 계산한다.

probs = torch.softmax(logits, dim=1)  # class 차원
print("probs:")
print(probs)

# 4. 확률과 argmax 예측을 반환한다.
preds_from_logits = torch.argmax(logits, dim=1) 
print("preds_from_logits:", preds_from_logits)



# 문제 3. 라우터 후보의 운영 계약 승인
print("문제 3. 라우터 후보의 운영 계약 승인")
# approved, remeasure, rejected, selected를 제출

# A acc=0.88 row_sum_error=0.50 target=float safety_recall=0.81
# B acc=0.86 row_sum_error=0.00 target=long  safety_recall=0.78
# C acc=0.90 row_sum_error=0.00 target=long  safety_recall=미측정

results = pd.DataFrame({
    "candidate": ["A", "B", "C"],
    "acc": [0.88, 0.86, 0.90],
    "row_sum_error": [0.50, 0.00, 0.00],
    "target": ["float", "long", "long"],
    "safety_recall": [0.81, 0.78, None]
})
results
# 1. Tensor 계약 위반 후보를 제외한다.
results["status"] = "approved"

results["row_sum_ok"] = results["row_sum_error"] == 0.00
print(results[["candidate", "row_sum_error", "row_sum_ok"]])

# 2. 필수 recall이 누락된 후보를 재측정으로 분리한다.
results.loc[results["safety_recall"].isna(), "status"] = "remeasure"

# 3. 모든 자료가 있는 후보에 accuracy와 recall 기준을 적용한다.
results.loc[
    results["safety_recall"].notna() &
    (results["safety_recall"] < 0.75)|(~results["row_sum_ok"])|(results["acc"] < 0.85)
,
    "status"
] = "rejected"


# 4. 승인·재측정·탈락 사유를 각각 보고한다.
print(results[["candidate", "acc", "safety_recall", "status"]])
