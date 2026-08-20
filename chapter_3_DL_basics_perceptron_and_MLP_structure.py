##[3장 1강 심화] - 퍼셉트론 계산 실습
## 문제 1. 점수를 확률로 오해한 운영 로그 진단

# 검토 큐 label: 1 -> 담당자가 logit 1.0을 "위험 확률 100%"라고 보고했습니다. 실제 batch 점수와 예측을 계산하고 잘못된 해석을 바로잡아야.
## logits, preds, shape와 로그 정정 문장을 제출

# 1. 세 샘플의 가중합을 계산한다.
import torch
X = torch.tensor([[2., 1.], [1., 3.], [-1., 2.]])   # (3,2)
w = torch.tensor([0.8, -0.5])                       # (2, ) -> broadcasting (3,2)
b = -0.1                                            # float -> shape 없음.

z = (X @ w) + b
print("z      :", z)

# 2. 결과 shape를 확인한다.
print("z shape     :", tuple(z.shape)) # Shape 예상: (3,2)

# 3. 0 기준 label을 만든다.logit을 기준으로 0/1 예측값(pred)을 만드는 것

preds = (z >= 0).long()
print("preds       :", preds.tolist())

## logits, preds, shape와 로그 정정 문장을 제출
# 4. logit을 확률로 부를 수 없는 이유를 설명한다.


# 문제 2. shape 계약이 있는 퍼셉트론 함수 작성
## 입력 feature 수가 바뀌었을 때 조용히 잘못 계산되지 않도록 batch 전용 함수: 함수와 정상 batch의 출력 shape를 제출

# X = torch.tensor([[2., 1.], [1., 3.], [-1., 2.]])   # (3,2)
# w = torch.tensor([0.8, -0.5])                       # (2, ) -> broadcasting (3,2)
# b = -0.1                                            # float -> shape 없음.

def predict_batch(X, w, b):
    # 1. X는 2차원, w는 1차원인지 검사한다.
    assert X.ndim == 2 and w.ndim == 1, "expected (B,F) and (F,)"
     ## feature 수가 바뀌었는데 우연히 broadcasting으로 계산되는 상황을 잡을 수 있음.
    
    # 2. 마지막 feature 수가 같은지 검사한다.
    assert X.shape[-1] == w.shape[0], "feature/weight mismatch"
    ## 이 함수에 X와 w를 넣으려면 X의 feature 수와 w의 가중치 수가 같아야 한다.
    
    # 3. scalar bias만 허용한다.
    assert torch.tensor(b).ndim == 0 , "bias must be scalar"
    # 4. logits와 preds를 함께 반환한다.
    logits = (X * w) + b
    preds = (logits >= 0).long()
    return logits, preds

logits, preds = predict_batch(X, w, b)

print("logits shape:", logits.shape)
print("preds shape:", preds.shape)

# 문제 3. 오판 비용으로 bias 후보 선택
## 위험 요청 실패 비용 > 정상 요청 추가 검토 비용 * 5. 같은 weight에서 bias 두 개를 비교해 총 비용이 낮은 후보 선택.
### 후보별 비용과 selected bias를 제출

import torch

X = torch.tensor([[1., 0.], [0.2, 0.8], [0.6, 0.5]])
y = torch.tensor([1, 1, 0])
w = torch.tensor([1., -1.])
bias_candidates = [0.0, 0.7]

# 1. 후보별 예측을 계산한다.
scores = [X @ w + b for b in bias_candidates]

results=[]
for bias, z in zip(bias_candidates, scores):
    print("bias:", bias, "logits:", z)
# 2. false negative와 false positive를 센다.
    pred = (z > 0).long()
    fn = ((y == 1) & (pred == 0)).sum()
    fp = ((y == 0) & (pred == 1)).sum()
    print("bias:", bias, "FN:", fn.item(), "FP:", fp.item())

# 3. `5*FN + 1*FP`를 계산한다.  
    cost = 5 * fn + 1 * fp

    results.append({
        "bias": bias,
        "FN": fn.item(),
        "FP": fp.item(),
        "cost": cost.item()
    })
print(results)

selected = min(results, key=lambda r: r["cost"])

print("selected bias:", selected["bias"])
print("selected cost:", selected["cost"])

# 4. 선택과 선형 모델의 한계를 보고.
# -끝-

# 3장 2강
print("# 3장 2강")

## 설계: 6 -> 6 -> 3
## 코드: Linear(6,6) -> ReLU -> Linear(8,3)
## 입력 shape는 (B,6), 출력은 (B,3)

# 문제 1. 중간 shape 로그로 첫 오류 찾기
print("문제 1. 중간 shape 로그로 첫 오류 찾기")

import torch.nn as nn
torch.manual_seed(42)

X = torch.randn(5, 6)

## 설계와 코드 메타데이터를 비교해 forward가 실패하는 정확한 경계를 지정.
### 오류 경계, actual/expected feature, 최소 수정안을 제출
layers = [("fc1", 6, 6), ("relu", 6, 6), ("fc2", 8, 3)]
current = 6
issues = []
for name, expected_in, out_dim in layers:
    if current != expected_in:
        issues.append(f"{name}: expected {expected_in}, got {current}")
        break
    current = out_dim
print("first_issue:", issues[0])
print("minimal_fix:", "Linear(6, 3)")
# 4. 최소 수정과 대안 수정의 차이를 설명한다.



# 문제 2. 층별 shape를 반환하는 MLP 작성
print("문제 2. 층별 shape를 반환하는 MLP 작성")

## 입력 6, hidden 4, class 3인 MLP를 만들고 forward 검증 시 중간 shape도 반환
### 모델 코드, hidden/logits shape, parameter 수를 제출

# 1. 두 Linear를 `__init__`에 등록한다.
class DebugMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=4, num_classes=3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
# 2. 첫 Linear 뒤 ReLU를 적용한다.
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
# 3. logits에는 별도 후처리를 넣지 않는다.

# 4. batch 5개의 shape 흐름을 검증한다.
    def forward(self, x):
        print("input  :", x.shape)

        x = self.flatten(x)
        print("flatten:", x.shape)

        x = self.fc1(x)
        print("fc1    :", x.shape)

        x = self.relu(x)
        print("relu   :", x.shape)

        logits = self.fc2(x)
        print("fc2    :", logits.shape)

        return logits
   
model = DebugMLP()
params = sum(p.numel() for p in model.parameters())
logits = model(X)
print("parameters:", params)
     
# 문제 3. 65개 parameter 예산의 구조 선택
print("문제 3. 65개 parameter 예산의 구조 선택")

## 세 후보 중 parameter가 65개 이하이면서 학습 가능한 Linear 층 수가 가장 많은 구조 선택. 동률이면 parameter가 적은 것을 선택.

# 1. 각 Linear의 weight와 bias를 센다.
# A: 6 -> 4 -> 3
model_A= nn.Sequential(
    nn.Linear(6, 4),
    nn.ReLU(),
    nn.Linear(4, 3)
)
# 각 parameter의 이름과 shape도 확인합니다.
for name, param in model_A.named_parameters():
    print("model_A")
    print(name, param.shape, "numel=", param.numel())

# B: 6 -> 8 -> 3
model_B= nn.Sequential(
    nn.Linear(6, 8),
    nn.ReLU(),
    nn.Linear(8, 3)
)
# 각 parameter의 이름과 shape도 확인합니다.
for name, param in model_B.named_parameters():
    print("model_B")
    print(name, param.shape, "numel=", param.numel())

# C: 6 -> 4 -> 4 -> 3
model_C= nn.Sequential(
    nn.Linear(6, 4),
    nn.ReLU(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 3)
)
# 각 parameter의 이름과 shape도 확인합니다.
for name, param in model_C.named_parameters():
    print("model_C")
    print(name, param.shape, "numel=", param.numel())


# 2. 후보 총 parameter를 계산한다.
total_params_A = sum(p.numel() for p in model_A.parameters() if p.requires_grad)
print(f"model_A total_params={total_params_A}")

total_params_B = sum(p.numel() for p in model_B.parameters() if p.requires_grad)
print(f"model_B total_params={total_params_B}")

total_params_C = sum(p.numel() for p in model_C.parameters() if p.requires_grad)
print(f"model_C total_params={total_params_C}")

# 3. 예산 초과 후보를 제외한다.
models = {
    "model_A": model_A,
    "model_B": model_B,
    "model_C": model_C,
}

for name, model in models.items():
    total_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    if total_params <= 65:
        print(f"{name}: {total_params} → 선택")
    else:
        print(f"{name}: {total_params} → 제외")

# 4. 선택 기준과 표현력 판단의 한계를 쓴다.
# 끝

# [3장 3강 심화] - Linear layer shape 실습 (1)
print("[3장 3강 심화] - Linear layer shape 실습")
# layer=Linear(3,4)
# x.shape=(2,3)
# stored weight.shape=(4,3)

# 문제 1. 전치 누락 원인 진단: 잘못된/올바른 shape 계약과 수정식을 제출한다.
print("문제 1: 전치 누락 원인 진단")
# 1. x와 weight의 행렬 곱 내부 차원을 비교한다.
x = torch.randn(2, 3)
weight = torch.randn(4, 3)
print("x shape:", x.shape)
print("weight shape:", weight.shape)

# x @ weight에서 내부 차원 비교: 전치 누락 진단
print("x 내부 차원:", x.shape[-1])
print("weight 내부 차원:", weight.shape[0])

print("일치:", x.shape[-1] == weight.shape[0])

# 2. 저장 weight를 전치해야 하는 이유를 쓴다.
# weight = (out_features, in_features) = (4,3)
# PyTorch Linear가 weight를 (out_features, in_features)로 저장한다.
# 입력 (batch, in_features)와 행렬 곱을 하려면 weight를 (in_features, out_features)로 맞춰야 하므로 전치한다.

# 3. 올바른 출력 shape를 계산한다.

correct_out_shape = (x.shape[0], weight.shape[0]) # x.shape[0] Batch_size. weight.shape[0]은 output feature 수.
print("correct output shape:", correct_out_shape)

# 4. weight shape를 `(in,out)`로 저장한다고 오해한 지점을 지적한다.


# 문제 2. 수동 계산과 모듈 결과 검증: 두 출력과 same=True를 제출한다.
print("문제 2. 수동 계산과 모듈 결과 검증")

# 1. Linear(3,2)를 만든다.
linear = nn.Linear(3, 2)

    
# 2. gradient 추적 없이 parameter 값을 복사한다.
with torch.no_grad():
    linear.weight.copy_(
        torch.tensor([
            [1., 0., -1.],
            [0.5, 0.5, 0.5]
        ])
    )
    linear.bias.copy_(torch.tensor([0.2, -0.5]))
x = torch.tensor([[1., 2., 3.], [0., -1., 2.]])


# 3. 모듈과 수동 출력을 각각 계산한다.
y_module = linear(x)
y_manual = x @ linear.weight.T + linear.bias
print("module output:")
print(y_module)
print("manual output:")
print(y_manual)
# 4. shape와 근사 동일성을 검증한다.
print("same?", torch.allclose(y_module, y_manual))

 
# 문제 3. 출력 head 예산 검토: 후보별 parameter와 selected를 제출한다.
print("문제 3. 출력 head 예산 검토")

A=nn.Linear(12,4)
B=nn.Linear(12,5)
C=nn.Linear(12,7)

required_classes=5
budget=80

# 1. 후보별 weight와 bias shape를 적는다.
print("A weight shape:", A.weight.shape)
print("A bias shape  :", A.bias.shape)

print("B weight shape:", B.weight.shape)
print("B bias shape  :", B.bias.shape)

print("C weight shape:", C.weight.shape)
print("C bias shape  :", C.bias.shape)

# 2. 총 parameter 수를 계산한다.
print("total params:", A.weight.numel() + A.bias.numel())
print("total params:", B.weight.numel() + B.bias.numel())
print("total params:", C.weight.numel() + C.bias.numel())

# 3. 필요한 class 수 5를 만족하는지 확인한다.

print("A:", A.out_features == required_classes)
print("B:", B.out_features == required_classes)
print("C:", C.out_features == required_classes)

# 4. 예산과 출력 계약을 모두 통과한 후보를 고른다.

models = {
    "A": A,
    "B": B,
    "C": C
}

eligible = []

for name, model in models.items():
    total_params = model.weight.numel() + model.bias.numel()

    if total_params <= budget and model.out_features == required_classes:
        eligible.append(name)

print("eligible:", eligible)

selected = eligible[0] if eligible else None

print("selected:", selected)



# [3장 4강 심화] - 입출력 차원 계산
print("[3장 4강 심화] - 입출력 차원 계산")

# 문제 1. 실행 가능한 샘플 혼합 버그 진단
print("문제 1. 실행 가능한 샘플 혼합 버그 진단")

# wrong/correct shape, batch 보존 판정, 원인 설명을 제출한다.

# 입력은 (B,1,4,4), 샘플당 feature는 16개다.
# images.shape=(2,1,4,4)
# wrong_flat.shape=(32,)
# expected_flat.shape=(2,16)

images = torch.randn(2, 1, 4, 4)

# 1. 전체 flatten과 batch 보존 flatten 두 방식의 shape를 계산한다.

flat_all = torch.flatten(images, start_dim=0)
print("before:", images.shape)
print("after :", flat_all.shape)

flat_batch = torch.flatten(images, start_dim=1)
print("batch_coserve_before:", images.shape)
print("batch_coserve_after :", flat_batch.shape)


# 2. 원소 수가 같은지 확인한다.
assert flat_all.numel() == images.numel()
assert flat_batch.numel() == images.numel()

# 3. batch 축 보존 여부를 확인한다.
## start_dim=1이면 images.shape[0]이 그대로 유지되는지

assert flat_batch.shape[0] == images.shape[0]
assert flat_all.shape[0] != images.shape[0]

# 4. `Linear(32,3)`로 맞추는 수정이 왜 틀렸는지 설명한다.



# 문제 2. 동적 shape flatten guard 작성
print("문제 2. 동적 shape flatten guard 작성")

def flatten_for_mlp(images, expected_features):
    # 1. 입력이 4차원인지 검사한다.
    assert images.ndim == 4, "expected (B,C,H,W)"

    # 2. batch만 남겨 flatten한다.
    flat = images.reshape(images.shape[0], -1)

    # 3. 기대 feature 수를 인자로 검사한다.
    assert flat.shape[1] == expected_features, "in_features mismatch"

    return flat

# 4. batch 1과 batch 3에서 재사용한다.

images_batch1 = torch.randn(1, 1, 4, 4)
images_batch3 = torch.randn(3, 1, 4, 4)

flat1 = flatten_for_mlp(images_batch1, expected_features=16)
flat3 = flatten_for_mlp(images_batch3, expected_features=16)

print("batch 1:", flat1.shape)
print("batch 3:", flat3.shape)


# 문제 3. shape는 같지만 샘플 순서가 다른 전처리 감사
print("문제 3. shape는 같지만 샘플 순서가 다른 전처리 감사")

## 원본 batch에는 샘플 두 개가 있고 각 샘플 값의 간단한 checksum은 [120,376]입니다.
## checksum은 여기서는 각 데이터가 제대로 보존됐는지 확인하기 위한 간단한 검사값(원소 합 등)

# batch와 이미지 크기가 달라져도 샘플당 feature를 자동 계산하고 기대 feature와 비교하는 함수를 작성
## 세 전처리 후보의 output shape와 샘플별 checksum을 함께 감사해 Linear(16,3)에 넘길 수 있는 안을 승인

original_shape = (2, 16)
original_checksum = [120, 376]

candidates = {
    "A": ((2, 16), [120, 376]),
    "B": ((2, 16), [376, 120]),
    "C": ((1, 32), [496])
}

def select_candidate(candidates, source_checksums, model):
    audit = {}

    for name, candidate in candidates.items():

        shape_ok = (
            candidate["shape"][0] == 2
            and candidate["shape"][1] == model.in_features
        )

        sample_order_ok = (
            candidate["checksums"] == source_checksums
        )

        audit[name] = {
            "input_contract": shape_ok,
            "sample_order": sample_order_ok,
        }

    approved = [
        name
        for name, checks in audit.items()
        if all(checks.values())
    ]

    selected = approved[0] if len(approved) == 1 else "보류"

    if selected != "보류":
        batch_size = candidates[selected]["shape"][0]
        logits_shape = (batch_size, model.out_features)
    else:
        logits_shape = None

    return audit, selected, logits_shape

source_checksums = [120, 376]

candidates = {
    "A": {"shape": (2, 16), "checksums": [120, 376]},
    "B": {"shape": (2, 16), "checksums": [376, 120]},
    "C": {"shape": (1, 32), "checksums": [496]},
}

model = nn.Linear(16, 3)

audit, selected, logits_shape = select_candidate(
    candidates,
    source_checksums,
    model
)

print("audit:", audit)
print("selected:", selected)
print("logits_shape:", logits_shape)