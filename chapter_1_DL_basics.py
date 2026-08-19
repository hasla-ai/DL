
# 2-1 적용판단 도우미: 회의 반복 주제 체크리스트 만들기

# 검증 가능 정답 코드
# 샘플 수·규칙 유지비·문맥 의존도를 각각 검사해 딥러닝 적용 여부를 한 조건으로 단정하지 않습니다.
# 입력 숫자는 팀의 수업용 기준과 비교하고 결과는 배포 승인이 아니라 다음 실험 권고로 제한합니다.
def recommend_start(raw_unstructured, labeled_count, stable_rule):
    # 명시적이고 안정적인 규칙은 데이터 양과 무관하게 먼저 보존합니다.
    if stable_rule:
        return "rule_first"
    # 원본 텍스트·이미지처럼 표현 설계가 어려우면서 검증 가능한 라벨이 있을 때만 후보로 올립니다.
    if raw_unstructured and labeled_count >= 5000:
        return "deep_learning_candidate"
    # 데이터가 적거나 구조화 입력이면 단순 기준선을 먼저 만들어 비교 기준을 남깁니다.
    return "simple_baseline_first"

cases = [(False, 400, True), (True, 28000, False), (True, 120, False)]
print([recommend_start(*case) for case in cases])

# 3-1 불완전한 검증 근거로 승인과 재측정 구분

candidates = {
    "A": {"accuracy": 0.884, "latency_ms": 3, "latency_runs": 3},
    "B": {"accuracy": 0.921, "latency_ms": 10, "latency_runs": 1},
    "C": {"accuracy": 0.908, "latency_ms": 13, "latency_runs": 3},
}
approved, remeasure, rejected = [], [], {}
for name, result in candidates.items():
    failed = []
    if result["accuracy"] < 0.90:
        failed.append("accuracy")
    if result["latency_ms"] > 12:
        failed.append("latency")
    if failed:
        rejected[name] = failed
    elif result["latency_runs"] < 3:
        remeasure.append(name)
    else:
        approved.append(name)
decision = max(approved, key=lambda n: candidates[n]["accuracy"]) if approved else "보류"
print("approved:", approved)
print("remeasure:", remeasure)
print("rejected:", rejected)
print("decision:", decision)


## 1-3강 학습 파이프라인 순서
# 문제 1. 뒤섞인 실행 기록 진단하기

observed = ["forward", "loss", "step", "zero_grad", "backward"]

# 전체 목록을 한 줄 순서로 강제하지 않고, 실제로 값이 필요한 의존 관계만 검사합니다.

dependencies = [
    ("forward", "loss"),
    ("loss", "backward"),
    ("zero_grad", "backward"),
    ("backward", "step"),
]

position = {stage: observed.index(stage) for stage in observed}

violations = [
    f"{left} must precede {right}"
    for left, right in dependencies
    if position[left] > position[right] ##의존관계가 위반
]

print("first_violation:", violations[0])
print("recommended:", "zero_grad -> forward -> loss -> backward -> step")

## 문제 2. 한 batch 학습 코드를 직접 복구하기

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

print("PyTorch 버전:", torch.__version__)
print("CUDA 사용 가능 여부 (is_available):", torch.cuda.is_available())

if torch.cuda.is_available():
    print("현재 사용 중인 GPU 이름:", torch.cuda.get_device_name(0))
    print("CUDA 버전:", torch.version.cuda)
    print("연결된 GPU 개수:", torch.cuda.device_count())
else:
    print("❌ 현재 PyTorch 환경에서 CUDA(GPU)를 사용할 수 없습니다. (CPU 모드로 동작 중)")
    
x = torch.tensor([[1.0], [2.0]]) # torch.Size([2, 1])
y = torch.tensor([[2.0], [4.0]]) # torch.Size([2, 1])
# 문서 임베딩을 점수 하나로 바꾸는 작은 회귀 모델과 한 step을 완성.
# 조건: 한 batch에서 실제로 파라미터가 바뀌도록 코드를 완성하고, loss 전후가 아니라 weight 전후를 검증.
# 1. seed를 고정하고 모델·loss·optimizer를 만든다.
torch.manual_seed(42)

dataset = TensorDataset(x, y)

# dataset의 전체 길이 확인
total_len = len(dataset)
train_len = int(total_len * 0.8)
valid_len = total_len - train_len

# 하드코딩 [160, 40] 대신 계산된 변수 전달
train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(
    nn.Linear(1, 16),   # 데이터 $X$의 Feature 개수(마지막 차원)와 nn.Linear의 첫 번째 입력 차원을 1로 맞춰
    nn.ReLU(),
    nn.Linear(16, 1),
).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# 한 batch에서 weight 변화 검증하기
# ---------------------------------------------------------

# (1) 첫 번째 batch 데이터 가져오기
batch_x, batch_y = next(iter(train_loader))
batch_x, batch_y = batch_x.to(device), batch_y.to(device)

# (2) 학습 전 첫 번째 레이어의 Weight 복사해두기
weight_before = model[0].weight.clone()

# (3) 한 Step 학습 진행 (Forward -> Loss -> Backward -> Step)
optimizer.zero_grad()            # 1. 기울기 초기화
pred = model(batch_x)            # 2. 순전파
loss = loss_fn(pred, batch_y)    # 3. 손실 계산
loss.backward()                  # 4. 역전파
optimizer.step()                 # 5. 가중치 업데이트

# (4) 학습 후 첫 번째 레이어의 Weight 가져오기
weight_after = model[0].weight

# (5) Weight 전/후 변화량 검증 (0이 아니어야 실제 업데이트된 것!)
weight_diff = torch.abs(weight_after - weight_before).sum().item()

print(f"Weight 업데이트 변화량: {weight_diff:.6f}")
assert weight_diff > 0, "Weight가 업데이트되지 않았습니다!"
print("✅ 한 batch 학습 후 Weight가 성공적으로 변경되었습니다.")