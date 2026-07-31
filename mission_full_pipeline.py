import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

print("=" * 50)
print("🚀 PyTorch 핸즈온 미션 9탄: 전체 학습/평가 루프 (Full Pipeline) 시작")
print("=" * 50)

# ----------------------------------------------------
# [준비 단계] 커스텀 데이터셋 및 MLP 모델 클래스 정의
# ----------------------------------------------------
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=200, feature_dim=8):
        # 재현성을 위한 시드 고정
        torch.manual_seed(42)
        self.X = torch.randn(num_samples, feature_dim)
        # 선형 경계에 기반한 0 또는 1 라벨 생성
        self.y = (self.X.sum(dim=1) > 0).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ClassificationMLP(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=16, num_classes=2):
        super(ClassificationMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# [MISSION 9-1] Training Loop 및 Loss 감소 검증
# ==========================================
# 목표:
# 5 Epoch 동안 학습을 진행하면서 첫 번째 Epoch의 Loss보다 마지막 Epoch의 Loss가 줄어들었는지 검증하세요.

# 1. 데이터셋 & 데이터로더
train_dataset = SyntheticDataset(num_samples=200, feature_dim=8)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 2. 모델, 손실함수, 최적화기
model = ClassificationMLP(in_dim=8, hidden_dim=16, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

epoch_losses = []

# 학습 루프 실행
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()  # 학습 모드 전환
    running_loss = 0.0

    for inputs, targets in train_loader:
        optimizer.zero_grad()           # 1. Gradient 초기화
        outputs = model(inputs)         # 2. Forward Pass
        loss = criterion(outputs, targets)  # 3. Loss 계산
        loss.backward()                 # 4. Backward Pass (역전파)
        optimizer.step()                # 5. 가중치 갱신

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    epoch_losses.append(epoch_loss)
    print(f"  • Epoch [{epoch}/{epochs}] Train Loss: {epoch_loss:.4f}")

# [검증 9-1]
# 학습이 진행됨에 따라 손실이 감소했는지 검증
assert epoch_losses[-1] < epoch_losses[0], f"Loss가 감소하지 않았습니다: 초기 {epoch_losses[0]:.4f} -> 최종 {epoch_losses[-1]:.4f}"

print("\n✅ MISSION 9-1 PASSED!")
print(f"  • 초기 Epoch Loss : {epoch_losses[0]:.4f}")
print(f"  • 최종 Epoch Loss : {epoch_losses[-1]:.4f} (손실 감소 성공!)")


# ==========================================
# [MISSION 9-2] Evaluation Loop 및 Accuracy(정확도) 측정 검증
# ==========================================
# 목표:
# torch.no_grad() 상태에서 Test 데이터셋에 대한 Accuracy를 산출하고, 50% 이상을 달성하는지 검증하세요.

test_dataset = SyntheticDataset(num_samples=50, feature_dim=8)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model.eval()  # 평가 모드 전환 (Dropout/BatchNorm 등이 있다면 비활성화됨)

correct = 0
total = 0

with torch.no_grad():  # Gradient 계산 끄기 (메모리 절약 & 추론 속도 향상)
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)  # 가장 높은 확률/Logit의 인덱스 추출
        correct += (preds == targets).sum().item()
        total += targets.size(0)

accuracy = (correct / total) * 100.0

# [검증 9-2]
assert accuracy >= 50.0, f"평가 정확도가 기준치에 미달합니다: {accuracy:.2f}%"

print("\n✅ MISSION 9-2 PASSED!")
print(f"  • Test Dataset 평가 결과: {correct}/{total} 정답 맞춤")
print(f"  • Accuracy (정확도)      : {accuracy:.2f}%")


# ==========================================
# [MISSION 9-3] 모델 가중치 체크포인트 저장 및 불러오기 검증
# ==========================================
# 목표:
# 1. model.state_dict()를 사용하여 모델 가중치를 'model_checkpoint.pth'로 저장합니다.
# 2. 새로운 모델 인스턴스를 생성하고 저장된 가중치를 불러와 예측 결과가 100% 동일한지 검증합니다.

ckpt_path = "model_checkpoint.pth"

# 1. 가중치 저장
torch.save(model.state_dict(), ckpt_path)

# 2. 새 모델 인스턴스 생성 및 가중치 로드
new_model = ClassificationMLP(in_dim=8, hidden_dim=16, num_classes=2)
new_model.load_state_dict(torch.load(ckpt_path))
new_model.eval()

# 3. 기존 모델과 새 모델의 예측 출력 동일성 비교
dummy_sample = torch.randn(1, 8)
with torch.no_grad():
    orig_output = model(dummy_sample)
    new_output = new_model(dummy_sample)

# [검증 9-3]
assert torch.allclose(orig_output, new_output), "저장/로드된 모델의 출력값이 일치하지 않습니다!"

# 테스트 후 임시 체크포인트 파일 삭제
if os.path.exists(ckpt_path):
    os.remove(ckpt_path)

print("\n✅ MISSION 9-3 PASSED!")
print(f"  • original model output : {orig_output.numpy()}")
print(f"  • loaded model output   : {new_output.numpy()}")
print("  • 모델 체크포인트 Save/Load 일치성 검증 완료!")

print("\n" + "=" * 50)
print("🎉 ALL MISSIONS IN MISSION 9 PASSED!")
print("🏆 PyTorch 핸즈온 프로젝트의 모든 미션을 완벽히 통과하셨습니다!")
print("=" * 50)