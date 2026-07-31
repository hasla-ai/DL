import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 50)
print("🚀 PyTorch 핸즈온 미션 8탄: Loss Function & Optimizer 시작")
print("=" * 50)

# ==========================================
# [MISSION 8-1] 대표적인 손실 함수 (MSELoss & CrossEntropyLoss) 검증
# ==========================================
# 목표:
# 1. 회귀(Regression)용 손실함수 MSELoss 계산
# 2. 다중 클래스 분류(Classification)용 CrossEntropyLoss 계산

# 1) 회귀 손실 계산 (Mean Squared Error)
mse_criterion = nn.MSELoss()
pred_reg = torch.tensor([2.5, 3.0], requires_grad=True)
target_reg = torch.tensor([3.0, 1.0])
# MSE = ((2.5 - 3.0)^2 + (3.0 - 1.0)^2) / 2 = (0.25 + 4.0) / 2 = 2.125
mse_loss = mse_criterion(pred_reg, target_reg)

# 2) 분류 손실 계산 (Cross Entropy)
ce_criterion = nn.CrossEntropyLoss() # 내부적으로 Softmax를 적용하고 CrossEntropy.
pred_cls = torch.tensor([[2.0, 0.5], [0.1, 3.1]])  # Batch Size=2, Class=2 (Logits)
target_cls = torch.tensor([0, 1])                 # 정답 클래스 인덱스 (LongTensor)
ce_loss = ce_criterion(pred_cls, target_cls)

# [검증 8-1]
assert torch.isclose(mse_loss, torch.tensor(2.125)), f"MSE 손실 계산 불일치: {mse_loss.item()}"
assert ce_loss.item() > 0.0, "CrossEntropy 손실 값이 올바르지 않습니다."

print("\n✅ MISSION 8-1 PASSED!")
print(f"  • MSE Loss 계산 결과          : {mse_loss.item():.4f}")
print(f"  • CrossEntropy Loss 계산 결과 : {ce_loss.item():.4f}")


# ==========================================
# [MISSION 8-2] Optimizer 연동 및 1-Step 가중치 업데이트 검증
# ==========================================
# 목표:
# 1. 간단한 선형 모델(nn.Linear)과 Adam Optimizer를 생성합니다.
# 2. 역전파(backward) 및 optimizer.step() 실행 후, 가중치(Weight)가 실제로 갱신되었는지 검증합니다.

model = nn.Linear(5, 2)
optimizer = optim.Adam(model.parameters(), lr=0.1) #lr;learning rate.

# 업데이트 전 초기 가중치 복사해 두기
initial_weight = model.weight.clone().detach()

# 더미 데이터 및 손실 계산
dummy_x = torch.randn(4, 5)  # Batch Size=4, Feature=5
dummy_y = torch.tensor([0, 1, 0, 1])  # Target Classes

# 1. Forward Pass
logits = model(dummy_x)
loss = ce_criterion(logits, dummy_y)

# 2. Backward Pass & Weight Update
optimizer.zero_grad()  # step 1. 기울기 초기화
loss.backward()        # step 2. 역전파로 기울기 계산
optimizer.step()       # step 3. Optimizer로 가중치 갱신

# [검증 8-2]
# optimizer.step() 이후 가중치가 이전과 달라졌는지 확인
updated_weight = model.weight
is_weight_changed = not torch.equal(initial_weight, updated_weight)

assert is_weight_changed == True, "optimizer.step() 이후 가중치가 갱신되지 않았습니다."

print("\n✅ MISSION 8-2 PASSED!")
print(f"  • 계산된 Loss 값                   : {loss.item():.4f}")
print(f"  • 가중치(Weight) 변경 여부 검증 완료 : {is_weight_changed}")


# ==========================================
# [MISSION 8-3] 학습률(Learning Rate) 스케줄러 기본 작동 검증
# ==========================================
# 목표:
# optim.lr_scheduler.StepLR을 활용하여 학습 진행에 따라 Learning Rate가 감소하는지 검증합니다.

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
initial_lr = optimizer.param_groups[0]['lr']

# 스케줄러 1 스텝 진행 (Learning Rate 감소)
scheduler.step()
updated_lr = optimizer.param_groups[0]['lr']

# [검증 8-3]
assert updated_lr == initial_lr * 0.5, f"Learning Rate 스케줄러 감소 실패: {updated_lr}"

print("\n✅ MISSION 8-3 PASSED!")
print(f"  • 초기 Learning Rate : {initial_lr}")
print(f"  • 감쇄 후 Learning Rate: {updated_lr}")

print("\n" + "=" * 50)
print("🎉 ALL MISSIONS IN MISSION 8 PASSED! Loss & Optimizer 검증 완료!")
print("=" * 50)