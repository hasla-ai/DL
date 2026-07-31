import torch
import torch.nn as nn

print("=" * 50)
print("🚀 PyTorch 핸즈온 미션 7탄: nn.Module 기반 인공신경망(MLP) 시작")
print("=" * 50)

# ==========================================
# [MISSION 7-1] nn.Module 상속 및 다층 인공신경망(MLP) 구조 정의
# ==========================================
# 목표:
# 입력 차원 10 -> 은닉층 32 (ReLU 활성화 함수) -> 출력 차원 2 구조를 가지는 SimpleMLP 정의

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=2):
        super(SimpleMLP, self).__init__()
        
        # 선형 레이어(Linear Layer) 및 활성화 함수 정의
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = SimpleMLP(input_dim=10, hidden_dim=32, output_dim=2)

# [검증 7-1]
# 모델 파라미터(fc1.weight, fc1.bias, fc2.weight, fc2.bias) 총 4개가 올바르게 등록되었는지 확인
param_names = [name for name, _ in model.named_parameters()]
assert len(param_names) == 4, f"등록된 파라미터 수가 올바르지 않습니다: {param_names}"
assert "fc1.weight" in param_names and "fc2.bias" in param_names, "레이어 파라미터명이 일치하지 않습니다."

print("\n✅ MISSION 7-1 PASSED!")
print(f"  • 생성된 신경망 모델 레이어 구조:\n{model}")


# ==========================================
# [MISSION 7-2] Forward Pass(순전파) 연산 및 출력 Shape 검증
# ==========================================
# 목표:
# Batch Size가 16이고 Feature Size가 10인 입력 텐서가 입력되었을 때,
# 최종 출력 Shape이 (16, 2)로 변환되는지 검증하세요.

dummy_input = torch.randn(16, 10)  # (Batch Size=16, Feature Size=10)
output = model(dummy_input)        # forward() 자동 호출

# [검증 7-2]
assert output.shape == (16, 2), f"출력 Shape 불일치: {output.shape}"
assert output.requires_grad == True, "모델 출력의 Gradient 추적이 비활성화되어 있습니다."

print("\n✅ MISSION 7-2 PASSED!")
print(f"  • 입력 Shape : {dummy_input.shape}")
print(f"  • 출력 Shape : {output.shape} (Batch Size x Output Classes)")


# ==========================================
# [MISSION 7-3] Sequential Container 방식 및 레이어 가중치 형태 검증
# ==========================================
# 목표:
# nn.Sequential을 활용하여 동일한 구조를 가볍게 선언해 봅니다.

seq_model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

seq_output = seq_model(dummy_input)

# [검증 7-3]
assert seq_output.shape == (16, 2), f"Sequential 출력 Shape 불일치: {seq_output.shape}"
assert seq_model[0].weight.shape == (32, 10), f"fc1 가중치 Matrix Shape 불일치: {seq_model[0].weight.shape}"

print("\n✅ MISSION 7-3 PASSED!")
print(f"  • Sequential 모델 출력 Shape: {seq_output.shape}")
print(f"  • 첫 번째 선형 레이어 가중치 Shape: {seq_model[0].weight.shape} (Out Features x In Features)")

print("\n" + "=" * 50)
print("🎉 ALL MISSIONS IN MISSION 7 PASSED! nn.Module 신경망 구축 검증 완료!")
print("=" * 50)