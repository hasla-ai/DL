import torch

print("=" * 50)
print("🚀 PyTorch 핸즈온 미션 5탄: Autograd (자동 미분) 시작")
print("=" * 50)

# ==========================================
# [MISSION 5-1] requires_grad 설정 및 Gradient 추적
# ==========================================
# 목표: 
# 1. 스칼라 값 3.0을 가지는 텐서 `x`를 만들고, 미분 추적(`requires_grad=True`)을 활성화하세요.
# 2. 수식 y = 2 * x^2 + 5 * x + 1 에 해당하는 계산 그래프 `y`를 작성하세요.

x = torch.tensor(3.0, requires_grad=True)
y = 2 * (x ** 2) + 5 * x + 1

# [검증 5-1]
# 수식 y = 2x^2 + 5x + 1 에 x=3을 대입하면 y = 2(9) + 5(3) + 1 = 34.0
assert y.requires_grad == True, "y에 Gradient 추적이 활성화되지 않았습니다."
assert y.item() == 34.0, f"y의 계산 결과가 올바르지 않습니다: {y.item()}"

print("\n✅ MISSION 5-1 PASSED!")
print(f"  • 입력 x: {x.item()}")
print(f"  • y = 2x^2 + 5x + 1 계산 결과: {y.item()}")


# ==========================================
# [MISSION 5-2] 역전파(backward)와 미분값(grad) 검증
# ==========================================
# 목표:
# 1. `y.backward()`를 호출하여 dy/dx 미분값을 계산하세요.
# 2. x = 3 일 때 dy/dx = 4x + 5 = 17.0 이 맞는지 `x.grad`로 검증하세요.

y.backward()  # 역전파 수행

# [검증 5-2]
# dy/dx = 4x + 5 -> x=3 일 때 dy/dx = 17.0
expected_grad = 17.0
assert x.grad is not None, "x.grad가 계산되지 않았습니다."
assert x.grad.item() == expected_grad, f"Gradient 값이 일치하지 않습니다: {x.grad.item()}"

print("\n✅ MISSION 5-2 PASSED!")
print(f"  • PyTorch가 계산한 dy/dx (x.grad): {x.grad.item()}")
print(f"  • 수학적 해석해 (4x + 5): {expected_grad}")


# ==========================================
# [MISSION 5-3] Gradient 누적 방지 (zero_grad의 필요성)
# ==========================================
# 목표:
# PyTorch는 .backward()를 호출할 때마다 기본적으로 grad를 '누적(더함)'합니다.
# 1. 동일한 연산을 한 번 더 실행하고 backward()를 호출하여 grad가 누적되는 현상을 확인하세요.
# 2. 그 후 `x.grad.zero_()`를 사용하여 경사값을 0으로 리셋하세요.

# 두 번째 연산 실행 및 누적 테스트
y2 = 2 * (x ** 2) + 5 * x + 1
y2.backward()  # grad가 이전값(17.0)에 추가로 더해짐 -> 17.0 + 17.0 = 34.0

assert x.grad.item() == 34.0, f"Gradient 누적 실패: {x.grad.item()}"

# Gradient 리셋
x.grad.zero_()  # In-place 연산으로 grad를 0으로 만듦

# [검증 5-3]
assert x.grad.item() == 0.0, "x.grad가 초기화되지 않았습니다."

print("\n✅ MISSION 5-3 PASSED!")
print("  • backward() 재호출 시 누적된 Gradient: 34.0")
print(f"  • x.grad.zero_() 후 리셋된 Gradient: {x.grad.item()}")


# ==========================================
# [MISSION 5-4] torch.no_grad()를 활용한 기울기 계산 비활성화 (평가/추론 단계)
# ==========================================
# 목표:
# 모델 평가(Evaluation) 단계에서는 Gradient를 계산할 필요가 없어 메모리를 절약해야 합니다.
# `with torch.no_grad():` 블록 내에서 계산된 `z`는 `requires_grad`가 False임을 검증하세요.

with torch.no_grad():
    z = x ** 2 + 10

# [검증 5-4]
assert z.requires_grad == False, "no_grad 블록 내에서도 Gradient 추적이 활성화되어 있습니다."

print("\n✅ MISSION 5-4 PASSED!")
print(f"  • torch.no_grad() 블록 내 z.requires_grad: {z.requires_grad}")


print("\n" + "=" * 50)
print("🎉 ALL MISSIONS IN MISSION 5 PASSED! Autograd 검증 완료!")
print("=" * 50)