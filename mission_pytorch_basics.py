## 3. 미션 1탄 검증 코드 (`mission_pytorch_basics.py`)

import torch
import numpy as np

print("=" * 50)
print("🚀 PyTorch 텐서 핸즈온 미션 1탄 시작")
print("=" * 50)

# ==========================================
# [MISSION 1] 텐서 생성 및 기본 속성 검증
# ==========================================
# 목표:
# 1. 1부터 12까지 연속된 정수를 포함하는 1D 텐서 `m1`을 만드세요.
# 2. float32 타입의 3행 4열 1로 채워진 텐서 `m1_ones`를 만드세요.

m1 = torch.arange(1, 13)
m1_ones = torch.ones(3, 4, dtype=torch.float32)

# [검증 1](Unit Test)
## 만약 1차원 배열(Shape), 타입(Type), 파이토치 텐서 객체(합계값) 아니면 불일치 메세지.

assert m1.shape == (12,), f"M1 Shape 불일치: {m1.shape}"
assert m1_ones.dtype == torch.float32, f"M1 타입 불일치: {m1_ones.dtype}"
assert m1_ones.sum().item() == 12.0, "M1 합계 불일치"

print("\n✅ MISSION 1 PASSED!")
print(f"  • m1 값: {m1}")
print(f"  • m1_ones 합계: {m1_ones.sum().item()}")

# ==========================================
# [MISSION 2] 차원 변경 (Reshape & Squeeze/Unsqueeze)
# ==========================================
# 목표:
# 1. MISSION 1의 `m1`(원소 12개)을 (3, 4) 형상의 `m2_2d`로 변환하세요.
# 2. `m2_2d`에 배치(Batch) 차원을 앞에 1개 추가하여 (1, 3, 4) 형상의 `m2_batch`를 만드세요.
# 3. `m2_batch`에서 크기가 1인 차원을 전부 제거하여 (3, 4) 형상의 `m2_squeezed`를 만드세요.

m2_2d = m1.reshape(3, 4)
m2_batch = m2_2d.unsqueeze(0)
m2_squeezed = m2_batch.squeeze()

# [검증 2]
assert m2_2d.shape == (3, 4), f"m2_2d Shape 에러: {m2_2d.shape}"
assert m2_batch.shape == (1, 3, 4), f"m2_batch Shape 에러: {m2_batch.shape}"
assert m2_squeezed.shape == (3, 4), f"m2_squeezed Shape 에러: {m2_squeezed.shape}"

print("\n✅ MISSION 2 PASSED!")
print(f"  • 원본 Shape (1D): {m1.shape}")
print(f"  • 2D 변환 Shape  : {m2_2d.shape}")
print(f"  • 배치 차원 추가  : {m2_batch.shape}")
print(f"  • Squeeze 적용   : {m2_squeezed.shape}")


# ==========================================
# [MISSION 3] 하드웨어 장치(Device) 및 NumPy 연동
# ==========================================
# 목표:
# 1. 사용 가능한 디바이스(cuda 또는 cpu)를 `device` 변수에 정의하세요.
# 2. `m2_2d` 텐서를 해당 `device`로 이동시켜 `m3_dev`를 만드세요.
# 3. `m3_dev`를 다시 CPU로 가져온 뒤 NumPy 배열 `m3_np`로 변환하세요.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m3_dev = m2_2d.to(device)
m3_np = m3_dev.cpu().numpy()

# [검증 3]
assert str(m3_dev.device).startswith("cuda") or str(m3_dev.device) == "cpu"
assert isinstance(m3_np, np.ndarray), "NumPy 배열 변환 실패"
assert m3_np.shape == (3, 4), "NumPy 배열 Shape 에러"

print("\n✅ MISSION 3 PASSED!")
print(f"  • 감지 및 할당된 장치: {device}")
print(f"  • 텐서 디바이스 위치  : {m3_dev.device}")
print(f"  • NumPy 변환 완료 타입: {type(m3_np)}")


# ==========================================
# [MISSION 4] 인덱싱 및 슬라이싱 다루기
# ==========================================
# 목표: m2_2d 행렬에서 2행 전체(인덱스 1)와 3열 전체(인덱스 2)를 추출하세요.

row_1 = m2_2d[1, :]   # [5, 6, 7, 8]
col_2 = m2_2d[:, 2]   # [3, 7, 11]

# [검증 4]
assert torch.equal(row_1, torch.tensor([5, 6, 7, 8])), f"행 추출 실패: {row_1}"
assert torch.equal(col_2, torch.tensor([3, 7, 11])), f"열 추출 실패: {col_2}"

print("\n✅ MISSION 4 PASSED!")
print(f"  • 추출된 2번째 행: {row_1.tolist()}")
print(f"  • 추출된 3번째 열: {col_2.tolist()}")


print("\n" + "=" * 50)
print("🎉 ALL MISSIONS PASSED! 모든 텐서 기본 연산 검증 완료!")
print("=" * 50)