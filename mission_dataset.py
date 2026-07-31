import torch
from torch.utils.data import Dataset, DataLoader

print("=" * 50)
print("🚀 PyTorch 핸즈온 미션 6탄: Custom Dataset & DataLoader 시작")
print("=" * 50)

# ==========================================
# [MISSION 6-1] Custom Dataset 클래스 구현
# ==========================================
# 목표:
# 1. torch.utils.data.Dataset을 상속받는 SimpleDataset 클래스를 정의합니다.
# 2. __init__: 입력 데이터 X(특징)와 y(라벨)를 텐서로 보관합니다.
# 3. __len__: 데이터셋의 전체 샘플 개수를 반환합니다.
# 4. __getitem__: 주어진 인덱스(idx)에 해당하는 (X[idx], y[idx]) 튜플을 반환합니다.

class SimpleDataset(Dataset):
    def __init__(self, X, y): # Lazy Loading: 데이터가 매우 크다면 이름만 저장하고 파일을 읽도록 설계 가능함.
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx): # 데이터 실시간 변형(Augmentation), 이미지 파일의 텐서화 등이 가능함.
        return self.X[idx], self.y[idx] # 보통 데이터 회전, 밝기 조절하는 transform 인자 추가함.

# 테스트용 더미 데이터 생성 (100개의 샘플, 각 샘플은 10개 특징)
dummy_X = torch.randn(100, 10)
dummy_y = torch.randint(0, 2, (100,))  # 0 또는 1의 이진 분류 라벨

dataset = SimpleDataset(dummy_X, dummy_y)

# [검증 6-1]
assert len(dataset) == 100, f"Dataset 길이 불일치: {len(dataset)}"
sample_x, sample_y = dataset[0]
assert sample_x.shape == (10,), f"샘플 X Shape 불일치: {sample_x.shape}"
assert sample_y.ndim == 0, f"샘플 y 스칼라 여부 불일치: {sample_y.ndim}"

print("\n✅ MISSION 6-1 PASSED!")
print(f"  • 데이터셋 전체 샘플 수: {len(dataset)}")
print(f"  • 첫 번째 샘플 X Shape : {sample_x.shape}, y 값: {sample_y.item()}")


# ==========================================
# [MISSION 6-2] DataLoader를 통한 배치(Batch) 추출 및 셔플 검증
# ==========================================
# 목표:
# 1. batch_size=16, shuffle=True로 설정된 DataLoader를 생성하세요.
# 2. 첫 번째 배치(Batch)를 꺼내어 배치 차원이 올바르게 생성되었는지 검증하세요.

dataloader = DataLoader(dataset, batch_size=16, shuffle=True) # Batching-Shuffling-Paralle Computing
# num_workers: number of DataLoaders as Worker.

# 첫 번째 배치 가져오기
first_batch_x, first_batch_y = next(iter(dataloader))

# [검증 6-2] Dataset[0]: (10,) 1차원 벡터가 출력 (16,10) 2차원 행렬(뭉치)로 배치 차원화하는 것이 PyTorch 데이터 처리의 핵심.
# 전체 100개 데이터 중 batch_size=16이면 첫 배치의 크기는 (16, 10), (16,) 이어야함
assert first_batch_x.shape == (16, 10), f"배치 X Shape 불일치: {first_batch_x.shape}"
assert first_batch_y.shape == (16,), f"배치 y Shape 불일치: {first_batch_y.shape}"

print("\n✅ MISSION 6-2 PASSED!")
print(f"  • 배치 X Shape: {first_batch_x.shape} (Batch Size x Feature Size)")
print(f"  • 배치 y Shape: {first_batch_y.shape}")


# ==========================================
# [MISSION 6-3] Epoch 루프에서 전체 배치 개수 및 드롭라스트 검증
# ==========================================
# 목표:
# 100개의 데이터셋을 batch_size=16으로 순회하면:
# 16 * 6 = 96개 + 마지막 자투리(Epoch) 4개 = 총 7개의 배치가 순회되어야 함을 검증하세요.
# 실무적으로 drop_last=True를 하기도.
total_batches = 0
last_batch_size = 0

for bx, by in dataloader:
    total_batches += 1
    last_batch_size = len(bx)

# [검증 6-3]
assert total_batches == 7, f"총 배치 수 불일치: {total_batches}"
assert last_batch_size == 4, f"마지막 자투리 배치 크기 불일치: {last_batch_size}"

print("\n✅ MISSION 6-3 PASSED!")
print(f"  • 1 Epoch 동안 생성된 총 배치 수: {total_batches}")
print(f"  • 마지막 자투리 배치의 크기      : {last_batch_size}")

print("\n" + "=" * 50)
print("🎉 ALL MISSIONS IN MISSION 6 PASSED! Dataset & DataLoader 검증 완료!")
print("=" * 50)