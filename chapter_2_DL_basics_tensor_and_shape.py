# 2장 3강
#문제1.

# 모델 parameter와 입력·target device를 각각 읽어 최초 불일치 지점을 찾습니다.
# 오류 메시지 뒤 모든 Tensor를 무조건 CPU로 내리지 않고 실행 목표 device에 필요한 값만 이동합니다.
log = {"model": "cuda:0", "inputs": "cuda:0", "labels": "cpu", "offset": "cpu"}
model_device = log["model"]

# 실제 연산에 참여하며 모델과 다른 위치에 있는 항목만 이동 대상으로 잡습니다.
move = [name for name in ("inputs", "labels", "offset") if log[name] != model_device]
print("move_to_model_device:", move)
print("keep:", [name for name in ("model", "inputs") if log[name] == model_device])

# 문제 2.

import torch
import torch.nn as nn

# 1. 사용 가능한 device 결정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 모델을 한 번 이동한다.
model = nn.Linear(5,2).to(device)

#3. helper가 반환값을 새 변수에 저장한다.
x = torch.randn(8, 5)
labels = torch.randint(0, 2, (8,))

def move_batch(x, labels, device):
    return x.to(device), labels.to(device)
x_device, labels_device = move_batch(x, labels, device)

print("x device:", x_device.device)
print("labels device:", labels_device.device)

#4. 세 device가 모두 같은지 assert한다.
assert next(model.parameters()).device == x_device.device
assert x_device.device == labels_device.device

model_device = next(model.parameters()).device
assert model_device == x_device.device == labels_device.device
print("devices_equal:", model_device == x_device.device == labels_device.device)
print("output_shape:", tuple(model(x_device).shape))


## 문제 3. device 수정 뒤 재측정할 실행안 고르기

runs = {
    "A": {
        "model": "cpu",
        "input": "cpu",
        "label": "cpu",
        "mask": "cpu",
        "throughput": 42,
        "memory": 70,
    },
    "B": {
        "model": "cuda:0",
        "input": "cuda:0",
        "label": "cpu",
        "mask": "cuda:0",
        "throughput": 118,
        "memory": 24,
    },
    "C": {
        "model": "cuda:0",
        "input": "cuda:0",
        "label": "cuda:0",
        "mask": "cpu",
        "throughput": 130,
        "memory": 6,
    },
}


import torch
import torch.nn as nn
for name, run in runs.items():
    devices = [
        run["model"],
        run["input"],
        run["label"],
        run["mask"],
    ]
# 1. 모델·입력·label·새 mask의 device를 비교한다.
    same_device = len(set(devices)) == 1


# 2. device 일치안은 처리량 80건/초 조건을 검사한다.
    if same_device:
        # 2. device가 일치하는 경우에만 처리량 검사
        throughput_ok = run["throughput"] >= 80

        print(
            name,
            "device 일치 →",
            "처리량 조건:", throughput_ok
        )

# 3. 불일치안은 수정할 필드와 수정 뒤 메모리 여유 15% 조건을 확인한다.

    # 기준 device = model device
    target_device = run["model"]

    # model과 다른 device에 있는 필드 찾기
    fields = ["input", "label", "mask"]

    to_move = []

    for field in fields:
        if run[field] != target_device:
            to_move.append(field)

    print(f"{name}: 수정할 필드 =", to_move)

    # 수정 뒤 메모리 여유 조건
    memory_ok = run["memory"] >= 15

    print(f"{name}: 메모리 여유 15% 이상 =", memory_ok)


# 4. 다음 재측정 후보와 아직 배포 승인할 수 없는 이유를 쓴다.

