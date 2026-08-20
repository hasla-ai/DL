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


# 2장 4강

# 문제 1.

meta = {"x_shape": (4, 5), "target_shape": (4,), "target_dtype": "float32",
        "model_in": 4, "model_device": "cuda:0", "x_device": "cuda:0", "target_device": "cpu"}

# 1. 입력 마지막 차원과 in_features 비교
input_features = meta["x_shape"][-1]
model_in = meta["model_in"]

print("input features:", input_features)
print("model in_features:", model_in)

if input_features == model_in:
    print("shape OK")
else:
    print("shape ERROR")
# 2. target shape와 dtype을 다중 분류 계약과 비교한다. # model output : (batch, num_classes)

target_shape = meta["target_shape"]                 # target : (batch,)
target_dtype = meta["target_dtype"]                 # target_dtype : long

shape_ok = len(target_shape) == 1
dtype_ok = target_dtype == "int64"

print("target shape:", target_shape)
print("target dtype:", target_dtype)
print("shape OK:", shape_ok)
print("dtype OK:", dtype_ok)

# 3. 모델·입력·target device를 비교한다.

model_device = meta["model_device"]
x_device = meta["x_device"]
target_device = meta["target_device"]

same_device = (
    model_device == x_device == target_device
)

print("model device:", model_device)
print("x device:", x_device)
print("target device:", target_device)
print("device OK:", same_device)

## 문제 2. 계약을 복구하는 `prepare_batch` 작성


#1. 모델을 `Linear(5,3)`으로 만든다.

import torch
import torch.nn as nn

model = nn.Linear(5, 3)

print(model)
print("in_features:", model.in_features)
print("out_features:", model.out_features)

#2. 입력을 float32로 변환한다.
x = torch.tensor([
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8]
    ], dtype=torch.float32)

x = x.to(torch.float32)

print(x)
print(x.dtype)

#3. target을 `(B,)` long으로 정규화한다.
target = torch.tensor([0., 2., 1., 0.])
target = target.long().reshape(-1)

#4. device와 shape를 assert하고 forward를 실행한다.
assert x.device == next(model.parameters()).device
assert target.device == x.device
assert x.shape[-1] == 5
assert target.ndim == 1

out = model(x)

print("x shape:", x.shape)
print("target shape:", target.shape)
print("out shape:", out.shape)

logits = model(x)
print("logits:", tuple(logits.shape), logits.dtype, str(logits.device))
print("target:", tuple(target.shape), target.dtype, str(target.device))



## 문제 3. 장애 티켓 우선순위 정하기

trace = {
    "T1": "forward: mat1 32x768, mat2 512x4",  
    "T2": "loss: expected Long but found Float",
    "T3": "evaluation: cpu/cuda mismatch",
}

# 1. 각 오류 범주를 지정한다.

categories = {
    "T1": "shape mismatch",
    "T2": "dtype mismatch",
    "T3": "device mismatch",
}

for trace_id in trace:
    print(trace_id, "→", categories[trace_id])


# 2. 로그의 연산명으로 pipeline 위치를 판단한다.
logs = {
    "T1": "forward",
    "T2": "loss",
    "T3": "evaluation",
}

pipeline = {
    "forward": "모델 추론 단계",
    "loss": "손실 계산 단계",
    "evaluation": "평가 단계",
}

for name, operation in logs.items():
    print(name, "→", pipeline[operation])

#  3. 앞 실패가 뒤 오류를 가리는 순서로 정렬한다.

# 순서 의존성 명시: T1이 실패하면 T2가 실행되지 않을 가능성이 있고, T2가 실패하면 T3가 실행되지 않을 가능성이 있다.
dependencies = {
    "T2": ["T1"],   # loss는 forward 결과에 의존
    "T3": ["T2"],   # evaluation은 loss 이후 단계라고 가정
}

for task, depends_on in dependencies.items():
    print(task, "depends on", depends_on)

print("\n재현 우선순위: T1 → T2 → T3")




