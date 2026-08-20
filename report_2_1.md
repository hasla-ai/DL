# [2장 3강 심화] - Device 이동 실습

## 실습 배경

루멘 팀은 개발 노트북의 CPU와 학습 서버의 GPU에서 같은 문서 분류 코드를 실행합니다. 서버 전환 뒤 모델과 입력은 GPU에 있었지만, 루프 안에서 새로 만든 보정 Tensor와 label이 CPU에 남아 device mismatch가 발생했습니다.
GPU 런타임을 켰다는 사실과 Tensor가 GPU에 있다는 사실은 다릅니다. 또한 Tensor의 `.to(device)`는 이동된 Tensor를 반환하므로 반환값을 저장하지 않으면 원래 Tensor가 그대로 남을 수 있습니다.
이번 페이지는 특정 장비 없이도 실행되도록 현재 환경에서 CPU/GPU를 선택하고, 모델·입력·정답·새 Tensor를 하나의 device 계약으로 맞춥니다.

## 실습 목표

- 실행 device와 Tensor 실제 device를 구분한다.
- 모델·입력·target을 같은 device로 이동한다.
- `.to(device)` 반환값 누락을 진단한다.
- CPU fallback에서도 동일 코드가 동작하도록 작성한다.

## 진행 방식

- `cuda`가 없으면 CPU를 정상 경로로 사용한다.
- 모델 device는 `next(model.parameters()).device`로 확인한다.
- 루프 안에서 만드는 Tensor는 기존 Tensor의 device를 따른다.

## 오늘의 업무 흐름

device 선택 → 모델 이동 → batch 이동 → 새 Tensor 위치 지정 → forward 전 동등성 검사

 ## 상황 자료

```
model device=cuda:0
inputs device=cuda:0
labels device=cpu
offset device=cpu
```

## 문제 1. 장애 로그의 최소 수정 범위 찾기

### 업무 요청

위 로그에서 forward와 loss 계산에 참여하는 객체 중 이동이 필요한 항목을 찾고, 모델을 CPU로 되돌리지 않는 최소 수정안을 보고하세요.

### 수행해야 할 작업

1. 모델 device를 기준으로 다른 항목을 비교한다.
2. 입력은 정상인지 판정한다.
3. labels와 offset의 생성/이동 위치를 수정한다.
4. GPU가 없는 환경의 fallback을 설명한다.

### 제출해야 할 결과

- 이동 대상 목록과 수정 패턴을 제출한다.

- 모델과 인풋 데이터가 cuda 즉 GPU에 있으므로(`model device=cuda:0`, `inputs device=cuda:0`)
CPU에 위치한 `labels device` 는 GPU에 함께 있어야 한다. 따라서 이동한다.
`offset device`는 이동하지 않는다.
## 모든 객체를 무조건 CPU로 모으면 오류는 사라질 수 있지만 서버의 GPU 사용 목적을 잃습니다.

```bash

move_to_model_device: ['labels', 'offset']
keep: ['model', 'inputs']

```

입력은 이미 모델과 같으므로 다시 이동할 필요가 없습니다. labels는 `labels = labels.to(device)`, offset은 `torch.zeros(..., device=inputs.device)` 또는 `zeros_like`로 만들어야 합니다. 전체를 CPU로 되돌리는 답은 장애만 피하고 GPU 실행 요구를 위반합니다. GPU가 없으면 처음부터 `device=cpu`를 선택해 같은 동등성 계약을 적용합니다.

**접근 순서:** 모델 device를 기준점으로 정하고 forward·loss에 실제 참여하는 입력, label, 보조 Tensor를 하나씩 비교합니다. 이미 일치한 입력은 유지하고 불일치한 label과 offset만 이동해 최소 수정 범위를 분명히 합니다.

**오답 원인:** 모든 객체를 CPU로 되돌리면 mismatch는 사라지지만 GPU 서버 사용 요구를 위반합니다. 런타임에서 GPU를 켰다는 사실만 보고 새 Tensor도 자동으로 GPU에 생긴다고 가정하는 답 역시 offset 문제를 남깁니다.

**적용 한계:** 로그 문자열 비교는 객체가 실제로 같은 GPU index와 dtype을 쓰는지 추가 확인이 필요합니다. device를 맞춰도 메모리 용량, 속도, 모델 정확도가 보장되는 것은 아니며 각각 별도 운영 검증을 거쳐야 합니다.

## 문제 2. 안전한 batch 이동 helper 작성하기

### 업무 요청

입력과 label을 함께 받아 지정 device로 옮기는 helper를 작성하고, forward 직전에 모델과 같은 device인지 검증하세요.

### 수행해야 할 작업

1. 사용 가능한 실행 device를 정한다.
2. 모델을 한 번 이동한다.
3. helper가 반환값을 새 변수에 저장한다.
4. 세 device가 모두 같은지 assert한다.

## x.to(device)만 호출하지 말고 반환된 Tensor를 받아야.

```bash
move_to_model_device: ['labels', 'offset']
keep: ['model', 'inputs']
x device: cuda:0
labels device: cuda:0
devices_equal: True
output_shape: (8, 2)
```

## `x.to(device)` 반환값을 버리지 않도록 주의해야.
## 모델이 batch loop 안에서 매번 이동하는지 확인해야.

**해설**

  대표 출력은 실행 장치 이름 대신 세 객체의 동등성만 고정하므로 CPU와 CUDA 환경에서 같습니다. 핵심은 장치 이름 자체가 아니라 모델·입력·label이 같은 장치에 있다는 계약입니다. helper가 중첩 사전까지 처리한다고 가정하면 안 되며, 현재 계약은 `(x,y)` Tensor 쌍에 한정됩니다. 모델 이동은 batch마다 반복하지 않고 학습 시작 전에 한 번 수행합니다.

**접근 순서:** 실행 환경에서 device를 한 번 결정하고 모델을 먼저 이동한 다음, batch helper가 x와 y의 반환 Tensor를 모두 새 변수에 저장하게 합니다. forward 직전에 모델 parameter·x·y의 device 동등성을 assert로 확인합니다.

**오답 원인:** `x.to(device)`를 호출만 하고 반환값을 버리면 원본 Tensor는 CPU에 남을 수 있습니다. 모델을 batch loop마다 이동하면 불필요한 비용이 생기며, label을 연산에 쓰지 않는다고 보고 이동을 생략하면 loss 단계에서 실패합니다.

**적용 한계:** helper는 Tensor 두 개로 된 batch만 지원합니다. dict나 중첩 tuple, 길이가 다른 목록을 쓰는 실제 LLM batch에는 재귀 처리 또는 명시적 필드 이동이 필요하며, 대표 출력의 cpu는 검증 환경 결과일 뿐 고정 요구가 아닙니다.

## 문제 3. device 수정 뒤 재측정할 실행안 고르기

### 업무 요청

세 실행안의 device 감사와 운영 측정이 함께 도착했습니다. device가 이미 일치한 안은 현재 처리량을 보고, 불일치 안은 수정 가능한 필드와 수정 뒤 예상 운영 여유를 근거로 `즉시 실행`, `수정 후 재측정`, `보류`로 나누세요. 수정 전 GPU 수치를 곧바로 승인 근거로 쓰면 안 됩니다.

### 수행해야 할 작업

1. 모델·입력·label·새 mask의 device를 비교한다.
2. device 일치안은 처리량 80건/초 조건을 검사한다.
3. 불일치안은 수정할 필드와 수정 뒤 메모리 여유 15% 조건을 확인한다.
4. 다음 재측정 후보와 아직 배포 승인할 수 없는 이유를 쓴다.

상황 자료

실행  model   input   label   mask     처리량/예상 처리량  메모리 여유
A     cpu     cpu     cpu     cpu            42               70%
B     cuda:0  cuda:0  cpu     cuda:0        118               24%
C     cuda:0  cuda:0  cuda:0  cpu          130                6%

A device 일치 → 처리량 조건: False
A: 수정할 필드 = []
A: 메모리 여유 15% 이상 = True
B: 수정할 필드 = ['label']
B: 메모리 여유 15% 이상 = True
C: 수정할 필드 = ['mask']
C: 메모리 여유 15% 이상 = False

A device 일치 → 처리량 조건: False
B → label을 cuda:0으로 이동 → 메모리 여유 24% → 조건 충족
C → mask를 cuda:0으로 이동 → 메모리 여유 6% → 조건 불충족

즉 3번까지 검토하면 B가 수정 후보로 살아남고 C는 탈락. 그러나 B는 아직 GPU 환경에서 검증되지 않았음.
B의 label을 모델 device로 옮긴 뒤 같은 batch와 장비에서 처리량과 peak memory를 다시 측정합니다. 수정 가능한 contract 오류와 운영 자원 부족을 분리해야 작업 순서를 합리적으로 정할 수 있습니다. 표의 GPU 처리량은 수정 뒤 예상값이므로 재측정 전에는 확정 자료가 아닙니다. B가 재측정에서 기준을 통과하면 그때 validation 품질까지 별도로 확인해 배포 판단으로 넘어갑니다.

**해설**

A는 device 계약은 맞지만 처리량 42로 운영 조건을 넘지 못합니다. B는 label만 CPU에 남았고 수정 뒤 예상 처리량과 메모리 여유가 기준을 만족하므로 가장 먼저 수정·재측정할 후보입니다. C는 mask 문제를 고쳐도 메모리 여유가 6%라 보류합니다. 예상 GPU 수치는 수정 전 실행의 배포 승인이 아니므로 즉시 실행 후보는 없습니다.

**다음 티켓:** B의 label을 모델 device로 옮긴 뒤 같은 batch와 장비에서 처리량과 peak memory를 다시 측정합니다. 수정 가능한 contract 오류와 운영 자원 부족을 분리해야 작업 순서를 합리적으로 정할 수 있습니다.

**왜 가장 빠른 C가 아닌가:** 처리량만 보면 C가 좋아 보이지만, device mismatch와 6% 메모리 여유가 남습니다. 반대로 현재 실행 가능한 A를 자동 선택하면 처리량 SLA를 숨깁니다. `실행 가능`, `수정 가능`, `운영 가능`을 서로 다른 상태로 기록해야 합니다.
ex) routes: {'A': '보류:throughput', 'B': '수정 후 재측정:label', 'C': '보류:memory_headroom'}
next_candidate: B

표의 GPU 처리량은 수정 뒤 예상값이므로 재측정 전에는 확정 자료가 아닙니다. B가 재측정에서 기준을 통과하면 그때 validation 품질까지 별도로 확인해 배포 판단으로 넘어갑니다.

# [2장 4강 심화] - 오류 수정 실습 (1)

## 실습 배경

루멘의 안전 문서 분류기에서 한 번에 세 오류가 보고됐습니다. 입력 feature가 모델의 `in_features`와 다르고, 다중 분류 target은 float이며, 서버 실행에서는 label만 CPU에 남았습니다. 마지막 traceback 한 줄만 고치면 다음 오류가 이어져 디버깅 시간이 길어집니다.

팀은 오류 메시지를 `shape`, `dtype`, `device` 세 범주로 나눈 뒤, 실패한 연산 직전 Tensor와 모델 기대값을 비교하는 절차를 표준으로 삼으려 합니다. 수정 뒤에도 같은 속성을 다시 출력해 계약이 회복됐는지 확인해야 합니다.

이번 페이지에서는 증상별 한 줄 수정이 아니라 재사용 가능한 triage 순서와 batch 준비 함수를 만듭니다.

## 실습 목표

- traceback을 shape·dtype·device 문제로 분류한다.
- `nn.Linear` 입력 마지막 차원과 `in_features`를 비교한다.
- 다중 분류 target shape/dtype을 복구한다.
- 수정 후 속성 재검증을 자동화한다.

## 진행 방식

- 첫 실패 연산부터 고치되 나머지 계약 위반도 목록으로 남긴다.
- 입력 feature를 버릴지 모델을 바꿀지는 데이터 계약을 근거로 결정한다.
- 이번 상황에서는 실제 feature가 5개이므로 모델을 5개 입력으로 맞춘다.

## 오늘의 업무 흐름

오류 마지막 줄 읽기 → 범주 분류 → 직전 Tensor 감사 → 기대/실제 비교 → 수정 후 재감사

## 상황 자료

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x5 and 4x3)
다음 재실행: expected scalar type Long but found Float
서버 재실행: tensors on cuda:0 and cpu
```

## 문제 1. 연쇄 오류를 한 번에 triage하기

### 업무 요청

모델 기대값과 batch 메타데이터로 현재 잠재 오류를 모두 나열하고, 실제 실행에서 처음 드러날 오류를 지정하세요.

### 수행해야 할 작업

1. 입력 마지막 차원과 `in_features`를 비교한다.
2. target shape와 dtype을 다중 분류 계약과 비교한다.
3. 모델·입력·target device를 비교한다.
4. 실행 순서상 첫 실패를 보고한다.

시작 코드

```bash
meta = {"x_shape": (4, 5), "target_shape": (4,), "target_dtype": "float32",
        "model_in": 4, "model_device": "cuda:0", "x_device": "cuda:0", "target_device": "cpu"}
```

input features: 5
model in_features: 4
shape ERROR
target shape: (4,)
target dtype: float32
shape OK: True
dtype OK: False
model device: cuda:0
x device: cuda:0
target device: cpu
device OK: False

RuntimeError: mat1 and mat2 shapes cannot be multiplied
다음 재실행: expected scalar type Long but found Float
서버 재실행: tensors on cuda:0 and cpu
## 실제 실행에서는 shape 오류가 가장 먼저 발생. ##  
  1차 실행 이후 shape 검사 ❌ 여기서 실패. dtype 검사까지 도달하지 못함, device 검사까지 도달하지 못함.

**해설**
ex) issues: ['shape:model_input', 'dtype_or_shape:target', 'device:mismatch']
first_failure: shape:model_input

## forward에서 shape가 먼저 사용되므로 첫 표면 오류가 됩니다.##
  forward가 입력 shape를 먼저 소비하므로 Linear 오류가 가장 먼저 납니다. 이것만 고치면 loss 단계에서 target dtype 또는 device 문제가 드러납니다. 따라서 수정은 모델/입력 계약, target dtype·shape, device 순으로 진행하되 세 항목을 처음부터 목록화합니다. traceback 순서는 코드 흐름에 따라 달라질 수 있으므로 범주 목록 자체가 더 일반적인 도구입니다.

**접근 순서:** 파이프라인 실행 순서에 맞춰 forward의 입력 shape, loss의 target 계약, device 동등성을 모두 목록화합니다. 첫 표면 오류는 shape지만 뒤 잠재 오류도 같은 티켓에 남겨 한 번의 수정 뒤 재실행을 반복하는 시간을 줄입니다.

**오답 원인:** traceback 마지막 줄 하나만 보고 수정하면 다음 단계의 dtype·device 문제가 연달아 나타납니다. 실제 feature 5개 중 하나를 임의로 잘라 model_in 4에 맞추는 답은 데이터 계약과 정보를 변경하는 별도 의사결정입니다.

**적용 한계:** 첫 실패 순서는 현재 코드가 forward 후 loss를 호출한다는 전제에 따릅니다. 사전 device 검사나 다른 분기가 있으면 순서가 달라질 수 있으므로 범주 목록은 유지하되 실제 호출 경로에서 최초 실패를 다시 확인해야 합니다.


## 문제 2. 계약을 복구하는 `prepare_batch` 작성

### 업무 요청

실제 데이터는 feature 5개, class 3개의 다중 분류입니다. 입력은 float32, target은 1차원 long, 두 Tensor는 모델 device에 있도록 준비 함수를 작성하세요.

### 수행해야 할 작업

1. 모델을 `Linear(5,3)`으로 만든다.
2. 입력을 float32로 변환한다.
3. target을 `(B,)` long으로 정규화한다.
4. device와 shape를 assert하고 forward를 실행한다.

# prepare_batch 입구에서 sample 수·feature 수·label dtype을 확인한 뒤 지정 device로 함께 이동합니다.
# 반환 shape와 device를 다시 assert해 helper 내부 수정이 실제 모델 계약과 연결됐는지 검증합니다.

torch.float32
x shape: torch.Size([4, 5])
target shape: torch.Size([4])
out shape: torch.Size([4, 3])
logits: (4, 3) torch.float32 cpu
target: (4,) torch.int64 cpu

**해설**
## target을 float로 유지한 채 소수점 이하를 절삭하는 등 값만 정수처럼 보이게 두지 않도록 유의.

데이터 정의상 feature가 5개이므로 모델을 그 계약에 맞춥니다. 변환 뒤에 assert를 두어 reshape가 잘못된 원소 수를 조용히 숨기지 못하게 합니다. 대표 출력은 CPU 검증 환경 기준이며 CUDA에서도 동등성 계약은 같습니다. 이 helper는 다중 분류용이므로 회귀나 BCE target에 그대로 쓰면 안 됩니다.

**접근 순서:** 데이터 정의가 feature 5개임을 먼저 확정해 모델을 Linear(5,3)으로 만들고, 입력 float·target 1차원 long·공통 device 순서로 정규화합니다. 변환 직후 모든 계약을 assert한 뒤에만 forward를 호출합니다.

**오답 원인:** reshape(-1)는 원소 수가 잘못된 target도 길이만 맞아 보이게 할 수 있으므로 batch 크기 재검사가 필요합니다. 모델 device를 cuda 문자열로 하드코딩하면 CPU fallback과 다른 GPU index에서 다시 실패합니다.

**적용 한계:** 이 prepare_batch는 기본 다중 클래스용입니다. 회귀나 BCE target을 long으로 바꾸면 의미가 틀리고, feature 5개의 열 순서와 값 범위까지는 검증하지 않으므로 데이터 schema 감사가 별도로 필요합니다.

## 문제 3. 장애 티켓 우선순위 정하기

### 업무 요청

세 로그는 **같은 commit과 같은 요청 `req-742`의 단일 순차 trace**에서 수집됐습니다. 동일한 인력 한 명이 처리할 때 먼저 재현할 티켓을 "가장 이른 실패 + 다른 오류를 가릴 가능성" 기준으로 정렬하세요.

### 수행해야 할 작업

1. 각 오류 범주를 지정한다.
2. 로그의 연산명으로 pipeline 위치를 판단한다.
3. 앞 실패가 뒤 오류를 가리는 순서로 정렬한다.
4. 우선순위가 사업 중요도와 같지는 않다는 한계를 쓴다.

### 상황 자료

```
T1 forward: mat1 32x768, mat2 512x4
T2 loss: expected Long but found Float
T3 evaluation: cpu/cuda mismatch
```
결과

T1 → shape mismatch
T2 → dtype mismatch
T3 → device mismatch

T1 → 모델 추론 단계
T2 → 손실 계산 단계
T3 → 평가 단계

T2 depends on ['T1']
T3 depends on ['T2']
## 해답: order: ['T1', 'T2', 'T3']

다만 엄밀하게는 evaluation이 반드시 loss에 의존한다고 로그만으로 확정할 수는 없음.
단일순차 trace가 아니면 evaluation이 별도의 forward를 다시 수행하는 구조라면 T3는 T2에 직접 의존하지 않을 수도.

# 각 티켓의 실행 차단 여부와 silent corruption 위험을 분리해 우선순위를 계산합니다.
# 오류 메시지가 크다는 이유가 아니라 영향 범위와 탐지 가능성을 근거로 조사 순서를 정합니다.

**해설**

이 정렬은 코드 경로를 복구하는 디버깅 순서입니다. T1을 고쳐야 T2가 재현되고, 학습이 지나야 T3 평가 오류까지 확인할 수 있습니다. 그러나 개인정보 노출 같은 사업 위험이 연결돼 있다면 발생 단계가 늦어도 더 높은 대응 우선순위를 가질 수 있습니다. 여기서는 기술적 재현 순서만 결정합니다.

**접근 순서:** 각 티켓을 실제 연산 단계와 shape·dtype·device 범주로 먼저 태깅합니다. 그 다음 앞 단계 실패가 뒤 단계를 가릴 수 있다는 기준으로 forward, loss, evaluation 순서로 재현해 코드 경로를 단계적으로 복구합니다.

**오답 원인:** 메시지가 가장 길거나 발생 시간이 늦은 티켓을 먼저 고르면 앞 오류 때문에 동일 경로를 재현하지 못할 수 있습니다. 기술적 실행 순서를 사업 영향 우선순위와 같은 것으로 보고하는 것도 목적을 혼동한 답입니다.

**적용 한계:** 여기서 정한 것은 한 엔지니어의 디버깅 재현 순서입니다. 개인정보 노출이나 서비스 중단처럼 위험도가 큰 티켓은 별도 사고 대응 우선순위가 더 높을 수 있으며, 실제 운영에서는 두 기준을 함께 표기해야 합니다.

디버깅 순서는 순차 의존성 때문에: T1 forward → shape 오류, T2 loss    → dtype 오류, T3 eval    → device 오류.
(T1 → T2 → T3).
그런데 사업 우선순위는 전혀 다른 기준임. 예를 들어: T1: 특정 고객의 핵심 기능이 아예 실행되지 않음, T2: 내부 학습 작업에서만 발생,
T3: 평가 환경에서만 발생, 이라면 사업적으로는:T1 > T3 > T2가 될 수도 있고, 반대로 T3가 고객에게 잘못된 평가 결과를 제공하는 치명적 문제라면: T3 > T1 > T2가 될 수도 있음.

  즉, 디버깅 순서는 기술적 의존성에 따라 순차적으로 검토해야 하나 사업 우선순위는 영향도, 고객, 매출, SLA,  위험 등을 종합적으로 고려하여 개별적으로 판단하여야 한다.