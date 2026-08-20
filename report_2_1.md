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





