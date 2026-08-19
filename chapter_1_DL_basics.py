
# 2-1 적용판단 도우미: 회의 반복 주제 체크리스트 만들기

# 검증 가능 정답 코드
# 샘플 수·규칙 유지비·문맥 의존도를 각각 검사해 딥러닝 적용 여부를 한 조건으로 단정하지 않습니다.
# 입력 숫자는 팀의 수업용 기준과 비교하고 결과는 배포 승인이 아니라 다음 실험 권고로 제한합니다.
def recommend_start(raw_unstructured, labeled_count, stable_rule):
    # 명시적이고 안정적인 규칙은 데이터 양과 무관하게 먼저 보존합니다.
    if stable_rule:
        return "rule_first"
    # 원본 텍스트·이미지처럼 표현 설계가 어려우면서 검증 가능한 라벨이 있을 때만 후보로 올립니다.
    if raw_unstructured and labeled_count >= 5000:
        return "deep_learning_candidate"
    # 데이터가 적거나 구조화 입력이면 단순 기준선을 먼저 만들어 비교 기준을 남깁니다.
    return "simple_baseline_first"

cases = [(False, 400, True), (True, 28000, False), (True, 120, False)]
print([recommend_start(*case) for case in cases])

# 3-1 불완전한 검증 근거로 승인과 재측정 구분

candidates = {
    "A": {"accuracy": 0.884, "latency_ms": 3, "latency_runs": 3},
    "B": {"accuracy": 0.921, "latency_ms": 10, "latency_runs": 1},
    "C": {"accuracy": 0.908, "latency_ms": 13, "latency_runs": 3},
}
approved, remeasure, rejected = [], [], {}
for name, result in candidates.items():
    failed = []
    if result["accuracy"] < 0.90:
        failed.append("accuracy")
    if result["latency_ms"] > 12:
        failed.append("latency")
    if failed:
        rejected[name] = failed
    elif result["latency_runs"] < 3:
        remeasure.append(name)
    else:
        approved.append(name)
decision = max(approved, key=lambda n: candidates[n]["accuracy"]) if approved else "보류"
print("approved:", approved)
print("remeasure:", remeasure)
print("rejected:", rejected)
print("decision:", decision)


## 1-3강 학습 파이프라인 순서
# 문제 1. 뒤섞인 실행 기록 진단하기

observed = ["forward", "loss", "step", "zero_grad", "backward"]

# 전체 목록을 한 줄 순서로 강제하지 않고, 실제로 값이 필요한 의존 관계만 검사합니다.

dependencies = [
    ("forward", "loss"),
    ("loss", "backward"),
    ("zero_grad", "backward"),
    ("backward", "step"),
]

position = {stage: observed.index(stage) for stage in observed}

violations = [
    f"{left} must precede {right}"
    for left, right in dependencies
    if position[left] > position[right]
]

print("first_violation:", violations[0])
print("recommended:", "zero_grad -> forward -> loss -> backward -> step")