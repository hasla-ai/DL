##[3장 1강 심화] - 퍼셉트론 계산 실습
## 문제 1. 점수를 확률로 오해한 운영 로그 진단

# 검토 큐 label: 1 -> 담당자가 logit 1.0을 "위험 확률 100%"라고 보고했습니다. 실제 batch 점수와 예측을 계산하고 잘못된 해석을 바로잡아야.
## logits, preds, shape와 로그 정정 문장을 제출

# 1. 세 샘플의 가중합을 계산한다.
import torch
X = torch.tensor([[2., 1.], [1., 3.], [-1., 2.]])   # (3,2)
w = torch.tensor([0.8, -0.5])                       # (2, ) -> broadcasting (3,2)
b = -0.1                                            # float -> shape 없음.

z = (X @ w) + b
print("z      :", z)

# 2. 결과 shape를 확인한다.
print("z shape     :", tuple(z.shape)) # Shape 예상: (3,2)

# 3. 0 기준 label을 만든다.logit을 기준으로 0/1 예측값(pred)을 만드는 것

preds = (z >= 0).long()
print("preds       :", preds.tolist())

## logits, preds, shape와 로그 정정 문장을 제출
# 4. logit을 확률로 부를 수 없는 이유를 설명한다.


# 문제 2. shape 계약이 있는 퍼셉트론 함수 작성
## 입력 feature 수가 바뀌었을 때 조용히 잘못 계산되지 않도록 batch 전용 함수: 함수와 정상 batch의 출력 shape를 제출

# X = torch.tensor([[2., 1.], [1., 3.], [-1., 2.]])   # (3,2)
# w = torch.tensor([0.8, -0.5])                       # (2, ) -> broadcasting (3,2)
# b = -0.1                                            # float -> shape 없음.

def predict_batch(X, w, b):
    # 1. X는 2차원, w는 1차원인지 검사한다.
    assert X.ndim == 2 and w.ndim == 1, "expected (B,F) and (F,)"
     ## feature 수가 바뀌었는데 우연히 broadcasting으로 계산되는 상황을 잡을 수 있음.
    
    # 2. 마지막 feature 수가 같은지 검사한다.
    assert X.shape[-1] == w.shape[0], "feature/weight mismatch"
    ## 이 함수에 X와 w를 넣으려면 X의 feature 수와 w의 가중치 수가 같아야 한다.
    
    # 3. scalar bias만 허용한다.
    assert torch.tensor(b).ndim == 0 , "bias must be scalar"
    # 4. logits와 preds를 함께 반환한다.
    logits = (X * w) + b
    preds = (logits >= 0).long()
    return logits, preds

logits, preds = predict_batch(X, w, b)

print("logits shape:", logits.shape)
print("preds shape:", preds.shape)

# 문제 3. 오판 비용으로 bias 후보 선택
## 위험 요청 실패 비용 > 정상 요청 추가 검토 비용 * 5. 같은 weight에서 bias 두 개를 비교해 총 비용이 낮은 후보 선택.
### 후보별 비용과 selected bias를 제출
import torch
X = torch.tensor([[1., 0.], [0.2, 0.8], [0.6, 0.5]])
y = torch.tensor([1, 1, 0])
w = torch.tensor([1., -1.])
bias_candidates = [0.0, 0.7]

# 1. 후보별 예측을 계산한다.
scores = [X @ w + b for b in bias_candidates]

results=[]
for bias, z in zip(bias_candidates, scores):
    print("bias:", bias, "logits:", z)
# 2. false negative와 false positive를 센다.
    pred = (z > 0).long()
    fn = ((y == 1) & (pred == 0)).sum()
    fp = ((y == 0) & (pred == 1)).sum()
    print("bias:", bias, "FN:", fn.item(), "FP:", fp.item())

# 3. `5*FN + 1*FP`를 계산한다.  
    cost = 5 * fn + 1 * fp

    results.append({
        "bias": bias,
        "FN": fn.item(),
        "FP": fp.item(),
        "cost": cost.item()
    })
print(results)

selected = min(results, key=lambda r: r["cost"])

print("selected bias:", selected["bias"])
print("selected cost:", selected["cost"])

# 4. 선택과 선형 모델의 한계를 보고.