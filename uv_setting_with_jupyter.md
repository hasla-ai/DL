# VS Code + uv + Jupyter + PyTorch 환경 설정

Windows에서 `uv`로 `.venv`를 만들고, VS Code의 Jupyter Notebook에서 PyTorch + CUDA를 사용하는 환경 설정 방법.

---

## 1. 프로젝트 구조

프로젝트:

```text
C:\Users\hrjeo\DL
```

가상환경:

```text
C:\Users\hrjeo\DL\.venv
```

Notebook(`.ipynb`) 파일은 `Downloads` 등 다른 폴더에 있어도 된다.

---

## 2. 가상환경 활성화

VS Code의 Git Bash 터미널에서:

```bash
cd ~/DL
source .venv/Scripts/activate
```

정상적으로 활성화되면:

```text
(DL)
```

이 표시된다.

Python 경로 확인:

```bash
python -c "import sys; print(sys.executable)"
```

정상적인 결과:

```text
C:\Users\hrjeo\DL\.venv\Scripts\python.exe
```

---

## 3. PyTorch 설치

CUDA 12.8 Nightly:

```bash
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

설치 확인:

```bash
python -c "print('START'); import torch; print('TORCH OK'); print(torch.__version__)"
```

정상적인 결과 예:

```text
START
TORCH OK
2.12.0.dev20260408+cu128
```

---

## 4. Jupyter Kernel 설치

### 중요

`uv` 가상환경을 만들었다고 해서 VS Code/Jupyter Notebook이 자동으로 해당 환경을 Kernel로 사용하는 것은 아니다.

먼저 `ipykernel`을 설치한다.

```bash
uv pip install ipykernel
```

그 다음 현재 `.venv`를 Jupyter Kernel로 등록한다.

```bash
python -m ipykernel install --user --name dl-venv --display-name "Python (DL .venv)"
```

성공하면 VS Code Notebook의 Kernel 목록에서:

```text
Python (DL .venv)
```

를 선택할 수 있다.

---

## 5. VS Code Notebook에서 Kernel 선택

`.ipynb` 파일을 열고 오른쪽 위의 Kernel 선택 메뉴에서:

```text
Python (DL .venv)
```

를 선택한다.

다음과 같은 일반 Python을 선택하는 것과 구별한다.

```text
Python 3.12.13
```

핵심은 Notebook이 다음 Python을 사용하도록 하는 것이다.

```text
C:\Users\hrjeo\DL\.venv\Scripts\python.exe
```

---

## 6. Notebook에서 Python 환경 확인

첫 번째 셀에서:

```python
import sys

print(sys.executable)
```

정상적인 결과:

```text
C:\Users\hrjeo\DL\.venv\Scripts\python.exe
```

---

## 7. PyTorch 확인

```python
import torch

print(torch.__version__)
print(torch.__file__)
```

예상 결과:

```text
2.12.0.dev20260408+cu128
C:\Users\hrjeo\DL\.venv\Lib\site-packages\torch\__init__.py
```

---

## 8. CUDA 확인

```python
print("CUDA available:", torch.cuda.is_available())

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("선택된 device:", device)
```

GPU가 정상적으로 인식되면:

```text
CUDA available: True
선택된 device: cuda
```

---

## 9. Matplotlib 설치

필요한 경우:

```bash
uv pip install matplotlib
```

Notebook:

```python
import matplotlib.pyplot as plt
```

---

# 문제 발생 시 확인 순서

## PyTorch import가 안 될 때

먼저 Notebook이 아니라 **터미널에서** 확인한다.

```bash
python -c "print('START'); import torch; print('TORCH OK'); print(torch.__version__)"
```

### 정상

```text
START
TORCH OK
2.12.0.dev20260408+cu128
```

터미널에서는 정상인데 Notebook에서만 문제가 발생한다면 **PyTorch 재설치부터 하지 않는다.**

Jupyter Kernel을 확인한다.

---

## Notebook Kernel이 계속 `RESTARTING`될 때

다음 순서로 확인한다.

### 1. `.venv` 활성화

```bash
cd ~/DL
source .venv/Scripts/activate
```

### 2. Python 경로 확인

```bash
python -c "import sys; print(sys.executable)"
```

### 3. ipykernel 설치

```bash
uv pip install ipykernel
```

### 4. Kernel 재등록

```bash
python -m ipykernel install --user --name dl-venv --display-name "Python (DL .venv)"
```

### 5. VS Code에서 다시 Kernel 선택

```text
Python (DL .venv)
```

---

# 핵심 개념

```text
uv
 │
 ▼
.venv
 │
 ├── Python
 ├── PyTorch
 ├── CUDA PyTorch
 ├── matplotlib
 └── ipykernel
       │
       ▼
Jupyter Kernel
       │
       ▼
VS Code Notebook
```

가장 중요한 명령:

```bash
cd ~/DL
source .venv/Scripts/activate
uv pip install ipykernel
python -m ipykernel install --user --name dl-venv --display-name "Python (DL .venv)"
```

그리고 VS Code Notebook에서는:

```text
Python (DL .venv)
```

를 선택한다.

**Notebook 파일의 위치와 `.venv`의 위치는 달라도 된다.**

예:

```text
C:\Users\hrjeo\Downloads\study.ipynb
                │
                │ 실행
                ▼
C:\Users\hrjeo\DL\.venv\Scripts\python.exe
```

이 구성은 정상이다.
