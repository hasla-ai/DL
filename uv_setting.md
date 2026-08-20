$ uv python pin 3.12
Pinned `.python-version` to `3.12`

hrjeo@JEON MINGW64 ~/DL (main)
$ uv venv
source .venv/Scripts/activate

# 3. PyTorch 설치 (아까 다운받은 캐시에서 3초 만에 복사됨)
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install matplotlib

주피터 노트북인 경우
$ uv pip install ipykernel
$ python -m ipykernel install --user --name dl-venv --display-name "Python (DL .venv)"

C:\Users\hrjeo\DL\.venv\Scripts\python.exe가 Interpreter 목록에 아예 안 보인다면, .venv를 다시 만들 필요는 없다.

Ctrl + Shift + P
→ Python: Select Interpreter
→ Enter interpreter path...
→ Find...
로 들어가서:
C:\Users\hrjeo\DL\.venv\Scripts\python.exe


Notebook Kernel
     ↓
DL .venv             ✅
     ↓
import torch         ✅ 실제 실행

VS Code Interpreter
     ↓
다른 Python          ⚠️
     ↓
import torch 빨간 점


                 C:\Users\hrjeo\DL\.venv\Scripts\python.exe
                              │
                 ┌────────────┴────────────┐
                 ↓                         ↓
       Python Interpreter             Jupyter Kernel
          (Pylance)                   (Notebook)
                 │                         │
                 └────────────┬────────────┘
                              ↓
                         PyTorch