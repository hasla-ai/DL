$ uv python pin 3.12
Pinned `.python-version` to `3.12`

hrjeo@JEON MINGW64 ~/DL (main)
$ uv venv
source .venv/Scripts/activate

# 3. PyTorch 설치 (아까 다운받은 캐시에서 3초 만에 복사됨)
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128