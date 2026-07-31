# 1. Official PyTorch 이미지 사용 (PyTorch + Python + 기본 라이브러리 포함)
FROM pytorch/pytorch:latest

# 2. 컨테이너 내부 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 파일 복사 및 설치 (캐시 활용 optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 전체 프로젝트 소스코드 복사
COPY . .

# 5. 기본 실행 명령어 (미션 1 실행)
CMD ["python", "mission_pytorch_basics.py"]