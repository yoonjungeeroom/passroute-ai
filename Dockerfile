# Stage 1: ONNX 변환 + 양자화 (PyTorch는 이 스테이지에서만 사용)
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    "optimum[onnxruntime]" transformers

RUN optimum-cli export onnx \
    --model snunlp/KR-SBERT-V40K-klueNLI-augSTS \
    /tmp/onnx_model/ \
    --task feature-extraction

RUN python -c "\
from onnxruntime.quantization import quantize_dynamic, QuantType; \
quantize_dynamic('/tmp/onnx_model/model.onnx', '/tmp/kr-sbert-uint8.onnx', weight_type=QuantType.QUInt8)"

RUN python -c "\
from transformers import AutoTokenizer; \
t = AutoTokenizer.from_pretrained('snunlp/KR-SBERT-V40K-klueNLI-augSTS'); \
t.save_pretrained('/tmp/tokenizer/')"

# Stage 2: 런타임 (PyTorch 미포함, 이미지 경량화)
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ONNX 양자화 모델 + 토크나이저 복사
COPY --from=builder /tmp/kr-sbert-uint8.onnx /app/models/kr-sbert-uint8.onnx
COPY --from=builder /tmp/tokenizer/ /app/models/tokenizer/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# AWS RDS SSL 인증서 다운로드
RUN curl -o /app/global-bundle.pem https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem

# MediaPipe FaceLandmarker 모델 다운로드
RUN curl -L -o /app/models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
