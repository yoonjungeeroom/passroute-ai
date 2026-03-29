from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import os
import whisper
import librosa
import numpy as np
import warnings
import uuid

# 경고 메시지 무시
warnings.filterwarnings("ignore")

app = FastAPI()

# CORS 설정: 브라우저의 접속을 허용합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# [수정] Whisper 'turbo' 모델 로드 (한국어 성능이 비약적으로 향상됩니다)
print("🚀 AI 모델(Whisper Turbo) 로딩 중... 잠시만 기다려주세요.")
model = whisper.load_model("turbo")
print("✅ AI 모델 준비 완료!")

UPLOAD_DIR = "temp_videos"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def analyze_audio(file_path):
    try:
        # turbo 모델을 사용하여 텍스트 추출
        result = model.transcribe(file_path, language="ko", fp16=False)
        text = result['text']
        
        # 음성 지표 추출 (목소리 높이 분석)
        y, sr = librosa.load(file_path)
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        print(f"\n[실시간 조각 분석 결과 - Turbo]")
        print(f"인식된 문장: {text}")
        print(f"목소리 높이: {avg_pitch:.2f} Hz")
        return text, avg_pitch
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
        return "", 0

@app.websocket("/ws/interview")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ 클라이언트 연결 성공!")
    
    try:
        while True:
            # 5초 단위 데이터 수신
            data = await websocket.receive_bytes()
            chunk_filename = f"chunk_{uuid.uuid4()}.webm"
            chunk_path = os.path.join(UPLOAD_DIR, chunk_filename)
            
            # 조각 파일 저장
            async with aiofiles.open(chunk_path, 'wb') as f:
                await f.write(data)
            
            print(f"📥 데이터 수신 완료 -> 분석 중...")
            analyze_audio(chunk_path)
            
            # 분석 후 즉시 삭제
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    except Exception as e:
        print(f"🔌 연결 종료 또는 에러: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass