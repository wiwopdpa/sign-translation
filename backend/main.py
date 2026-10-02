import os
import json
import numpy as np
import tensorflow as tf
import tf_keras as keras
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any

app = FastAPI(
    title="Real-time Sign Language Recognition API",
    description="3D Hand Keypoint 기반 수어 인식 백엔드 서버",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "my_model.h5 ")

ACTIONS = np.array([
    "안녕하세요", "감사합니다", "미안합니다", "만나다", "반갑다",
    "사람", "집", "학교", "음식", "물",
    "좋다", "나쁘다", "크다", "작다", "가다",
    "오다", "먹다", "마시다", "주다", "받다"
])


try:
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        print(f"✅ [SUCCESS] TensorFlow Keras 모델이 성공적으로 로드되었습니다: {MODEL_PATH}")
    else:
        model = None
        print(f"⚠️ [WARNING] {MODEL_PATH} 파일이 없습니다. 모델 파일 경로를 확인해주세요.")
except Exception as e:
    model = None
    print(f"❌ [ERROR] 모델을 로드하는 중 오류가 발생했습니다: {e}")


sequence_buffer = []


class FrameInput(BaseModel):
    # 왼손(63) + 오른손(63) = 126차원 키포인트 배열
    keypoints: List[float] = Field(..., description="126차원 (왼손 63 + 오른손 63) 키포인트 리스트")

class PredictionResponse(BaseModel):
    status: str
    action: str
    confidence: float
    is_recognized: bool
    current_frames: int



@app.get("/")
def read_root():
    """서버 상태 확인용 헬스체크 엔드포인트"""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "input_shape": "(1, 30, 126)",
        "num_classes": len(ACTIONS)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_sign_language(data: FrameInput):
    """
    프론트엔드/웹캠에서 전달받은 126차원 프레임을 수집하고,
    30프레임이 쌓이면 LSTM 모델로 수어 동작을 추론합니다.
    """
    global sequence_buffer

    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="모델이 로드되지 않았습니다. 백엔드 서버의 model.h5 파일 경로를 확인하세요."
        )

    keypoints = data.keypoints

    
    if len(keypoints) != 126:
        raise HTTPException(
            status_code=400,
            detail=f"키포인트 길이는 반드시 126이어야 합니다. (입력된 차원: {len(keypoints)})"
        )

    
    sequence_buffer.append(keypoints)
    sequence_buffer = sequence_buffer[-30:]

   
    if len(sequence_buffer) < 30:
        return PredictionResponse(
            status="buffering",
            action="Waiting...",
            confidence=0.0,
            is_recognized=False,
            current_frames=len(sequence_buffer)
        )

    
    input_tensor = np.expand_dims(sequence_buffer, axis=0)
    
  
    res = model.predict(input_tensor, verbose=0)[0]

  
    best_idx = int(np.argmax(res))
    confidence = float(res[best_idx])
    predicted_action = str(ACTIONS[best_idx])

   
    THRESHOLD = 0.80

    if confidence >= THRESHOLD:
        return PredictionResponse(
            status="success",
            action=predicted_action,
            confidence=round(confidence, 4),
            is_recognized=True,
            current_frames=30
        )
    else:
        return PredictionResponse(
            status="uncertain",
            action="Uncertain",
            confidence=round(confidence, 4),
            is_recognized=False,
            current_frames=30
        )

@app.post("/reset")
async def reset_buffer():
    """새로운 동작 인식을 시작할 때 시퀀스 버퍼를 초기화합니다."""
    global sequence_buffer
    sequence_buffer = []
    return {"status": "cleared", "message": "Sequence buffer has been reset."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)