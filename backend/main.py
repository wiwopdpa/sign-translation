import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sign_model import predict_sign

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Sign Translation Backend Running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("\n[WebSocket] 클라이언트가 정상 연결되었습니다.")
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            landmarks = message.get("landmarks", [])

            # 1) 프론트엔드에서 보낸 랜드마크가 객체 배열[{x,y,z},...] 형태일 경우 126차원으로 평탄화
            if landmarks and isinstance(landmarks[0], dict):
                flat_lm = []
                for p in landmarks:
                    flat_lm.extend([p.get('x', 0.0), p.get('y', 0.0), p.get('z', 0.0)])
                landmarks = flat_lm

            # 2) 126차원 확인 및 추론
            if landmarks and len(landmarks) == 126:
                word, confidence = predict_sign(landmarks)
                
                # 예측 결과가 있으면 무조건 프론트엔드로 텍스트와 퍼센트 전달
                if word:
                    conf_percent = round(confidence * 100, 1)
                    payload = {
                        "text": word,
                        "translation": word,
                        "confidence": conf_percent
                    }
                    await websocket.send_text(json.dumps(payload))
                    print(f" [-> 화면 전송] 단어: {word} | 신뢰도: {conf_percent}%")

    except WebSocketDisconnect:
        print("\n[WebSocket] 클라이언트 연결 종료")
    except Exception as e:
        print(f"\n[WebSocket 에러 발생] {e}")