from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sign_model import predict_sign
import json

app = FastAPI(title="Sign Translation Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "running", "message": "Sign Language Translation API"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("웹소켓 클라이언트 연결 완료")
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            landmarks = message.get("landmarks", [])
            if landmarks and len(landmarks) == 126:
                predicted_word, confidence = predict_sign(landmarks)
                
                # 예측 결과가 있으면 클라이언트로 즉시 전송
                if predicted_word:
                    await websocket.send_json({
                        "status": "success",
                        "word": predicted_word,
                        "confidence": round(confidence * 100, 1)
                    })
    except WebSocketDisconnect:
        print("웹소켓 클라이언트 연결 종료")
    except Exception as e:
        print(f"웹소켓 에러 발생: {e}")