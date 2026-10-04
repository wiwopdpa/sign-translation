import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sign_model import predict_sign

app = FastAPI(title="Sign Translation Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("\n[WebSocket] 클라이언트가 정상 연결되었습니다.")
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            raw_landmarks = message.get("landmarks", [])

            # 42개 객체 [{x, y, z}, ...] 형태로 올 경우 -> 126개 float 배열로 변환
            flattened = []
            if isinstance(raw_landmarks, list):
                if len(raw_landmarks) == 42 and isinstance(raw_landmarks[0], dict):
                    for point in raw_landmarks:
                        flattened.extend([
                            float(point.get('x', 0.0)),
                            float(point.get('y', 0.0)),
                            float(point.get('z', 0.0))
                        ])
                elif len(raw_landmarks) == 126:
                    flattened = [float(val) for val in raw_landmarks]

            # 126개 완성 시 추론 수행
            if len(flattened) == 126:
                predicted_word, confidence = predict_sign(flattened)
                if predicted_word:
                    await websocket.send_json({
                        "status": "success",
                        "word": predicted_word,
                        "confidence": round(confidence * 100, 1)
                    })
            else:
                print(f"\r[수신 데이터 크기 확인] {len(raw_landmarks)}개 수신됨", end="", flush=True)

    except WebSocketDisconnect:
        print("\n[WebSocket] 클라이언트 연결 종료.")
    except Exception as e:
        print(f"\n[WebSocket 오류] {e}")