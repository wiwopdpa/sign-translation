# backend/main.py
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from websocket_manager import manager
from mock_model import sign_model

app = FastAPI(
    title="SignBridge Backend API",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 루트 접속 시 frontend/sign_translator.html 반환
@app.get("/", response_class=HTMLResponse)
def read_root():
    html_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "sign_translator.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>HTML 파일 없음</h1>"

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "SignBridge Backend is running"}


# WebSocket 수어 추론 엔드포인트
@app.websocket("/ws/sign")
async def websocket_sign_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 1. 프론트엔드가 보낸 데이터 수신
            data = await websocket.receive_json()
            landmarks = data.get("landmarks", [])

            # 2. (Mock) AI 모델 추론
            result = sign_model.predict(landmarks)

            # 3. 번역 결과 반환
            response = {
                "status": "success",
                "translated_text": result["text"],
                "confidence": result["confidence"]
            }
            await manager.send_personal_message(response, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)