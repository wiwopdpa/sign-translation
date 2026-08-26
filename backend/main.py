import os
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from websocket_manager import manager
from sign_model import SignLanguageModel

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
    ai_service = SignLanguageModel()

    try:
        while True:
            # 1. 프론트엔드가 보낸 데이터 수신
            data = await websocket.receive_json()
            landmarks = data.get("landmarks", [])

            # 2. 양손 기준 126개 좌표 확인 후 AI 모델 추론 진행
            if landmarks and len(landmarks) == 126:
                result = ai_service.process_frame(landmarks)

                # 3. 30프레임 축적 및 확신도/연속성 조건 충족 시에만 프론트엔드로 결과 전송
                if result:
                    response = {
                        "status": "success",
                        "translated_text": result["action"],
                        "confidence": round(result["confidence"], 2)
                    }
                    await manager.send_personal_message(json.dumps(response), websocket)

    except WebSocketDisconnect:
        ai_service.reset()
        manager.disconnect(websocket)
    except Exception as e:
        ai_service.reset()
        manager.disconnect(websocket)