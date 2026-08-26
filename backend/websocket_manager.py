# backend/websocket_manager.py
from fastapi import WebSocket
from typing import List

class ConnectionManager:
    """
    WebSocket 연결 관리자
    - 클라이언트의 접속 및 해제 관리
    - 연결된 클라이언트에게 메시지 전송 / 브로드캐스트
    """
    def __init__(self):
        # 현재 연결된 WebSocket 클라이언트 목록
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """클라이언트 접속 수락 및 목록 추가"""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WebSocket] 새 클라이언트 연결됨. 현재 접속 수: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """클라이언트 접속 해제 처리"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WebSocket] 클라이언트 연결 해제. 현재 접속 수: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """특정 클라이언트에게만 JSON 메시지 전송"""
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        """연결된 모든 클라이언트에게 JSON 메시지 전송 (화상통화/공유용)"""
        for connection in self.active_connections:
            await connection.send_json(message)

# 전역 관리자 인스턴스 생성
manager = ConnectionManager()