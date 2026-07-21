import asyncio
import random

class MockSignModel:
    def __init__(self):
        self.labels = ["안녕하세요", "감사합니다", "반갑습니다", "수어", "번역", "테스트"]
        print("[AI Model] 비동기 Mock 수어 인식 모델 준비 완료")

    async def predict(self, landmarks: list) -> dict:
        # 1. 데이터 검증 (좌표가 비어있거나 형식이 안 맞을 때)
        if not landmarks:
            return {
                "status": "warning",
                "text": "손이 인식되지 않았습니다.",
                "confidence": 0.0
            }

        # 2. 비동기 추론 연산 대기 시뮬레이션 (0.05초)
        await asyncio.sleep(0.05)
        
        # 3. 규격화된 결과 반환
        return {
            "status": "success",
            "text": random.choice(self.labels),
            "confidence": round(random.uniform(0.85, 0.99), 2)
        }

sign_model = MockSignModel()