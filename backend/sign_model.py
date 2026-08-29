import os
import numpy as np
import tensorflow as tf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'model')
DATA_PATH = os.path.join(MODEL_DIR, 'MP_Data')
MODEL_PATH = os.path.join(MODEL_DIR, 'my_model.h5')

# MP_Data 기준으로 actions 자동 동기화 (train.py와 동일한 순서 보장)
if os.path.exists(DATA_PATH):
    ACTIONS = np.array(sorted([
        d for d in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, d)) and not d.startswith('.')
    ]))
else:
    ACTIONS = np.array(['happy', 'hello', 'iloveyou', 'ok', 'sad', 'thanks'])

print(f"인식 단어 목록 로드 완료 ({len(ACTIONS)}개):", ACTIONS)

# 모델 로드
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"가중치 모델 로드 성공: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ 모델 로드 실패: {e}")
    model = None

sequence = []
CONFIDENCE_THRESHOLD = 0.70  # 신뢰도 70% 이상일 때만 번역 결과 인정

def predict_sign(frame_landmarks):
    """
    frame_landmarks: 126차원 1차원 float 배열
    반환: (예측단어, 신뢰도) 또는 (None, 0.0)
    """
    global sequence, model

    if model is None:
        return None, 0.0

    if len(frame_landmarks) != 126:
        return None, 0.0

    sequence.append(frame_landmarks)
    # 최근 30개 프레임 슬라이딩 윈도우 유지
    sequence = sequence[-30:]

    if len(sequence) == 30:
        input_data = np.expand_dims(sequence, axis=0)
        res = model.predict(input_data, verbose=0)[0]
        
        best_idx = int(np.argmax(res))
        confidence = float(res[best_idx])

        if confidence >= CONFIDENCE_THRESHOLD:
            predicted_action = ACTIONS[best_idx]
            return predicted_action, confidence

    return None, 0.0