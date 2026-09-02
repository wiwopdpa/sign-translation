import os
import numpy as np
import tensorflow as tf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'model')
DATA_PATH = os.path.join(MODEL_DIR, 'MP_Data')
MODEL_PATH = os.path.join(MODEL_DIR, 'my_model.h5')

# MP_Data 폴더 기준 actions 동기화
if os.path.exists(DATA_PATH):
    ACTIONS = np.array(sorted([
        d for d in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, d)) and not d.startswith('.')
    ]))
else:
    ACTIONS = np.array(['happy', 'hello', 'iloveyou', 'ok', 'sad', 'thanks'])

print(f"인식 대상 단어 목록 ({len(ACTIONS)}개):", ACTIONS)

# 모델 로드
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"가중치 모델 로드 성공: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ 모델 로드 실패: {e}")
    model = None

sequence = []
CONFIDENCE_THRESHOLD = 0.35  # 인식 민감도 35%로 완화

def predict_sign(frame_landmarks):
    global sequence, model

    if model is None or len(frame_landmarks) != 126:
        return None, 0.0

    sequence.append(frame_landmarks)
    sequence = sequence[-30:]

    if len(sequence) == 30:
        input_data = np.expand_dims(sequence, axis=0)
        res = model.predict(input_data, verbose=0)[0]
        
        best_idx = int(np.argmax(res))
        confidence = float(res[best_idx])
        predicted_action = ACTIONS[best_idx]

        # 터미널에 실시간 예측 로그 출력
        print(f"[실시간 인식] {predicted_action} ({confidence * 100:.1f}%)")

        if confidence >= CONFIDENCE_THRESHOLD:
            return predicted_action, confidence

    return None, 0.0