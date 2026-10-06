import os
import numpy as np
import tensorflow as tf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "model", "my_model.h5"))

DEFAULT_ACTIONS = np.array(['hello', 'thanks', 'happy', 'sad', 'iloveyou'])
SEQUENCE_LENGTH = 30

model = None
sequence = []

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[Model] 모델 로드 성공: {MODEL_PATH}")
        except Exception as e:
            print(f"[Model] 모델 로드 실패: {e}")
            model = None
    else:
        print(f"[Model] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

load_model()

def normalize_hand_stream(pts_63):
    """실시간 웹캠 63차원 좌표를 손목 기준 상대 정규화"""
    arr = np.array(pts_63, dtype=np.float32)
    if arr.size != 63 or np.all(arr == 0):
        return np.zeros(63, dtype=np.float32)
    
    arr = arr.reshape(21, 3)
    wrist = arr[0].copy()
    rel = arr - wrist
    
    max_val = np.max(np.abs(rel))
    if max_val > 1e-5:
        rel = rel / max_val
    return rel.flatten()

def predict_sign(landmarks_126):
    global sequence, model

    if model is None:
        load_model()
        if model is None:
            return None, 0.0

    # 1) 왼손(0~62)과 오른손(63~125) 분리 후 손목 기준 상대 정규화
    left_hand = landmarks_126[:63]
    right_hand = landmarks_126[63:]
    
    norm_left = normalize_hand_stream(left_hand)
    norm_right = normalize_hand_stream(right_hand)
    normalized_frame = np.concatenate([norm_left, norm_right])

    # 2) 30프레임 버퍼 관리
    sequence.append(normalized_frame)
    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    # 30프레임이 다 모였을 때 추론
    if len(sequence) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(sequence, axis=0)  # (1, 30, 126)
        res = model.predict(input_data, verbose=0)[0]
        
        best_idx = int(np.argmax(res))
        confidence = float(res[best_idx])
        word = DEFAULT_ACTIONS[best_idx]

        # 디버깅용: 각 단어별 확률 터미널 출력
        probs_str = " | ".join([f"{DEFAULT_ACTIONS[i]}: {round(res[i]*100, 1)}%" for i in range(len(DEFAULT_ACTIONS))])
        print(f"[확률 분포] {probs_str}")

        # 신뢰도가 최소 40% 이상일 때만 결과 반환 (노이즈 방지)
        if confidence >= 0.20:
            return word, confidence

    return None, 0.0