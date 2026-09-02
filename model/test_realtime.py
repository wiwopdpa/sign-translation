import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, 'MP_Data')
MODEL_PATH = os.path.join(CURRENT_DIR, 'my_model.h5')

# 1. 학습된 단어 목록 및 모델 로드
actions = np.array(sorted([
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d)) and not d.startswith('.')
]))

model = tf.keras.models.load_model(MODEL_PATH)
print("로드된 단어 목록:", actions)

# 2. MediaPipe Hands 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_landmarks(results):
    lh = np.zeros(63)
    rh = np.zeros(63)
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            coords = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if label == 'Left':
                lh = coords
            elif label == 'Right':
                rh = coords
    return np.concatenate([lh, rh])

sequence = []
sentence = ""
confidence_text = ""
threshold = 0.3  # 인식 기준치 (30%)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 처리
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 랜드마크 시각화
    if results.multi_hand_landmarks:
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

    # 예측 파이프라인
    keypoints = extract_landmarks(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        best_idx = np.argmax(res)
        confidence = res[best_idx]

        if confidence > threshold:
            sentence = actions[best_idx]
            confidence_text = f"{confidence * 100:.1f}%"
        else:
            confidence_text = f"({confidence * 100:.1f}%)"

    # 화면 상단에 검은색 배경 바 및 텍스트 렌더링
    cv2.rectangle(image, (0, 0), (640, 50), (0, 0, 0), -1)
    display_str = f"Sign: {sentence} {confidence_text}"
    cv2.putText(image, display_str, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Real-time Sign Test (Press Q to Exit)', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()