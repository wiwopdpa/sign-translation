import cv2
import numpy as np
import os
import mediapipe as mp

# 1. 저장 경로 및 파라미터 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "MP_Data")
os.makedirs(DATA_PATH, exist_ok=True)

# 5개 단어, 단어당 20개 시퀀스 수집 (30프레임씩)
ACTIONS = np.array(['hello', 'thanks', 'happy', 'sad', 'iloveyou'])
NO_SEQUENCES = 20
SEQUENCE_LENGTH = 30

# 폴더 자동 생성
for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# 2. MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalize_hand(landmarks):
    """21개 랜드마크를 손목(0번) 기준 상대 좌표로 정규화 (63차원)"""
    if landmarks is None:
        return np.zeros(63, dtype=np.float32)
    
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
    wrist = pts[0].copy()
    rel = pts - wrist
    
    max_val = np.max(np.abs(rel))
    if max_val > 1e-5:
        rel = rel / max_val
    return rel.flatten()

def extract_landmarks(results):
    """왼손, 오른손 분리 후 126차원 벡터 생성"""
    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, handedness in enumerate(results.multi_handedness):
            label = handedness.classification[0].label
            # 거울 모드 기준 좌우 매핑
            if label == 'Left':
                left_hand = results.multi_hand_landmarks[idx]
            elif label == 'Right':
                right_hand = results.multi_hand_landmarks[idx]

    norm_lh = normalize_hand(left_hand)
    norm_rh = normalize_hand(right_hand)
    return np.concatenate([norm_lh, norm_rh])

# 3. 웹캠 녹화 루프
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    for action in ACTIONS:
        # 단어 시작 전 대기 (카메라 화면을 보고 준비할 시간)
        for sequence in range(NO_SEQUENCES):
            sequence_data = []

            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break

                # 화면 좌우 반전 및 RGB 변환
                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True

                # 화면에 손 관절 그리기
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 첫 프레임일 때 안내 메시지 및 2초 준비 카운트다운
                if frame_num == 0:
                    cv2.putText(frame, f'STARTING COLLECTION for [{action.upper()}]', (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'Sequence: {sequence + 1}/{NO_SEQUENCES}', (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(frame, f'Recording [{action}] Seq: {sequence + 1} Frame: {frame_num + 1}', 
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)

                # 126차원 피처 추출 및 누적
                keypoints = extract_landmarks(results)
                sequence_data.append(keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # (30, 126) 넘파이 배열로 저장
            npy_path = os.path.join(DATA_PATH, action, str(sequence), "0.npy")
            np.save(npy_path, np.array(sequence_data, dtype=np.float32))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
print("\n모든 모션 데이터 수집이 완료되었습니다!")