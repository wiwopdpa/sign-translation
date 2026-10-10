import cv2
import numpy as np
import os
import time
import mediapipe as mp

# 1. 설정 및 경로 정의
DATA_PATH = os.path.join(os.path.dirname(__file__), 'MP_Data')
actions = np.array(['hello', 'thanks', 'happy', 'sad', 'iloveyou'])

no_sequences = 20  # 단어당 수집할 시퀀스 횟수 (20회 권장)
sequence_length = 30  # 시퀀스당 프레임 수 (30프레임)

# 폴더 생성
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_normalized_keypoints(results):
    """
    126차원 손목 기준 정규화 벡터 추출 (프론트엔드/백엔드 규격과 100% 일치)
    - 왼손 21개 * 3 = 63
    - 오른손 21개 * 3 = 63
    - 총 126차원
    """
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_handedness in enumerate(results.multi_handedness):
            label = hand_handedness.classification[0].label  # 'Left' or 'Right'
            landmarks = results.multi_hand_landmarks[idx].landmark
            
            # 21개 랜드마크 추출 (x, y, z)
            pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            
            # 손목(0번) 기준 상대 좌표 변환
            wrist = pts[0].copy()
            rel_pts = pts - wrist
            
            # 스케일 정규화 (최대 절댓값 기준)
            max_val = np.max(np.abs(rel_pts))
            if max_val < 1e-5:
                max_val = 1.0
            norm_pts = rel_pts / max_val
            flattened = norm_pts.flatten()

            # 거울 모드 기준 좌우 배치
            if label == 'Left':
                left_hand = flattened
            else:
                right_hand = flattened

    return np.concatenate([left_hand, right_hand])  # 126차원 반환

# 웹캠 실행
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2) as hands:

    print("=== 수어 모션 데이터 수집 시작 ===")
    print("준비가 되면 웹캠 창을 클릭하고 's' 키를 눌러 녹화를 시작하세요. (종료는 'q')")

    # 시작 대기 루프
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1) # 거울 모드
        cv2.putText(frame, "Press 's' to START, 'q' to QUIT", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # 데이터 수집 메인 루프
    for action in actions:
        for sequence in range(no_sequences):
            
            # 시퀀스 시작 전 2초간 카운트다운 및 손 위치 준비 시간
            for countdown in range(20, 0, -1):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, hands)

                # 스켈레톤 라인 렌더링
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 안내 텍스트 표시
                cv2.putText(image, f"READY: [{action.upper()}] Sequence #{sequence + 1}/{no_sequences}", 
                            (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"STARTING IN: {countdown / 10:.1f}s (HANDS UP!)", 
                            (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
                
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(100)

            # 30프레임 실제 모션 녹화 루프
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, hands)

                # 스켈레톤 라인 렌더링
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 녹화 중 붉은색 표시
                cv2.putText(image, f"RECORDING: [{action.upper()}] Frame {frame_num + 1}/{sequence_length}", 
                            (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(image, (30, 50), 12, (0, 0, 255), -1)

                cv2.imshow('OpenCV Feed', image)

                # 126차원 키포인트 저장
                keypoints = extract_normalized_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
    print("=== 모든 데이터 수집 완료 ===")