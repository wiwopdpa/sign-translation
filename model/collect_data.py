import cv2
import numpy as np
import os
import mediapipe as mp

# 1. 'idle' 포함 3개 클래스 설정
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['hello', 'thanks', 'idle']) # idle 추가
no_sequences = 30 # 각 단어당 30번 수집
sequence_length = 30 # 1번 수집할 때 30프레임(약 1초)

# 데이터 보관 폴더 자동 생성
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# 2. MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

print("데이터 수집을 시작합니다. 'q'를 누르면 중단됩니다.")

for action in actions:
    for sequence in range(no_sequences):
        for frame_num in range(sequence_length):

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 카운트다운 및 시작 카드의 안내 문구
            if frame_num == 0: 
                cv2.putText(frame, 'PREPARE ACTION...', (120, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f'Collecting "{action}" ({sequence + 1}/{no_sequences})', (15, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                cv2.waitKey(1000) # 1초 대기 후 촬영 시작
            else: 
                cv2.putText(frame, f'Collecting "{action}" ({sequence + 1}/{no_sequences})', (15, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)

            # 랜드마크 좌표 추출
            keypoints = []
            if results.multi_hand_landmarks:
                if len(results.multi_hand_landmarks) == 1:
                    for res in results.multi_hand_landmarks[0].landmark:
                        keypoints.append([res.x, res.y, res.z])
                    keypoints.extend([[0.0, 0.0, 0.0]] * 21)
                else:
                    for hand_res in results.multi_hand_landmarks[:2]:
                        for res in hand_res.landmark:
                            keypoints.append([res.x, res.y, res.z])
            else:
                keypoints = [[0.0, 0.0, 0.0]] * 42

            res_data = np.array(keypoints).flatten()
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, res_data)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()