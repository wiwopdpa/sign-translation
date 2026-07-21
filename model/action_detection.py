import cv2
import numpy as np
import mediapipe as mp
import keras

load_model = keras.models.load_model

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# 1. 저장된 AI 두뇌 불러오기
model = load_model('my_model.h5')
actions = np.array(['hello', 'thanks']) # 단어 목록

# 2. 미디어나이프 및 카메라 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

sequence = []
sentence = []
threshold = 0.8 # 인식 확신도 기준 (80% 이상 확신할 때 출력)

cap = cv2.VideoCapture(0)

print("실시간 수어 번역 테스트 시작! 카메라를 바라보세요. ('q'로 종료)")

while cap.isOpened():
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

        # 좌표 추출
        keypoints = []
        if len(results.multi_hand_landmarks) == 1:
            for res in results.multi_hand_landmarks[0].landmark:
                keypoints.append([res.x, res.y, res.z])
            keypoints.extend([[0.0, 0.0, 0.0]] * 21)
        else:
            for hand_res in results.multi_hand_landmarks[:2]:
                for res in hand_res.landmark:
                    keypoints.append([res.x, res.y, res.z])
        
        res_data = np.array(keypoints).flatten()
        sequence.append(res_data)
        sequence = sequence[-30:] # 최근 30프레임 유지를 통해 실시간 감지

        # 30프레임이 채워지면 AI가 실시간 예측
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            
            # 확신도가 threshold 이상이면 결과 화면 출력
            if res[np.argmax(res)] > threshold:
                detected_action = actions[np.argmax(res)]
                
                # 결과 텍스트 업데이트
                if len(sentence) > 0:
                    if detected_action != sentence[-1]:
                        sentence.append(detected_action)
                else:
                    sentence.append(detected_action)

            if len(sentence) > 5:
                sentence = sentence[-5:]

    # 화면에 실시간 인식된 수어 번역 텍스트 출력!
    cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
    cv2.putText(frame, ' '.join(sentence), (3, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Real-time Sign Translation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()