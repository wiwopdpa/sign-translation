import cv2
import numpy as np
import mediapipe as mp
import keras

load_model = keras.models.load_model

model = load_model('my_model.h5')
actions = np.array(['hello', 'thanks', 'idle']) # idle 포함

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

sequence = []
sentence = []
predictions = []
threshold = 0.85 # 확신도 85% 기준

cap = cv2.VideoCapture(0)

print("실시간 테스트 시작! ('q'로 종료)")

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
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            best_action_idx = np.argmax(res)
            predictions.append(best_action_idx)
            predictions = predictions[-10:]

            # 최근 10번 예측이 일치하고 확신도가 85% 이상일 때
            if np.unique(predictions[-10:])[0] == best_action_idx:
                if res[best_action_idx] > threshold:
                    detected_action = actions[best_action_idx]
                    
                    # 🟢 idle이 아닐 때만 자막에 추가 (idle일 때는 자막 추가 생략)
                    if detected_action != 'idle':
                        if len(sentence) > 0:
                            if detected_action != sentence[-1]:
                                sentence.append(detected_action)
                        else:
                            sentence.append(detected_action)

            if len(sentence) > 5:
                sentence = sentence[-5:]
    else:
        sequence = []
        predictions = []

    # 화면 상단 자막 바 출력
    cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
    cv2.putText(frame, ' '.join(sentence), (10, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Real-time Sign Translation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()