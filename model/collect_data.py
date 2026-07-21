import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ================================================================= #
# [데이터 수집 설정] 수집할 단어와 세트 수를 여기서 조절하세요.
# ================================================================= #
action_name = "thanks"      # 수집할 수어 단어 이름 (폴더명이 됩니다)
sequence_length = 30      # 하나의 동작을 녹화할 프레임 수 (30프레임 = 약 1초)
no_sequences = 10         # 총 수집할 세트 수 (테스트용으로 우선 10개만 수집)
DATA_PATH = os.path.join('MP_Data') # 데이터가 저장될 메인 폴더명

# 수집할 단어의 폴더 생성
os.makedirs(os.path.join(DATA_PATH, action_name), exist_ok=True)
# ================================================================= #

# 미디어나이프(MediaPipe) 손 인식 관련 도구 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,               # 양손 모두 인식 가능하도록 설정
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print(f"==================================================")
print(f" [{action_name}] 수어 데이터 수집 시스템을 시작합니다.")
print(f" 카메라 화면을 클릭한 상태에서 'q'를 누르면 안전하게 종료됩니다.")
print(f"==================================================")

# 데이터 수집 루프 시작
for sequence in range(no_sequences):
    # 각 세트가 시작되기 전, 손 위치를 잡을 수 있도록 2초간 딜레이 제공
    for countdown in range(2, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'PREPARING SET {sequence+1}/{no_sequences} (Starts in {countdown}s)', 
                    (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Sign Language Data Collector', frame)
        cv2.waitKey(1000) # 1초 대기
        
    # 30프레임 동안의 연속 좌표를 담을 임시 공간
    sequence_data = []

    print(f"-> 현재 {sequence+1}/{no_sequences} 번째 세트 데이터 캡처 중...")

    for frame_num in range(sequence_length):
        ret, frame = cap.read()
        if not ret:
            break

        # 화면 좌우 반전 및 미디어나이프 인식을 위한 RGB 변환
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 화면에 손 랜드마크 시각화 뼈대 그리기
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # [핵심] 랜드마크 좌표(x, y, z) 추출 및 데이터 정규화
            keypoints = []
            
            # 한 손만 감지되었을 때와 두 손 모두 감지되었을 때의 데이터 구조 통일
            if len(results.multi_hand_landmarks) == 1:
                # 감지된 첫 번째 손의 21개 마디 좌표 추출
                for res in results.multi_hand_landmarks[0].landmark:
                    keypoints.append([res.x, res.y, res.z])
                # 크기를 맞추기 위해 나머지 한 손 분량은 0으로 채움 (21마디 * 3좌표)
                keypoints.extend([[0.0, 0.0, 0.0]] * 21)
            else:
                # 두 손이 다 보일 때는 순서대로 42개 마디의 좌표를 모두 추출
                for hand_res in results.multi_hand_landmarks[:2]:
                    for res in hand_res.landmark:
                        keypoints.append([res.x, res.y, res.z])
            
            # 리스트를 넘파이 1차원 배열로 평평하게 폅니다 (크기: 126)
            res_data = np.array(keypoints).flatten()
            sequence_data.append(res_data)
        else:
            # 화면에 손이 전혀 보이지 않으면 0으로 가득 찬 빈 데이터를 넣어 프레임 유지
            sequence_data.append(np.zeros(21 * 3 * 2))

        # 실시간 상태 레이아웃 정보 출력 (단어명, 현재 세트, 진행 프레임)
        cv2.putText(frame, f'DATA COLLECTING - Action: {action_name.upper()}', (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f'Set: {sequence+1}/{no_sequences} | Frame: {frame_num+1}/{sequence_length}', (15, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imshow('Sign Language Data Collector', frame)

        # 'q' 키를 누르면 루프 탈출 후 즉시 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 30프레임(1세트) 분량의 시퀀스 데이터가 모이면 .npy 파일로 최종 빌드 및 저장
    if len(sequence_data) == sequence_length:
        npy_path = os.path.join(DATA_PATH, action_name, f"{action_name}_{sequence}.npy")
        np.save(npy_path, np.array(sequence_data))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\n==================================================")
print(f" 🎉 모든 수집이 완료되었습니다!")
print(f" 프로젝트 루트 폴더 안의 '{DATA_PATH}/{action_name}' 폴더를 확인해 보세요.")
print(f"==================================================")

cap.release()
cv2.destroyAllWindows()