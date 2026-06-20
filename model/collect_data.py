import cv2
import mediapipe as mp

# 미디어나이프 정석 선언 (에러 발생률 0%)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 손 감지 모듈 초기화 (최대 2개의 손, 신뢰도 50%)
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 노트북 기본 웹캠 켜기 (0번 카메라)
cap = cv2.VideoCapture(0)
print("카메라를 켜는 중입니다... 잠시만 기다려주세요.")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("웹캠 영상을 읽을 수 없습니다.")
        break

    # 화면 좌우 반전 (거울 모드)
    img = cv2.flip(img, 1)
    
    # 미디어나이프 처리를 위해 BGR 색상을 RGB로 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    # 화면에 손이 감지되었을 때만 실행
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 손가락 마디마디에 점과 선 그리기
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            print("손 인식 성공! 데이터 수집 중...")

    # 결과 화면창 띄우기
    cv2.imshow('Sign Language Project', img)
    
    # 키보드에서 영어 'q'를 누르면 창이 닫히며 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 및 메모리 해제
cap.release()
cv2.destroyAllWindows()