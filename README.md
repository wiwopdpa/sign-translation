<<<<<<< HEAD
# 실시간 수어 번역기 (프로젝트명 추후 확정)

농인이 수어(지문자)로 표현하면 AI가 인식해 텍스트·음성으로 변환하고, 청인의 음성은 텍스트로 변환해 보여주는 실시간 소통 도구

## 팀 구성

| 담당 | 역할 | 주요 작업 |
|---|---|---|
| A | 프론트엔드 |  |
| B | 백엔드 |  |
| C | AI / 인식로직 |  |

## 폴더 구조

```
/frontend   → React 앱 (A)
/backend    → API 서버 (B)
/model      → 인식 모델 코드 (C)
```

## 실행 방법

(추후 작성 — 각자 본인 폴더 세팅 끝나면 채우기)

## 참고자료

- SignTalk 레포: (링크 붙여넣기)
  - 빌려온 것: 지문자 인식 모델
  - 직접 만든 것: 웹 프론트, 양방향 통역 기능, 화상통화 연동
=======
# 🤟 SignBridge — 실시간 수어 번역기

> 농인이 수어(지문자)로 표현하면 AI가 실시간으로 인식해 텍스트·음성으로 변환하고,
> 청인의 음성은 자막으로 변환하는 양방향 소통 웹 서비스

<br>

## 📌 프로젝트 소개

**차별점**
- 기존 서비스: 음성 → 수어/자막 (단방향)
- SignBridge: 수어 → 텍스트/음성 + 음성 → 자막 (양방향)
- 브라우저만으로 즉시 실행 (별도 앱 설치 불필요)
- OBS 가상카메라를 통해 Zoom, Google Meet 등 기존 화상통화 플랫폼에서 활용 가능

<br>

## 🛠 기술 스택

| 파트 | 기술 |
|---|---|
| 프론트엔드 | HTML5, JavaScript, MediaPipe Hands, Web Speech API, SpeechSynthesis API |
| 백엔드 | Python, FastAPI, Uvicorn, WebSocket |
| AI 모델 | (추후 기재) |

<br>

## 📁 폴더 구조

```
SignBridge/
├── frontend/
│   └── sign_translator.html   # 수어 번역기 UI
├── backend/
│   ├── main.py                # FastAPI 서버 (WebSocket 엔드포인트)
│   ├── mock_model.py          # AI 모델 mock (테스트용)
│   ├── websocket_manager.py   # WebSocket 연결 관리
│   └── requirements.txt
└── README.md
```

<br>

## ⚙️ 실행 방법

### 1. 레포 클론
```bash
git clone https://github.com/wiwopdpa/sign-translation
cd sign-translation
```

### 2. 백엔드 실행
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
터미널에 `http://127.0.0.1:8000` 뜨면 성공

### 3. 프론트엔드 실행
VS Code에서 `frontend/sign_translator.html` 우클릭 → **Open with Live Server**

브라우저에서 `http://127.0.0.1:5500/frontend/sign_translator.html` 열림

> ⚠️ 마이크·카메라 권한 허용 필요
> ⚠️ 크롬(Chrome) 브라우저 권장 (Web Speech API 지원)

### 4. 사용 방법
1. 📷 버튼으로 카메라 켜기 (MediaPipe가 손 관절 자동 감지)
2. 🔌 연결 버튼으로 백엔드 WebSocket 연결
3. 웹캠 앞에서 지문자 표현 → 번역 결과 자동 표시
4. 👂 버튼으로 음성 인식 시작 → 상대방 음성 자막 표시

<br>

## 👥 팀 구성

| 이름 | 역할 |
|---|---|
| 이예진 | 프론트엔드 |
| 배준범 | 백엔드 |
| 이동규 | AI 모델 |

<br>

## 📄 참고자료

- MediaPipe Hands: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- FastAPI: https://fastapi.tiangolo.com/
- AI 모델: (추후 기재)

>>>>>>> c638d7f53f7fb480f52234fc4d80d216e59f0613
