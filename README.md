# 🤟 SignBridge — 실시간 수어 번역기

> 농인이 수어(지문자)로 표현하면 AI가 실시간으로 인식해 텍스트·음성으로 변환하고,
> 청인의 음성은 자막으로 변환해 보여주는 양방향 소통 도구

<br>

## 📌 프로젝트 소개

청각장애인과 비장애인 사이의 소통 장벽을 줄이기 위해 개발한 실시간 수어 번역 웹입니다.
Zoom, Google Meet 등 화상통화 환경에서 화면공유 또는 OBS 가상 카메라를 통해 바로 활용할 수 있습니다.

### 기존 서비스와 차이점

| | 기존 서비스 | SignBridge |
|---|---|---|
| 방향 | 음성 → 수어/자막 | **수어 → 텍스트/음성** + 음성 → 자막 |
| 통역 방식 | 통역사 연결 또는 아바타 | AI 실시간 인식 |
| 화상통화 연동 | 별도 앱 필요 | 브라우저에서 바로 실행 |

<br>

## 🛠 기술 스택

| 파트 | 기술 |
|---|---|
| 프론트엔드 | React, Web Speech API, WebRTC |
| 백엔드 | Python, FastAPI |
| AI 모델 | (SignTalk 기반 지문자 인식 모델) |

<br>

## 📁 폴더 구조

```
SignBridge/
├── frontend/   # React 프론트엔드
├── backend/    # FastAPI 백엔드 서버
└── model/      # 지문자 인식 AI 모델
```

<br>

## ⚙️ 실행 방법

### 프론트엔드
```bash
cd frontend
npm install
npm start
```

### 백엔드
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

> 상세 환경변수 설정은 각 폴더의 README 참고

<br>

## 👥 팀 구성

| 이름 | 역할 |
|---|---|
| (이름) | 프론트엔드 |
| (이름) | 백엔드 |
| (이름) | AI 모델 |

<br>

## 📄 참고자료

- SignTalk (지문자 인식 모델 참고)
  - 활용 부분: 지문자 인식 모델 구조
  - 자체 개발 부분: 웹 UI, 양방향 번역 기능, 화상통화 연동

<br>

---

