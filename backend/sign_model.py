import os
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "my_model.h5")

# AI-Hub 5개 단어 라벨 목록
DEFAULT_ACTIONS = ['hello', 'thanks', 'happy', 'sad', 'iloveyou']

class SignLanguageModel:
    def __init__(self, model_path=MODEL_PATH, threshold=0.2):  # 임계값을 0.2(20%)로 설정
        print(f"[*] AI 모델 로딩 경로: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = threshold
        self.sequence = []

        output_dim = self.model.output_shape[-1]
        print(f"[*] 모델 출력 클래스 수: {output_dim}개")

        if output_dim == len(DEFAULT_ACTIONS):
            self.actions = DEFAULT_ACTIONS
        elif output_dim > len(DEFAULT_ACTIONS):
            self.actions = DEFAULT_ACTIONS + [f"class_{i}" for i in range(len(DEFAULT_ACTIONS), output_dim)]
        else:
            self.actions = DEFAULT_ACTIONS[:output_dim]

    def predict(self, landmarks):
        # 126차원 프레임 버퍼 유지 (최근 30프레임)
        self.sequence.append(landmarks)
        self.sequence = self.sequence[-30:]

        print(f"\r[수신 중] 프레임 버퍼: {len(self.sequence)}/30", end="", flush=True)

        if len(self.sequence) == 30:
            input_data = np.expand_dims(self.sequence, axis=0)
            res = self.model.predict(input_data, verbose=0)[0]
            best_idx = int(np.argmax(res))
            confidence = float(res[best_idx])
            
            if best_idx < len(self.actions):
                predicted_word = self.actions[best_idx]
            else:
                predicted_word = f"action_{best_idx}"

            print(f"\n[추론 성공] 단어: {predicted_word} | 신뢰도: {confidence * 100:.1f}%")

            # 20% 이상일 때 화면으로 전달
            if confidence >= self.threshold:
                return predicted_word, confidence

        return None, 0.0

# 인스턴스 생성
sign_detector = SignLanguageModel()

def predict_sign(landmarks):
    return sign_detector.predict(landmarks)