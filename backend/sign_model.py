import os
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "my_model.h5")

ACTIONS = np.array(['happy', 'hello', 'iloveyou', 'ok', 'sad', 'thanks'])

class SignLanguageModel:
    def __init__(self, model_path=MODEL_PATH, threshold=0.80):
        print(f"Loading AI Model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = threshold
        self.sequence = []
        self.predictions = []

    def process_frame(self, frame_landmarks):

        if not frame_landmarks or len(frame_landmarks) != 126:
            return None

        self.sequence.append(frame_landmarks)
        self.sequence = self.sequence[-30:]

        if len(self.sequence) < 30:
            return None

        # (1, 30, 126) 차원으로 변환 후 예측
        input_data = np.expand_dims(self.sequence, axis=0)
        res = self.model.predict(input_data, verbose=0)[0]

        best_idx = np.argmax(res)
        best_prob = res[best_idx]

        self.predictions.append(best_idx)
        self.predictions = self.predictions[-5:]

        if best_prob > self.threshold:
            if len(self.predictions) == 5 and len(set(self.predictions)) == 1:
                return {
                    "action": ACTIONS[best_idx],
                    "confidence": float(best_prob)
                }

        return None

    def reset(self):
        self.sequence.clear()
        self.predictions.clear()