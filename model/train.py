import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 1. 경로 및 방금 변환한 5개 단어 라벨 지정
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MP_Data')
actions = np.array(['hello', 'thanks', 'happy', 'sad', 'iloveyou'])
no_sequences = 5       # 변환된 시퀀스 수 (단어당 5개)
sequence_length = 30   # 30 프레임

# 2. 데이터 불러오기
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        continue
    dir_list = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
    for seq_dir in dir_list:
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(action_path, seq_dir, f"{frame_num}.npy")
            if os.path.exists(npy_path):
                window.append(np.load(npy_path))
            else:
                window.append(np.zeros(126, dtype=np.float32))
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 3. 모델 빌드 (126차원 입력, 5개 클래스 출력)
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)),
    Dropout(0.2),
    LSTM(128, return_sequences=True, activation='relu'),
    Dropout(0.2),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(actions.shape[0], activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print(f"[*] 총 학습 데이터 형태: {X.shape}, 라벨 수: {len(actions)}")
model.fit(X_train, y_train, epochs=120, batch_size=8, validation_data=(X_test, y_test))

# 4. 새 가중치 파일로 저장
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'my_model.h5')
model.save(save_path)
print(f"\n[🎉 성공] 최신 모델이 저장되었습니다: {save_path}")