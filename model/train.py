import numpy as np
import os
from sklearn.model_selection import train_test_split
import keras

Sequential = keras.models.Sequential
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense

# 데이터 경로
DATA_PATH = os.path.join('MP_Data')

# 1. MP_Data 폴더 내 단어들을 알파벳순으로 정렬하여 자동 인식
actions = np.array(sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]))
print(f"학습 대상 단어 ({len(actions)}개): {actions}")

label_map = {label: idx for idx, label in enumerate(actions)}

sequences, labels = [], []

# 2. 모든 시퀀스 데이터 로드
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    sequences_in_action = [s for s in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, s))]
    
    print(f" -> '{action}': {len(sequences_in_action)}개 시퀀스 불러오는 중...")
    for sequence in sequences_in_action:
        window = []
        try:
            for frame_num in range(30):
                res = np.load(os.path.join(action_path, sequence, f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
        except Exception:
            continue

X = np.array(sequences)
y = keras.utils.to_categorical(labels).astype(int)

# 데이터 분할 (90% 학습, 10% 검증)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 3. 모델 구성
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, X.shape[2])))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# 4. 모델 컴파일 및 학습
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=200, batch_size=32)

# 5. 모델 저장
model.save('my_model.h5')
print("\n🎉 my_model.h5 저장이 완료되었습니다!")