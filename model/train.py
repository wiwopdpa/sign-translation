import numpy as np
import os
from sklearn.model_selection import train_test_split
import keras

Sequential = keras.models.Sequential
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'idle']) # idle 포함 3개
no_sequences = 30
sequence_length = 30

label_map = {label: idx for idx, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = keras.utils.to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# LSTM 모델 구조
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax')) # 3개 클래스 (hello, thanks, idle)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("학습 시작...")
model.fit(X_train, y_train, epochs=200)
model.summary()

# 새로 재학습된 my_model.h5 덮어쓰기 저장
model.save('my_model.h5')
print("\n[성공] 새로운 모델 학습 완료: my_model.h5 저장됨!")