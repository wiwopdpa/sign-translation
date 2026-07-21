import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 1. 학습할 라벨(단어 목록) 설정
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks']) # 수집한 단어 이름들
label_map = {label:num for num, label in enumerate(actions)}

# 2. .npy 데이터 불러오기
sequences, labels = [], []
for action in actions:
    for sequence in range(10): # 수집한 세트 수
        res = np.load(os.path.join(DATA_PATH, action, f"{action}_{sequence}.npy"))
        sequences.append(res)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# 3. LSTM AI 모델 구조 생성
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# 4. 모델 컴파일 및 학습
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print("AI 모델 학습을 시작합니다...")
model.fit(X_train, y_train, epochs=200)

# 5. 두뇌 파일 저장
model.save('my_model.h5')
print("학습 완료! 'my_model.h5' 파일이 생성되었습니다.")