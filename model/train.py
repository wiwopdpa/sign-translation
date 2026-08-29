import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 현재 위치 기준 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, 'MP_Data')
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, 'my_model.h5')

# MP_Data 내부의 모든 단어 폴더 자동 감지 및 정렬
actions = np.array(sorted([
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d)) and not d.startswith('.')
]))

print(f"학습 대상 단어 목록 ({len(actions)}개):", actions)

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

# 데이터셋 로드
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    sequence_folders = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]
    
    for sequence in sequence_folders:
        window = []
        seq_path = os.path.join(action_path, sequence)
        frame_files = [f for f in os.listdir(seq_path) if f.endswith('.npy')]
        
        # 30프레임이 채워진 시퀀스만 로드
        if len(frame_files) < 30:
            continue
            
        for frame_num in range(30):
            res = np.load(os.path.join(seq_path, f"{frame_num}.npy"))
            window.append(res)
            
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels, num_classes=len(actions)).astype(int)

# 데이터 수가 적은 경우(클래스당 1개 등) stratify 없이 안전하게 분할
if len(X) < 40:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
else:
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(f"데이터셋 크기 - Train: {X_train.shape}, Test: {X_test.shape}")

# LSTM 모델 구성 (입력: 30프레임 x 126차원)
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)),
    Dropout(0.2),
    LSTM(128, return_sequences=True, activation='relu'),
    Dropout(0.2),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='loss', save_best_only=True, mode='min')

print("\n🚀 모델 학습 시작...")
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    callbacks=[checkpoint]
)

# 최종 모델 저장 보장
model.save(MODEL_SAVE_PATH)
print(f"\n🎉 모델 학습 완료! 파일 저장 위치: {MODEL_SAVE_PATH}")