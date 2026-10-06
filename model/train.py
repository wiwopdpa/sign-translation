import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "MP_Data")
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "my_model.h5")

ACTIONS = np.array(['hello', 'thanks', 'happy', 'sad', 'iloveyou'])
SEQUENCE_LENGTH = 30
FEATURES = 126

label_map = {label: num for num, label in enumerate(ACTIONS)}

sequences, labels = [], []

print("=" * 60)
print("학습 데이터셋 검증 및 로딩 시작")
print("=" * 60)

for action in ACTIONS:
    action_dir = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_dir):
        print(f"[경고] {action} 디렉터리를 찾을 수 없습니다: {action_dir}")
        continue
    
    seq_folders = sorted(os.listdir(action_dir), key=lambda x: int(x) if x.isdigit() else 9999)
    for seq in seq_folders:
        npy_path = os.path.join(action_dir, seq, "0.npy")
        if os.path.exists(npy_path):
            res = np.load(npy_path)
            
            # 형상이 (30, 126)과 일치하는 정상 데이터만 수집
            if res.shape == (SEQUENCE_LENGTH, FEATURES):
                sequences.append(res)
                labels.append(label_map[action])
            else:
                print(f"[제외됨] 형상 불일치: {action}/{seq} -> {res.shape}")

# 안전하게 넘파이 배열 변환
X = np.array(sequences, dtype=np.float32)
y = to_categorical(labels, num_classes=len(ACTIONS)).astype(int)

print(f"\n최종 로드 완료:")
print(f" - 입력 데이터 X 형태: {X.shape}")
print(f" - 라벨 데이터 y 형태: {y.shape}")

if len(X) == 0:
    raise ValueError("학습 가능한 정상 시퀀스가 0개입니다.")

# 훈련셋 분할 (25개 시퀀스 전체 패턴 유지를 위해 stratify 적용)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

# 모델 구성
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, FEATURES)),
    Dropout(0.2),
    LSTM(128, return_sequences=True, activation='relu'),
    Dropout(0.2),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(ACTIONS), activation='softmax')
])

model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.summary()

early_stop = EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)

print("\n" + "=" * 60)
print("모델 훈련 시작")
print("=" * 60)

history = model.fit(
    X, y,
    epochs=250,
    batch_size=2,
    shuffle=True,
    callbacks=[early_stop]
)

# 학습된 모델 덮어쓰기
model.save(MODEL_SAVE_PATH)
print("\n" + "=" * 60)
print(f"새 모델 저장 완료: {MODEL_SAVE_PATH}")
print("=" * 60)