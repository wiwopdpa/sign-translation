import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# 1. 경로 및 파라미터 정의
DATA_PATH = os.path.join(os.path.dirname(__file__), 'MP_Data')
actions = np.array(['hello', 'thanks', 'happy', 'sad', 'iloveyou'])
sequence_length = 30  # 30프레임 시퀀스

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

print("=" * 60)
print("학습 데이터셋 검증 및 시퀀스 로딩 시작")
print("=" * 60)

for action in actions:
    action_dir = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_dir):
        print(f"[경고] 단어 폴더 없음: {action}")
        continue

    # 시퀀스 폴더 목록 (0, 1, 2, ... 19)
    seq_folders = [f for f in os.listdir(action_dir) if os.path.isdir(os.path.join(action_dir, f))]
    
    # 숫자 기준 정렬
    seq_folders = sorted(seq_folders, key=lambda x: int(x) if x.isdigit() else x)

    valid_seq_count = 0
    for seq_folder in seq_folders:
        seq_path = os.path.join(action_dir, seq_folder)
        window = []
        is_valid = True

        # 시퀀스 내 30개 프레임 순회
        for frame_num in range(sequence_length):
            frame_path = os.path.join(seq_path, f"{frame_num}.npy")
            if not os.path.exists(frame_path):
                is_valid = False
                break
            
            res = np.load(frame_path)
            # 프레임 형상이 126차원인지 검증
            if res.shape != (126,):
                is_valid = False
                break
            window.append(res)

        if is_valid and len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
            valid_seq_count += 1
        else:
            print(f"[제외됨] 손상되거나 프레임 수 부족: {action}/{seq_folder}")

    print(f" -> '{action}': 총 {valid_seq_count}개 시퀀스 로드 완료")

# numpy 배열 변환
X = np.array(sequences)
y = to_categorical(labels, num_classes=len(actions)).astype(int)

print("-" * 60)
print(f"최종 로드 완료:")
print(f" - 입력 데이터 X 형태: {X.shape} (기대 형상: (샘플수, 30, 126))")
print(f" - 라벨 데이터 y 형태: {y.shape}")
print("-" * 60)

if len(X) == 0:
    raise ValueError("학습 가능한 정상 시퀀스가 0개입니다. 데이터 폴더를 확인하세요.")

# 학습용 / 검증용 분할 (8:2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. LSTM 신경망 모델 설계
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(
    optimizer='Adam', 
    loss='categorical_crossentropy', 
    metrics=['categorical_accuracy']
)

model.summary()

# 콜백 설정
callbacks = [
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
]

# 3. 모델 학습
print("\n>>> 모델 학습 시작...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=120,
    batch_size=8,
    callbacks=callbacks
)

# 4. 모델 저장
model_save_path = os.path.join(os.path.dirname(__file__), 'my_model.h5')
model.save(model_save_path)
print(f"\n============================================================")
print(f"학습 완료! 모델이 성공적으로 저장되었습니다: {model_save_path}")
print(f"============================================================")