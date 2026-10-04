import os
import glob
import json
import numpy as np

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 압축 해제된 실제 폴더 경로 검색 (temp_data 아래 keypoint/01)
SEARCH_PATTERNS = [
    os.path.join(BASE_DIR, "temp_data", "**", "keypoint", "01"),
    os.path.join(BASE_DIR, "temp_data", "*", "keypoint", "01"),
    os.path.join(BASE_DIR, "temp_data", "keypoint", "01"),
]

DATA_DIR = None
for p in SEARCH_PATTERNS:
    matches = glob.glob(p, recursive=True)
    if matches:
        DATA_DIR = matches[0]
        break

if not DATA_DIR or not os.path.exists(DATA_DIR):
    # 기본 경로 폴백
    DATA_DIR = os.path.join(BASE_DIR, "temp_data", "WORD", "keypoint", "01")

OUTPUT_DIR = os.path.join(BASE_DIR, "MP_Data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[*] AI-Hub 원본 데이터 경로: {DATA_DIR}")
print(f"[*] 변환 결과 저장 경로: {OUTPUT_DIR}")

# 2. 변환 대상 단어 설정 (예: 앞쪽 5개 단어 선택)
# 폴더명 규칙: NIA_SL_WORD0001_SYN01_F -> 정면 카메라(_F) 영상 우선 활용
TARGET_WORDS = {
    "WORD0001": "hello",     # 단어 1
    "WORD0003": "thanks",    # 단어 3
    "WORD0004": "happy",     # 단어 4
    "WORD0005": "sad",       # 단어 5
    "WORD0006": "iloveyou"   # 단어 6
}

# 원본 비디오 해상도 (AI-Hub 표준: FHD 1920x1080)
IMG_WIDTH = 1920.0
IMG_HEIGHT = 1080.0
SEQUENCE_LENGTH = 30  # 모델 학습 시퀀스 길이 (30 프레임)

def parse_hand_points(raw_list):
    """63개 원소 [x, y, c, x, y, c...] -> [x_norm, y_norm, c...] 63개 반환"""
    if not raw_list or len(raw_list) < 63:
        return np.zeros(63, dtype=np.float32)
    
    pts = np.array(raw_list, dtype=np.float32)
    # x, y 정규화 (0.0 ~ 1.0)
    pts[0::3] /= IMG_WIDTH
    pts[1::3] /= IMG_HEIGHT
    return pts[:63]

def process_sequence_folder(folder_path):
    """한 시퀀스 폴더 내의 json들을 읽어 (30, 126) 넘파이 배열 생성"""
    json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))
    if len(json_files) == 0:
        return None

    # 프레임 수가 부족하면 패딩, 많으면 30개로 균등 샘플링
    if len(json_files) >= SEQUENCE_LENGTH:
        indices = np.linspace(0, len(json_files) - 1, SEQUENCE_LENGTH, dtype=int)
        selected_files = [json_files[i] for i in indices]
    else:
        # 프레임 부족 시 마지막 프레임 복제 패딩
        selected_files = json_files + [json_files[-1]] * (SEQUENCE_LENGTH - len(json_files))

    sequence_data = []
    for jf in selected_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                content = json.load(f)

            people = content.get("people", {})
            if isinstance(people, list):
                people = people[0] if len(people) > 0 else {}

            lh_raw = people.get("hand_left_keypoints_2d", [])
            rh_raw = people.get("hand_right_keypoints_2d", [])

            lh = parse_hand_points(lh_raw)
            rh = parse_hand_points(rh_raw)

            # 왼손(63) + 오른손(63) = 126
            frame_126 = np.concatenate([lh, rh])
            sequence_data.append(frame_126)
        except Exception:
            sequence_data.append(np.zeros(126, dtype=np.float32))

    return np.array(sequence_data)  # shape: (30, 126)

# 3. 변환 메인 루프
print("\n[*] AI-Hub 데이터셋 변환을 시작합니다...")

for word_code, label_name in TARGET_WORDS.items():
    print(f"\n>> 단어 처리 중: [{label_name}] (코드: {word_code})")
    
    # 해당 단어의 정면(_F) 및 기타 카메라 폴더들 수집
    pattern = os.path.join(DATA_DIR, f"NIA_SL_{word_code}_*")
    seq_folders = glob.glob(pattern)
    
    if not seq_folders:
        print(f"   [!] 폴더를 찾지 못했습니다: {pattern}")
        continue

    seq_idx = 0
    for s_folder in seq_folders:
        seq_array = process_sequence_folder(s_folder)
        if seq_array is None:
            continue

        target_dir = os.path.join(OUTPUT_DIR, label_name, str(seq_idx))
        os.makedirs(target_dir, exist_ok=True)

        for frame_num in range(SEQUENCE_LENGTH):
            save_path = os.path.join(target_dir, f"{frame_num}.npy")
            np.save(save_path, seq_array[frame_num])

        seq_idx += 1

    print(f"   [완료] 총 {seq_idx}개의 시퀀스 생성 완료 (위치: MP_Data/{label_name}/)")

print("\n🎉 모든 데이터 변환이 성공적으로 완료되었습니다!")