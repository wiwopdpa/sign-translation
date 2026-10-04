import os
import glob
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# temp_data 내의 모든 keypoint 폴더 탐색 (01, 02 등 하위 전체)
DATA_ROOT = os.path.join(BASE_DIR, "temp_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "MP_Data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_WORDS = {
    "WORD0001": "hello",
    "WORD0003": "thanks",
    "WORD0004": "happy",
    "WORD0005": "sad",
    "WORD0006": "iloveyou"
}

IMG_WIDTH = 1920.0
IMG_HEIGHT = 1080.0
SEQUENCE_LENGTH = 30

def parse_hand_points(raw_list):
    if not raw_list or len(raw_list) < 63:
        return np.zeros(63, dtype=np.float32)
    pts = np.array(raw_list, dtype=np.float32)
    pts[0::3] /= IMG_WIDTH
    pts[1::3] /= IMG_HEIGHT
    return pts[:63]

def process_sequence_folder(folder_path):
    json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))
    if len(json_files) == 0:
        return None

    if len(json_files) >= SEQUENCE_LENGTH:
        indices = np.linspace(0, len(json_files) - 1, SEQUENCE_LENGTH, dtype=int)
        selected_files = [json_files[i] for i in indices]
    else:
        selected_files = json_files + [json_files[-1]] * (SEQUENCE_LENGTH - len(json_files))

    sequence_data = []
    for jf in selected_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                content = json.load(f)
            people = content.get("people", {})
            if isinstance(people, list):
                people = people[0] if len(people) > 0 else {}
            lh = parse_hand_points(people.get("hand_left_keypoints_2d", []))
            rh = parse_hand_points(people.get("hand_right_keypoints_2d", []))
            sequence_data.append(np.concatenate([lh, rh]))
        except Exception:
            sequence_data.append(np.zeros(126, dtype=np.float32))

    return np.array(sequence_data)

print("[*] 확장 데이터 변환 시작...")

for word_code, label_name in TARGET_WORDS.items():
    # 모든 하위 폴더에서 해당 단어 코드가 들어간 시퀀스 폴더 전부 검색
    pattern = os.path.join(DATA_ROOT, "**", f"NIA_SL_{word_code}_*")
    seq_folders = [f for f in glob.glob(pattern, recursive=True) if os.path.isdir(f)]
    
    print(f"\n>> [{label_name}] 검색된 폴더 수: {len(seq_folders)}개")
    
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

    print(f"   -> [{label_name}] 총 {seq_idx}개 시퀀스 변환 완료")

print("\n🎉 모든 데이터 증강 변환 완료!")