import os
import json
import glob
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(CURRENT_DIR, "temp_data")
DATA_PATH = os.path.join(CURRENT_DIR, "MP_Data")
os.makedirs(DATA_PATH, exist_ok=True)

ACTIONS = np.array(['hello', 'thanks', 'happy', 'sad', 'iloveyou'])
WORD_MAP = {
    'WORD0001': 'hello',
    'WORD0003': 'thanks',
    'WORD0004': 'happy',
    'WORD0005': 'sad',
    'WORD0006': 'iloveyou'
}
SEQUENCE_LENGTH = 30

def normalize_hand(points):
    """21개 (x, y, z) 랜드마크를 손목(0번) 기준 상대 좌표로 정규화 (총 63차원)"""
    arr = np.array(points, dtype=np.float32)
    if arr.size == 0 or np.all(arr == 0):
        return np.zeros(63, dtype=np.float32)
    
    try:
        arr = arr.reshape(21, 3)
    except Exception:
        return np.zeros(63, dtype=np.float32)

    wrist = arr[0].copy()
    rel = arr - wrist
    
    max_val = np.max(np.abs(rel))
    if max_val > 1e-5:
        rel = rel / max_val
        
    return rel.flatten()

def find_hand_landmarks(data):
    """
    JSON 구조(루트 키, people 리스트, people 딕셔너리 등)에 상관없이 
    왼손/오른손 랜드마크 배열을 안전하게 추출
    """
    candidate_dicts = []
    
    # 1) data 자체가 딕셔너리인 경우 후보에 추가
    if isinstance(data, dict):
        candidate_dicts.append(data)
        
        # people 필드 탐색
        people = data.get("people")
        if isinstance(people, list):
            for item in people:
                if isinstance(item, dict):
                    candidate_dicts.append(item)
        elif isinstance(people, dict):
            for item in people.values():
                if isinstance(item, dict):
                    candidate_dicts.append(item)

    lh, rh = [], []
    left_keys = ["left_hand_pts", "hand_left_pts", "left_hand_landmarks", "left_hand"]
    right_keys = ["right_hand_pts", "hand_right_pts", "right_hand_landmarks", "right_hand"]

    for d in candidate_dicts:
        if not lh:
            for k in left_keys:
                if k in d and isinstance(d[k], list) and len(d[k]) >= 63:
                    lh = d[k][:63]
                    break
        if not rh:
            for k in right_keys:
                if k in d and isinstance(d[k], list) and len(d[k]) >= 63:
                    rh = d[k][:63]
                    break
        if lh and rh:
            break

    return lh, rh

def process_aihub_data():
    print("=" * 60)
    print("AI-Hub 수어 데이터 정규화 및 NPY 변환 시작")
    print("=" * 60)

    for word_code, action in WORD_MAP.items():
        action_dir = os.path.join(DATA_PATH, action)
        os.makedirs(action_dir, exist_ok=True)
        
        search_pattern = os.path.join(RAW_DATA_PATH, "**", f"*{word_code}*")
        matched_folders = [p for p in glob.glob(search_pattern, recursive=True) if os.path.isdir(p)]
        
        if not matched_folders:
            matched_folders = [p for p in glob.glob(os.path.join(RAW_DATA_PATH, f"*{word_code}*")) if os.path.isdir(p)]
            
        print(f"\n[단어: {action} ({word_code})] 발견된 폴더: {len(matched_folders)}개")
        seq_idx = 0

        for folder in matched_folders:
            json_files = sorted(glob.glob(os.path.join(folder, "*.json")))
            if len(json_files) == 0:
                continue

            indices = np.linspace(0, len(json_files) - 1, SEQUENCE_LENGTH, dtype=int)
            sequence_data = []

            for idx in indices:
                with open(json_files[idx], 'r', encoding='utf-8') as f:
                    data = json.load(f)

                lh, rh = find_hand_landmarks(data)

                lh_norm = normalize_hand(lh)
                rh_norm = normalize_hand(rh)
                combined = np.concatenate([lh_norm, rh_norm])
                sequence_data.append(combined)

            save_folder = os.path.join(action_dir, str(seq_idx))
            os.makedirs(save_folder, exist_ok=True)
            np.save(os.path.join(save_folder, "0.npy"), np.array(sequence_data, dtype=np.float32))
            seq_idx += 1

        print(f" -> [{action}] 총 {seq_idx}개 시퀀스 변환 완료")

if __name__ == "__main__":
    process_aihub_data()