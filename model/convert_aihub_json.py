import os
import json
import numpy as np

def parse_single_json(json_path):
    """JSON 파일 하나에서 126차원(왼손 63 + 오른손 63) 키포인트를 추출합니다."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    people = data.get('people', {})
    
    # 3D 좌표 우선 사용, 없을 경우 2D 사용
    left_raw = people.get('hand_left_keypoints_3d', [])
    right_raw = people.get('hand_right_keypoints_3d', [])

    if not left_raw:
        left_raw = people.get('hand_left_keypoints_2d', [])
    if not right_raw:
        right_raw = people.get('hand_right_keypoints_2d', [])

    # 왼손 63개 (x, y, z)
    lh = []
    if left_raw and len(left_raw) >= 84:
        for i in range(0, 84, 4):
            lh.extend([left_raw[i], left_raw[i+1], left_raw[i+2]])
    else:
        lh = [0.0] * 63

    # 오른손 63개 (x, y, z)
    rh = []
    if right_raw and len(right_raw) >= 84:
        for i in range(0, 84, 4):
            rh.extend([right_raw[i], right_raw[i+1], right_raw[i+2]])
    else:
        rh = [0.0] * 63

    return np.array(lh + rh, dtype=np.float32)

def convert_single_folder(json_folder_path, action_name, sequence_idx, save_dir="MP_Data"):
    """폴더 하나 안의 JSON 시퀀스를 30프레임 .npy로 변환하여 저장합니다."""
    files = sorted([f for f in os.listdir(json_folder_path) if f.endswith('.json')])
    total_frames = len(files)

    if total_frames < 30:
        print(f"⚠️ 프레임 부족 건너뜀 ({total_frames}개): {os.path.basename(json_folder_path)}")
        return False

    indices = np.linspace(0, total_frames - 1, 30, dtype=int)
    sampled_files = [files[i] for i in indices]

    target_dir = os.path.join(save_dir, action_name, str(sequence_idx))
    os.makedirs(target_dir, exist_ok=True)

    for frame_idx, filename in enumerate(sampled_files):
        full_path = os.path.join(json_folder_path, filename)
        kp_126 = parse_single_json(full_path)
        np.save(os.path.join(target_dir, f"{frame_idx}.npy"), kp_126)

    print(f"✅ 변환 완료: [{action_name}] 시퀀스 {sequence_idx} ({os.path.basename(json_folder_path)})")
    return True

def batch_convert(root_dir, save_dir="MP_Data"):
    """최상위 폴더 내의 모든 _F(정면) 폴더를 자동으로 찾아 일괄 변환합니다."""
    folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f)) and f.endswith('_F')]
    folders.sort()

    if not folders:
        print(f"⚠️ 지정한 폴더 내에 '_F'로 끝나는 폴더가 없습니다. 경로를 확인해 주세요: {root_dir}")
        return

    word_seq_counter = {}

    for folder_name in folders:
        # 폴더명에서 단어 ID 추출 (예: NIA_SL_WORD1501_REAL01_F -> word1501)
        parts = folder_name.split('_')
        word_label = parts[2].lower() if len(parts) >= 3 else folder_name.lower()

        if word_label not in word_seq_counter:
            word_seq_counter[word_label] = 0

        folder_path = os.path.join(root_dir, folder_name)
        seq_idx = word_seq_counter[word_label]

        success = convert_single_folder(folder_path, word_label, seq_idx, save_dir)
        if success:
            word_seq_counter[word_label] += 1

    print("\n🎉 모든 정면 데이터 변환 작업이 완료되었습니다!")

if __name__ == "__main__":
    # 다운로드 및 압축 해제된 실제 경로 적용 완료
    ROOT_DATA_DIR = r"C:\Users\82102\Downloads\New_sample (1)\라벨링데이터\REAL\WORD\01_real_word_keypoint"
    
    batch_convert(ROOT_DATA_DIR, save_dir="MP_Data")