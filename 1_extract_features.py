# -*- coding: utf-8 -*-
"""
Step 1 — Extract per-frame features from skeleton CSVs
=======================================================
Input:  дата/*_features.csv  (frame | red_kps | blue_kps)
Output: final_clean_features.csv

Each *_features.csv contains pre-extracted COCO keypoints for both fighters.
This script computes per-frame biomechanical features and maps every file to
the official video_id from punches.csv / videos.csv.
"""

import ast, os, re, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR     = 'дата'
PUNCHES_CSV     = 'punches.csv'
TRAIN_META_CSV  = 'videos_train.csv'   # competition "videos (1).csv"
TEST_META_CSV   = 'videos.csv'
OUTPUT_CSV      = 'final_clean_features.csv'

# ── COCO keypoint indices ──────────────────────────────────────────────────────
L_SHOULDER, R_SHOULDER = 5,  6
L_ELBOW,    R_ELBOW    = 7,  8
L_WRIST,    R_WRIST    = 9,  10
L_HIP,      R_HIP      = 11, 12
L_KNEE,     R_KNEE     = 13, 14
NUM_JOINTS  = 17
CONF_THRESH = 0.4


# ── Name-based file → video_id mapping ────────────────────────────────────────

def _norm_names(s: str) -> str:
    """Normalise fighter names for matching.
    'Петросян-Зайнуллаев'   → 'петросянзайнуллаев'
    'Дашипылов и Санников'  → 'дашипыловсанников'
    """
    s = re.sub(r'(?<![а-яёА-ЯЁ])и(?![а-яёА-ЯЁ])', '', str(s), flags=re.IGNORECASE)
    return re.sub(r'[^а-яёА-ЯЁ]', '', s).lower()

def _names_from_folder(fight_folder: str) -> str:
    s = re.sub(r'^[Бб]ой\s*\d+[\.\s]+', '', str(fight_folder))
    return _norm_names(s)

def _names_from_file(fname: str) -> str:
    stem = fname.replace('_features.csv', '')
    s = re.sub(r'^[Бб]ой[\s_]*\d+[\s_\.]+', '', stem)
    s = re.sub(r'[\s_][Рр]аунд[\s_]*\d+$', '', s)
    return _norm_names(s)

def _round_from_file(fname: str):
    m = re.search(r'[Рр]аунд[\s_]*(\d+)', fname)
    return int(m.group(1)) if m else None

def _boks_key(s: str) -> str:
    s = re.sub(r'_features\.csv$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^бокс_', '', s, flags=re.IGNORECASE)
    return re.sub(r'[^a-z0-9]', '', os.path.splitext(s)[0].lower())


def build_file_map(dataset_dir, punches_df, train_meta, test_meta) -> dict:
    all_vids = pd.concat([
        punches_df[['video_id', 'fight_folder', 'round_number', 'source_video']]
                  .drop_duplicates('video_id'),
        train_meta[['video_id', 'fight_folder', 'round_number', 'source_video']]
                  .drop_duplicates('video_id'),
        test_meta[['video_id',  'fight_folder', 'round_number', 'source_video']]
                  .drop_duplicates('video_id'),
    ]).drop_duplicates('video_id')

    tourney_lookup, boks_lookup = {}, {}
    for _, row in all_vids.iterrows():
        vid = row['video_id']
        ff  = str(row['fight_folder'])
        rn  = row['round_number']
        sv  = str(row['source_video'])
        if ff.lower() == 'бокс':
            boks_lookup[_boks_key(sv)] = vid
        elif pd.notna(rn):
            tourney_lookup[(_names_from_folder(ff), int(float(rn)))] = vid

    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('_features.csv')]
    file_map, unmatched = {}, []
    for fname in sorted(csv_files):
        rn  = _round_from_file(fname)
        vid = None
        if fname.lower().startswith('бокс_'):
            vid = boks_lookup.get(_boks_key(fname))
        elif rn is not None:
            vid = tourney_lookup.get((_names_from_file(fname), rn))
        if vid:
            file_map[vid] = os.path.join(dataset_dir, fname)
        else:
            unmatched.append(fname)

    print(f'  Matched: {len(file_map)} / {len(csv_files)}')
    for u in unmatched:
        print(f'  UNMATCHED: {u}')
    return file_map


# ── Keypoint parsing ───────────────────────────────────────────────────────────

def load_keypoints(path: str) -> tuple:
    """Returns (red_kps, blue_kps) both shape (N, 17, 3)."""
    df       = pd.read_csv(path)
    n_frames = int(df['frame'].max()) + 1
    red_kps  = np.zeros((n_frames, 17, 3), dtype=np.float32)
    blue_kps = np.zeros((n_frames, 17, 3), dtype=np.float32)
    for _, row in df.iterrows():
        f = int(row['frame'])
        for col, arr in [('red_kps', red_kps), ('blue_kps', blue_kps)]:
            val = row[col]
            if isinstance(val, float) and np.isnan(val):
                continue
            try:
                arr[f] = np.array(ast.literal_eval(val), dtype=np.float32)
            except Exception:
                pass
    return red_kps, blue_kps


# ── Feature computation ────────────────────────────────────────────────────────

def _angle(a, b, c):
    """Angle at vertex b given points a-b-c. Returns degrees."""
    ba = a - b;  bc = c - b
    cos = np.clip(np.sum(ba*bc, axis=-1) /
                  (np.linalg.norm(ba, axis=-1)*np.linalg.norm(bc, axis=-1) + 1e-8), -1, 1)
    return np.degrees(np.arccos(cos))


def compute_features(red_kps: np.ndarray, blue_kps: np.ndarray) -> pd.DataFrame:
    """
    red_kps, blue_kps: (N, 17, 3)
    Returns DataFrame with N rows and ~60 biomechanical features.
    """
    N   = red_kps.shape[0]
    eps = 1e-8

    def xy(kps, idx):  return kps[:, idx, :2]
    def conf(kps, idx): return kps[:, idx, 2]

    rows = []
    for t in range(N):
        r = red_kps[t];   b = blue_kps[t]
        feat = {'frame': t}

        # ── Wrist speeds (velocity magnitude) ──
        for prefix, kps in [('red', r), ('blue', b)]:
            for name, idx in [('lwrist', L_WRIST), ('rwrist', R_WRIST),
                               ('lelbow', L_ELBOW), ('relbow', R_ELBOW)]:
                prev = red_kps[t-1] if prefix == 'red' else blue_kps[t-1]
                curr_pt = kps[idx, :2]
                prev_pt = prev[idx, :2]
                speed   = np.linalg.norm(curr_pt - prev_pt) if t > 0 else 0.
                feat[f'{prefix}_{name}_speed'] = float(speed)
                feat[f'{prefix}_{name}_x']     = float(curr_pt[0])
                feat[f'{prefix}_{name}_y']     = float(curr_pt[1])
                feat[f'{prefix}_{name}_conf']  = float(kps[idx, 2])

        # ── Inter-fighter distances ──
        r_center = (r[L_HIP, :2] + r[R_HIP, :2]) / 2
        b_center = (b[L_HIP, :2] + b[R_HIP, :2]) / 2
        feat['fighter_dist']    = float(np.linalg.norm(r_center - b_center))
        feat['fighter_dist_x']  = float(r_center[0] - b_center[0])
        feat['fighter_dist_y']  = float(r_center[1] - b_center[1])

        # Red wrist → Blue torso distances
        b_torso = (b[L_SHOULDER, :2] + b[R_SHOULDER, :2]) / 2
        feat['red_lwrist_to_blue_torso'] = float(np.linalg.norm(r[L_WRIST,:2] - b_torso))
        feat['red_rwrist_to_blue_torso'] = float(np.linalg.norm(r[R_WRIST,:2] - b_torso))
        # Blue wrist → Red torso distances
        r_torso = (r[L_SHOULDER, :2] + r[R_SHOULDER, :2]) / 2
        feat['blue_lwrist_to_red_torso'] = float(np.linalg.norm(b[L_WRIST,:2] - r_torso))
        feat['blue_rwrist_to_red_torso'] = float(np.linalg.norm(b[R_WRIST,:2] - r_torso))

        # ── Elbow angles ──
        for prefix, kps in [('red', r), ('blue', b)]:
            for side, sh, el, wr in [('left',  L_SHOULDER, L_ELBOW, L_WRIST),
                                      ('right', R_SHOULDER, R_ELBOW, R_WRIST)]:
                if kps[sh,2]>CONF_THRESH and kps[el,2]>CONF_THRESH and kps[wr,2]>CONF_THRESH:
                    ang = _angle(kps[sh,:2], kps[el,:2], kps[wr,:2])
                else:
                    ang = 0.
                feat[f'{prefix}_{side}_elbow_angle'] = float(ang)

        # ── Shoulder width (proxy for body rotation) ──
        feat['red_shoulder_width']  = float(np.linalg.norm(r[L_SHOULDER,:2]-r[R_SHOULDER,:2]))
        feat['blue_shoulder_width'] = float(np.linalg.norm(b[L_SHOULDER,:2]-b[R_SHOULDER,:2]))

        # ── Average confidence ──
        feat['red_avg_conf']  = float(r[:, 2].mean())
        feat['blue_avg_conf'] = float(b[:, 2].mean())

        rows.append(feat)

    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('Loading metadata...')
    punches_df = pd.read_csv(PUNCHES_CSV)
    train_meta = pd.read_csv(TRAIN_META_CSV)
    test_meta  = pd.read_csv(TEST_META_CSV)

    print('Building file map...')
    file_map = build_file_map(DATASET_DIR, punches_df, train_meta, test_meta)

    all_frames = []
    for vid, path in sorted(file_map.items()):
        print(f'  Processing {vid} ...')
        red_kps, blue_kps = load_keypoints(path)
        df = compute_features(red_kps, blue_kps)
        df.insert(0, 'video_id',   vid)
        df.insert(1, 'video_name', os.path.basename(path).replace('_features.csv', ''))
        all_frames.append(df)

    out = pd.concat(all_frames, ignore_index=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f'\nSaved {OUTPUT_CSV}  ({len(out):,} rows, {out.shape[1]} columns)')


if __name__ == '__main__':
    main()
