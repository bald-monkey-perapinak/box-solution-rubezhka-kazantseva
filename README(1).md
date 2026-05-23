# Boxing Action Recognition

Solution for the **RASCAR Boxing Action Recognition Challenge** — automatic detection and classification of punches in amateur boxing videos.

---

## Approach

```
*_features.csv  →  per-frame features  →  time-series augmentation  →  CatBoost + XGBoost ensemble  →  submission.csv
```

### Pipeline overview

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `1_extract_features.py` | Loads pre-extracted skeleton CSVs, normalises poses, computes wrist speeds, inter-fighter distances, joint angles → saves `final_clean_features.csv` |
| 2 | `2_build_timeseries.py` | Adds rolling statistics (shift ±1, diff, max/mean/std over windows of 5 and 15 frames) to every speed feature |
| 3 | `3_train_and_predict.py` | Trains CatBoost + XGBoost punch detector (50/50 blend), trains CatBoost attribute classifiers (fighter, punch_type, hand, target, effectiveness), outputs `submission_TITAN_ENSEMBLE.csv` |

### Key design decisions

- **Skeleton data** (`*_features.csv`) contains per-frame COCO keypoints for `red` and `blue` fighters — no raw video needed after extraction.
- **Matching** `*_features.csv` filenames to `video_id` is done by normalising fighter names from `fight_folder` (handles both hyphen-separated and `_и_`-separated formats).
- **NaN handling** — empty frames (missing detections) are filled with zeros before feature computation.
- **Class imbalance** — `scale_pos_weight` computed from actual punch/non-punch ratio per dataset.
- **Detection** — top-1594 frames by blended punch probability; `clear=true` for top-750.

---

## Reproduce

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Data layout

Place the following files in the project root:

```
boxing-action-recognition/
├── punches.csv            ← competition training labels
├── videos.csv             ← competition test video metadata
├── videos_train.csv       ← competition training video metadata  (videos (1).csv)
└── дата/                  ← pre-extracted skeleton CSVs
    ├── Бой1._Петросян-Зайнуллаев_Раунд1_features.csv
    ├── Бой_1._Дашипылов_и_Санников_Раунд1_features.csv
    ├── бокс_IMG_7390_features.csv
    └── ...  (72 files total)
```

Skeleton CSVs have columns: `frame`, `red_kps`, `blue_kps`  
Each keypoint cell contains a Python-literal list of 17 points: `[[x, y, conf], ...]`

### 3. Run

```bash
python 1_extract_features.py   # → final_clean_features.csv
python 2_build_timeseries.py   # reads final_clean_features.csv, writes to it in place
python 3_train_and_predict.py  # → submission_TITAN_ENSEMBLE.csv
```

Or run everything in one go inside Google Colab:  
`notebooks/full_pipeline.ipynb`

---

## Repository structure

```
boxing-action-recognition/
├── README.md
├── requirements.txt
├── .gitignore
├── 1_extract_features.py
├── 2_build_timeseries.py
├── 3_train_and_predict.py
└── notebooks/
    └── full_pipeline.ipynb
```

> **Note:** `final_clean_features.csv` and raw video files are **not** committed (see `.gitignore`). Generate `final_clean_features.csv` by running step 1.
