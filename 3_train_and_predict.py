# -*- coding: utf-8 -*-
"""
Step 3 — Train CatBoost + XGBoost ensemble and generate submission
===================================================================
Input:  final_clean_features.csv, punches.csv, videos.csv, videos_train.csv
Output: submission_TITAN_ENSEMBLE.csv

Detection:  50/50 blend of CatBoost + XGBoost, top-1594 frames by probability.
Attributes: CatBoost classifiers for fighter, punch_type, hand, target, effectiveness.
"""

import pandas as pd
import numpy as np
import warnings
from catboost import CatBoostClassifier
import xgboost as xgb

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_CSV    = 'final_clean_features.csv'
PUNCHES_CSV     = 'punches.csv'
TRAIN_META_CSV  = 'videos_train.csv'   # competition "videos (1).csv"
TEST_META_CSV   = 'videos.csv'
OUTPUT_CSV      = 'submission_TITAN_ENSEMBLE.csv'

# ── Config ────────────────────────────────────────────────────────────────────
TOP_N           = 1594    # total predictions (matches sample_submission row count)
CLEAR_TRUE_N    = 750     # top-N marked as clear=true
PUNCH_TOLERANCE = 3       # frames tolerance for label matching
ATTR_TOLERANCE  = 5       # frames tolerance for attribute matching
USE_GPU         = True    # set False if no GPU available


def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print('1. Loading data...')
    df_features  = pd.read_csv(FEATURES_CSV)
    df_train_meta = pd.read_csv(TRAIN_META_CSV)
    df_test_meta  = pd.read_csv(TEST_META_CSV)
    df_punches    = pd.read_csv(PUNCHES_CSV)

    base_cols = [c for c in df_features.columns
                 if c not in ('video_id', 'video_name', 'frame', 'norm_name')]
    df_features[base_cols] = df_features[base_cols].fillna(0)

    train_ids = df_train_meta['video_id'].unique()
    test_ids  = df_test_meta['video_id'].unique()

    train_ts  = df_features[df_features['video_id'].isin(train_ids)].copy()
    test_ts   = df_features[df_features['video_id'].isin(test_ids)].copy()

    feature_cols = [c for c in df_features.columns
                    if c not in ('video_id', 'video_name', 'frame', 'norm_name')]

    print(f'   Train rows: {len(train_ts):,}   Test rows: {len(test_ts):,}')
    print(f'   Feature columns: {len(feature_cols)}')

    # ── 2. Build detection target ─────────────────────────────────────────────
    print('2. Building punch detection labels...')
    punch_labels = df_punches[['video_id', 'frame']].copy()
    punch_labels['is_punch'] = 1

    merged_train = pd.merge_asof(
        train_ts.sort_values('frame'),
        punch_labels.sort_values('frame'),
        on='frame', by='video_id',
        direction='nearest', tolerance=PUNCH_TOLERANCE
    )
    merged_train['is_punch'] = merged_train['is_punch'].fillna(0).astype(int)
    pos  = merged_train['is_punch'].sum()
    neg  = len(merged_train) - pos
    pos_weight = neg / max(1, pos)
    print(f'   Positive: {pos:,}   Negative: {neg:,}   pos_weight: {pos_weight:.1f}')

    # ── 3. Train detectors ────────────────────────────────────────────────────
    print('3. Training detection ensemble...')
    task_type = 'GPU' if USE_GPU else 'CPU'
    tree_method = 'hist'
    device = 'cuda' if USE_GPU else 'cpu'

    print('   → CatBoost detector...')
    cb_det = CatBoostClassifier(
        iterations=800, depth=6, learning_rate=0.05,
        scale_pos_weight=pos_weight,
        random_seed=42, verbose=0, task_type=task_type)
    cb_det.fit(merged_train[feature_cols], merged_train['is_punch'])

    print('   → XGBoost detector...')
    xgb_det = xgb.XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.05,
        scale_pos_weight=pos_weight,
        random_state=42, tree_method=tree_method, device=device)
    xgb_det.fit(merged_train[feature_cols], merged_train['is_punch'])

    # ── 4. Predict on test ────────────────────────────────────────────────────
    print('4. Predicting on test set...')
    cb_probs   = cb_det.predict_proba(test_ts[feature_cols])[:, 1]
    xgb_probs  = xgb_det.predict_proba(test_ts[feature_cols])[:, 1]
    test_ts    = test_ts.copy()
    test_ts['punch_prob'] = (cb_probs + xgb_probs) / 2.0

    # Top-N frames → submission rows
    top = (test_ts.sort_values('punch_prob', ascending=False)
                  .head(TOP_N).copy()
                  .reset_index(drop=True))
    top['clear'] = 'false'
    top.loc[:CLEAR_TRUE_N - 1, 'clear'] = 'true'
    top = top.sort_values(['video_id', 'frame']).reset_index(drop=True)
    print(f'   Selected {len(top)} frames  (clear=true: {CLEAR_TRUE_N})')

    # ── 5. Train attribute classifiers ────────────────────────────────────────
    print('5. Training attribute classifiers...')
    train_exact = pd.merge_asof(
        train_ts.sort_values('frame'),
        df_punches[['video_id', 'frame',
                    'fighter', 'punch_type', 'hand', 'target', 'effectiveness']]
                  .sort_values('frame'),
        on='frame', by='video_id',
        direction='nearest', tolerance=ATTR_TOLERANCE
    ).dropna(subset=['punch_type'])

    targets = ['fighter', 'punch_type', 'hand', 'target', 'effectiveness']
    for target in targets:
        print(f'   → {target}...')
        clf = CatBoostClassifier(
            iterations=900, depth=7, learning_rate=0.04,
            random_seed=42, verbose=0, task_type=task_type)
        clf.fit(train_exact[feature_cols], train_exact[target].astype(str))
        top[target] = clf.predict(top[feature_cols]).flatten()

    # ── 6. Format submission ──────────────────────────────────────────────────
    print('6. Formatting submission...')
    final = top.merge(
        df_test_meta[['video_id', 'agn_index', 'video_key']],
        on='video_id', how='left')
    final['id'] = range(1, len(final) + 1)
    final = final[['id', 'video_id', 'agn_index', 'video_key',
                   'frame', 'fighter', 'punch_type', 'hand',
                   'target', 'effectiveness', 'clear']]
    final.to_csv(OUTPUT_CSV, index=False)

    print(f'\n✅ Done!  Saved {OUTPUT_CSV}  ({len(final)} rows)')
    print(final.head(5).to_string())


if __name__ == '__main__':
    main()
