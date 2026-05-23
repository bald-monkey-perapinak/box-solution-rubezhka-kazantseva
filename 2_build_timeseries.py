# -*- coding: utf-8 -*-
"""
Step 2 — Build time-series features
=====================================
Input:  final_clean_features.csv
Output: final_clean_features.csv  (extended in-place with rolling statistics)

For every speed column adds:
  shift ±1, diff, rolling max/mean (window=5), rolling mean/std (window=15)
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

INPUT_CSV  = 'final_clean_features.csv'
OUTPUT_CSV = 'final_clean_features.csv'   # overwrite in place


def add_timeseries_features(df: pd.DataFrame) -> pd.DataFrame:
    speed_cols = [c for c in df.columns if 'speed' in c]
    print(f'  Speed columns found: {len(speed_cols)}')

    for col in speed_cols:
        g = df.groupby('video_id')[col]

        df[col + '_shift_1']  = g.shift(1).fillna(0)
        df[col + '_shift_m1'] = g.shift(-1).fillna(0)
        df[col + '_diff']     = g.diff().fillna(0)

        df[col + '_max5']  = (g.rolling(5,  center=True, min_periods=1)
                               .max().reset_index(0, drop=True))
        df[col + '_mean5'] = (g.rolling(5,  center=True, min_periods=1)
                               .mean().reset_index(0, drop=True))

        df[col + '_mean15'] = (g.rolling(15, center=True, min_periods=1)
                                .mean().reset_index(0, drop=True))
        df[col + '_std15']  = (g.rolling(15, center=True, min_periods=1)
                                .std().fillna(0).reset_index(0, drop=True))

    return df


def main():
    print(f'Loading {INPUT_CSV} ...')
    df = pd.read_csv(INPUT_CSV)
    print(f'  Rows: {len(df):,}   Columns before: {df.shape[1]}')

    df = df.sort_values(['video_id', 'frame']).reset_index(drop=True)
    df = add_timeseries_features(df)

    print(f'  Columns after:  {df.shape[1]}')
    df.to_csv(OUTPUT_CSV, index=False)
    print(f'Saved {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
