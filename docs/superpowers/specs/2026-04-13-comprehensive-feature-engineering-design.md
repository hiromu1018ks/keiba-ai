# 包括的特徴量エンジニアリング設計書

**日付:** 2026-04-13
**対象ブランチ:** main (4655d84)
**バックログ:** `docs/backlog/2026-04-13-regime-detector-fix.md` 他 5件

---

## 概要

ROI 98.8% (損益分岐点) の現状から、PIT安全性を最優先に確保しながら
4フェーズの段階的特徴量改善を実施する。各フェーズでバックテスト検証を行い、
客観的なROI変動を測定する。

### 現状のバックテスト結果 (ベースライン)

| 指標 | 値 |
|------|-----|
| ROI | 98.8% |
| ベット数 | 499 |
| 利益 | -¥610 |
| 最大DD | 4.0% |
| 学習 | 2021-2024 / テスト: 2025 / ensemble / flat / JRAのみ |

---

## Phase 1: 既存特徴量の穴埋めと活用

### 目的

Stage1の37特徴量のうち2つが常にNaN (モデル容量の無駄) と、
定義済みだがパイプラインにwiringされていない5特徴量の活用による選別精度改善。

**正確な特徴量数 (実コード確認済):**

| モデル | 現在のFEATURE_COLS数 |
|--------|---------------------|
| `AbilityModel` | 37 |
| `PlaceAbilityModel` | 37 |
| `WinTwoStageModel` | 16 |
| `PlaceTwoStageModel` | 16 |
| `EVCorrectionModel` | 23 |
| `RaceQualityScreener` | 20 |

### 1-1. `blood_condition_wr` の実装

**現状:** `bloodline_features.py:112` で常に `np.nan`。

**実装内容:**
- `src/features/horse_career_stats.py` に `baba_cd` 別の累積成績計算を追加:
  - `turf_good_starts`, `turf_good_wins` (baba_cd in [1,2] の芝)
  - `turf_heavy_starts`, `turf_heavy_wins` (baba_cd in [3,4] の芝)
  - `dirt_good_starts`, `dirt_good_wins` (baba_cd in [1,2] のダート)
  - `dirt_heavy_starts`, `dirt_heavy_wins` (baba_cd in [3,4] のダート)
  - **データソース:** `baba_cd` は `races_df["track_condition_code"]` から取得。
    現在 `horse_career_stats.py` は `trackcd` と `kyori` を `races_df` からマージしているが、
    `track_condition_code` (baba_cd) も同時にマージする必要がある。
- `src/features/bloodline_features.py` でレースの `surface` + `baba_cd` に応じて
  該当する条件別勝率を Beta(1,10) 平滑化して出力
- `scripts/precompute_career_stats.py` を更新して新しい列を出力

**PIT保証:** 既存の `shift(1)` 机制により、レース当日以前の成績のみ使用。
`horse_career_stats.parquet` の累積値は既に PIT 安全。

**変更ファイル:**
| ファイル | 変更内容 |
|---------|----------|
| `src/features/horse_career_stats.py` | baba_cd 別累積統計の追加 |
| `src/features/bloodline_features.py` | `blood_condition_wr` の計算実装 |
| `scripts/precompute_career_stats.py` | 出力列の追加 |
| `tests/test_bloodline_features.py` | condition_wr のテスト追加 |

### 1-2. `blood_keito_cd` の実装

**現状:** `bloodline_features.py:115` で常に `np.nan`。

**実装内容:**
- ETL設定 (`config/etl_tables.yaml`) — `n_keito` は既に設定済み (行290)。ETL実行で `data/raw/keito.parquet` が生成される。
  **前提条件:** ETLを実行して `keito.parquet` を生成済みであること。未実行の場合は先に ETL を実行。
- `src/db/readers.py` に `load_keito()` メソッドを追加
- `src/features/bloodline_features.py` で:
  - JOINパス: `entries.kettonum` → `horses.kettonum` → `horses.ketto3infohansyokunum1` → `keito.keitoucode`
  - (中間テーブル `horses` を経由する2段JOIN)
  - 系統コードをカテゴリ変数として取得
  - 欠損 (外国馬・未知種牡馬) は `"unknown"` で補完

**PIT保証:** 種牡馬の系統コードは静的 (馬の一生で不変)。PIT上は常に安全。

**変更ファイル:**
| ファイル | 変更内容 |
|---------|----------|
| `src/db/readers.py` | `load_keito()` メソッド追加 (前提: `data/raw/keito.parquet` 存在) |
| `src/features/bloodline_features.py` | `blood_keito_cd` の計算実装 |
| `tests/test_readers.py` | `load_keito` テスト |
| `tests/test_bloodline_features.py` | keito_cd テスト |

### 1-3. 計算済み未使用特徴量の活用

**重要:** 以下の5特徴量は関数として定義されているが、**パイプライン (`feature_engine.py`, `training_pipeline.py`) で一度も呼ばれていない**。FEATURE_COLSに追加する前に、まず関数の呼び出し (wiring) を追加する必要がある。

| 特徴量 | 定義関数 | 呼び出し追加先 | 組込先モデル | FEATURE_COLS変更 |
|--------|---------|--------------|-------------|-----------------|
| `odds_skewness` | `compute_flb_slope()` in `market_bias_features.py` | `feature_engine.py` の `build_all()` (又は `training_pipeline.py`) | `WinTwoStageModel`, `PlaceTwoStageModel` | 16→18 |
| `implied_prob_hhi` | `compute_flb_slope()` (同上) | 同上 | `EVCorrectionModel` | 23→24 |
| `favorite_implied_prob_ema` | `compute_roi_ema()` in `odds_dynamics_features.py` | `training_pipeline.py` の `_build_race_level_features()` 内 | `RegimeDetector` | FEATURE_COLSには `favorite_implied_prob_rolling` が既にある。`_ema`版は別名なので、どちらを使うか要確認。`_rolling`と`_ema`は計算方法が異なる (rolling平均 vs EMA)。 |
| `overround_ema` | `compute_roi_ema()` (同上) | 同上 | `RaceQualityScreener` | 20→21 |
| `entropy_ema` | `compute_roi_ema()` (同上) | 同上 | `RaceQualityScreener` | 20→22 |

**Wiring手順:**

1. `feature_engine.py` の `build_all()` で `compute_market_bias()` 呼び出し後に `compute_flb_slope()` を呼び出し、結果をマージ
2. `training_pipeline.py` で `compute_roi_ema()` を呼び出し、`favorite_implied_prob_ema`, `overround_ema`, `entropy_ema` を DataFrame にマージ
3. 各モデルの `FEATURE_COLS` に新特徴量を追加

**PIT保証:** すべて発走前オッズ由来。PIT安全。

**変更ファイル:**
| ファイル | 変更内容 |
|---------|----------|
| `src/features/feature_engine.py` | `build_all()` に `compute_flb_slope()` の呼び出し追加 |
| `src/pipelines/training_pipeline.py` | `compute_roi_ema()` の呼び出し追加 + 結果マージ |
| `src/models/two_stage_return_model.py` | FEATURE_COLS に `odds_skewness` 追加 |
| `src/models/ev_correction_model.py` | FEATURE_COLS に `implied_prob_hhi` 追加 |
| `src/models/race_quality_screener.py` | FEATURE_COLS に `overround_ema`, `entropy_ema` 追加 |

### 1-4. Phase 1 検証計画

```
バックテスト条件:
  コマンド: python scripts/run_backtest.py \
    --train-start 20210101 --train-end 20241231 \
    --test-start 20250101 --test-end 20251231 \
    --ensemble
  比較対象: 現状 ROI 98.8% (499 bets, -¥610)
  期待効果: ROI 100-110%
  成功基準: ROI > 100% またはベット選別精度の改善
```

---

## Phase 2: 真の種牡馬産駒特徴量

### 目的

現在の「血統特徴量」は馬自身のキャリア統計。真の種牡馬産駎統計を追加し、
Stage1 が「馬自身の過去戦績」と「遺伝的適性」を独立して評価できるようにする。

### 2-1. データ可用性

| 項目 | 値 |
|------|-----|
| 種牡馬ID列 | `horses.ketto3infohansyokunum1` (0% NaN) |
| 母父ID列 | `horses.ketto3infohansyokunum3` (0% NaN) |
| JRA種牡馬数 | 1,071頭 |
| 500戦以上の種牡馬 | 193頭 |
| サンプル数 | 538,665件 (JRA完了エントリ) |
| 外国馬リンク不可率 | 12.2% |

### 2-2. 新規データパイプライン

**事前計算スクリプト:** `scripts/precompute_sire_stats.py`

```
entries.parquet (kettonum, kakuteijyuni, race_date)
  ↓ JOIN races.parquet (surface, kyori, baba_cd, trackcd)
  ↓ JOIN horses.parquet (kettonum → ketto3infohansyokunum1 = sire_id)
  ↓
Group by (sire_id, race_date) → 日次集計
  ↓ cumsum().shift(1) → PIT保証 (当日の結果を含まない)
  ↓
sire_career_stats.parquet
```

**出力列:**

| 列名 | 型 | 説明 |
|------|-----|------|
| `sire_id` | str | 種牡馬血統番号 |
| `race_date` | datetime | 集計日 |
| `sire_starts` | int | 累積出走数 (当日除く) |
| `sire_wins` | int | 累積勝利数 |
| `sire_places` | int | 累積複勝数 |
| `sire_turf_starts` | int | 芝累積出走数 |
| `sire_turf_wins` | int | 芝累積勝利数 |
| `sire_dirt_starts` | int | ダート累積出走数 |
| `sire_dirt_wins` | int | ダート累積勝利数 |
| `sire_short_starts` | int | 短距離累積出走数 (<=1600m) |
| `sire_short_wins` | int | 短距離累積勝利数 |
| `sire_long_starts` | int | 長距離累積出走数 (>1600m) |
| `sire_long_wins` | int | 長距離累積勝利数 |
| `sire_prize_total` | float | 累積賞金 |

**PIT保証:** `cumsum().shift(1)` により、各行の値は該当日以前の全レース結果の累積。
当日のレース結果は含まれない。

### 2-3. 新規特徴量モジュール

**ファイル:** `src/features/sire_features.py`

| 特徴量 | 計算方法 | 説明 |
|--------|---------|------|
| `sire_wr` | `Beta(1+wins, 1+10+starts-wins)` | 種牡馬産駒全体勝率 |
| `sire_place_rate` | `Beta(1+places, 1+10+starts-places)` | 種牡馬産駒複勝率 |
| `sire_surface_wr` | レースのサーフェスに応じて芝/ダート産駒勝率を選択 | サーフェス適性 |
| `sire_distance_wr` | レースの距離に応じて短/長距離産駎勝率を選択 | 距離適性 |
| `sire_prize_avg` | `log(1 + prize_total / max(1, starts))` | 産駎平均賞金 |
| `bms_wr` | 母父 (ketto3infohansyokunum3) の産駎勝率 | 母系適性 |

**PIT保証:**
- `searchsorted(sire_id, race_date)` で PIT安全な累積統計を取得
- 種牡馬デビュー前 (産駎出走なし) → Beta(1,10) の事前分布 (9.09%) にフォールバック
- 外国種牡馬 (horses.parquetにない) → NaNのまま LightGBM に処理させる

### 2-4. モデル統合

| モデル | 追加特徴量 | 変更前 | 変更後 |
|--------|----------|--------|--------|
| `AbilityModel` (Stage1) | `sire_wr`, `sire_surface_wr`, `sire_distance_wr`, `sire_prize_avg`, `bms_wr` | 37 | 42 |
| `PlaceAbilityModel` | 同上 | 37 | 42 |
| `EVCorrectionModel` | 追加なし | 23 | 23 |

**現行 `blood_*` 特徴量は残置** (馬自身のキャリア統計として有用性は変わらない)。

### 2-5. Phase 2 検証計画

```
バックテスト条件: Phase 1 と同じ
比較対象: Phase 1 完了後のROI
期待効果: ROI +5-15pt
仮説: 種牡馬産駒勝率は馬自身のキャリアに含まれない「遺伝的適性」を捉える
失敗基準: ROI低下 > 5pt → 種牡馬特徴量のFeature Importance を分析し原因特定
```

---

## Phase 3: Market Model OOF (Out-of-Fold) 対応

### 目的

Market Model の IN-SAMPLEリーク (学習データと同じデータで予測生成) を排除。
Stage2に入る `signed_log_error_win` と `abs_log_error_win` を真の予測値にする。

### 3-1. 現状の問題フロー

```
全データ → Market Model 学習 → 全データ予測 → signed_log_error_win
                                                    ↓
                                        Stage2 の FEATURE_COLS
問題: Stage2の学習データに Market Model が既に見たデータの予測が入る
```

### 3-2. 修正後のフロー

```
全データ → 5-Fold CV (shuffle=False) → 各foldの「fold外」予測を結合
                                        ↓
                             signed_log_error_win (OOF版)
                                        ↓
                             Stage2 の FEATURE_COLS
効果: Stage2が常に「見たことないデータへの予測」で学習
```

### 3-3. 実装内容

**`src/models/market_model.py`:**
- 新メソッド `predict_oof(df, n_splits=5)` を追加
- `sklearn.model_selection.KFold(n_splits=5, shuffle=False)` — 時系列なので shuffle なし
- 各foldで train/valid 分割 → valid に対して予測
- 戻り値: `pd.Series` (OOF予測値)
- 最後に全データで再学習したモデルを保持 (推論用)

**`src/pipelines/training_pipeline.py`:**
- `MarketModel.train()` 後に `predict_oof()` を呼び出し
- OOF予測値を `signed_log_error_win` と `abs_log_error_win` に上書き (学習データのみ)
- Stage2 の入力に OOF版を使用
- 推論時は全データ学習モデルを使用 (OOFは推論に不要)

**`src/backtest/engine.py` — 変更なし:**
- バックテストでは `training_pipeline` で学習済みのモデルをそのまま使用
- OOF予測値の適用は `training_pipeline` 内で完結するため、engine.py は変更不要
- テスト期間の予測は全データ学習モデルの `predict_and_calc_error()` を使用

### 3-4. PIT安全性

- OOF は学習データ内でのCV → shuffle=False で時系列順序保持
- テストデータには一切触れない
- バックテストの年次スプリットと直交

### 3-5. Phase 3 検証計画

```
バックテスト条件: Phase 1-2 と同じ
比較対象: Phase 2 完了後のROI
成功指標: BT/PT乖離の縮小 (現状 +24.3pt → 15pt以下が目標)
注意: ROIは短期的に低下する可能性 (リーク排除)
```

---

## Phase 4: 過去走拡張 + ペース適性 + コース適性

### 目的

過去走情報の拡充による予測精度の微改善。

### 4-1. 過去走3→5走拡張

**ファイル:** `src/features/horse_history_features.py`

| 変更 | 内容 |
|------|------|
| `harontimel3_avg` → `harontimel5_avg` | 5走の3ハロンタイム平均に拡張 (NaNスキップ) |
| `harontimel3_zscore` → `harontimel5_zscore` | 5走に拡張したz-score |
| 新規: `harontime_late_trend` | 最後2走 vs 最初3走のハロンタイム差 (フォームサイクル終盤加速/減速) |

**PIT保証:** 既存の `searchsorted` 机制をそのまま使用。

### 4-2. ペース適性特徴量

**ファイル:** `src/features/pace_aptitude_features.py` (新規)

| 特徴量 | 計算方法 |
|--------|---------|
| `pace_aptitude` | 逃げ/先行有利レース vs 差し/追込有利レースでの着順差 (expanding median) |
| `front_pace_wr` | ペースが速いレース (1C上位が好走) での過去勝率 |
| `closing_pace_wr` | ペースが遅いレース (4C上位が好走) での過去勝率 |

**データソース:** 過去走の `jyuni1c` (1コーナー通過順位) と `jyuni4c` (4コーナー通過順位) と `kakuteijyuni` (着順)。レースのペースは角通過順位から推定 (明示的なペース指標はデータに存在しないため)。脚質 `kyakusitukubun_cd` は補助的に使用。

**パイプライン統合:**
- `training_pipeline.py` の特徴量計算フローに追加
- Stage1 `AbilityModel` の `FEATURE_COLS` に `pace_aptitude`, `front_pace_wr`, `closing_pace_wr` を追加
- `feature_engine.py` の `build_all()` に呼び出しを追加

**PIT保証:** `searchsorted` で過去レースのみ参照。

### 4-3. コース別適性

**ファイル:** `src/features/course_features.py` (新規)

| 特徴量 | 計算方法 |
|--------|---------|
| `course_wr` | 競馬場 (`jyo_cd`) ごとの過去勝率、Beta(1,10) 平滑化 |
| `course_distance_wr` | 競馬場 x 距離帯の過去勝率 |

**PIT保証:** `searchsorted` + `shift(1)` で当日以前の成績のみ。

### 4-4. Phase 4 検証計画

```
バックテスト条件: Phase 1-3 と同じ
比較対象: Phase 3 完了後のROI
期待効果: ROI +3-8pt
```

---

## 全体ロードマップ

```
Phase 1: 既存の穴埋め + 未使用特徴量活用
  ↓ BT検証 → ROI 100-110% 期待
Phase 2: 真の種牡馬産駒特徴量
  ↓ BT検証 → ROI 105-125% 期待
Phase 3: Market Model OOF対応
  ↓ BT検証 → BT/PT乖離縮小 (ROI低下の可能性)
Phase 4: 過去走拡張 + ペース適性 + コース適性
  ↓ BT検証 → ROI 108-130% 期待
最終: 3年度マルチBT (2023-2025) で全体堅牢性確認
```

### 各Phaseの中断基準

- ROIが前Phase比で **10pt以上低下** → 原因分析、改善または除外
- ベット数が **50件未満** に激減 → 特徴量の追加による過剰フィルタの疑い
- 最大DDが **10%以上** → リスク管理の観点から要検討

### 変更ファイル一覧 (全体)

| Phase | 新規ファイル | 変更ファイル |
|-------|------------|------------|
| P1 | なし | `feature_engine.py` (wiring追加), `training_pipeline.py` (wiring追加), `bloodline_features.py`, `horse_career_stats.py`, `precompute_career_stats.py`, `two_stage_return_model.py`, `ev_correction_model.py`, `race_quality_screener.py`, `readers.py` |
| P2 | `sire_features.py`, `precompute_sire_stats.py` | `stage1_ability_model.py`, `place_ability_model.py`, `readers.py`, `training_pipeline.py` |
| P3 | なし | `market_model.py`, `training_pipeline.py` (engine.py変更なし) |
| P4 | `pace_aptitude_features.py`, `course_features.py` | `feature_engine.py` (wiring追加), `horse_history_features.py`, `stage1_ability_model.py`, `training_pipeline.py` |

---

## PIT安全性の総合方針

全Phaseを通じて以下を厳守:

1. **累積統計は `shift(1)` または `cumsum().shift(1)`** — 当日結果を含まない
2. **過去データ参照は `searchsorted`** — O(log n) で時点前のデータのみ取得
3. **静的データ (系統コード等) はPIT安全** — 時間不変
4. **CV は `shuffle=False`** — 時系列順序を保持
5. **新特徴量は `leakage_validators.py` で検証** — POST_RACE 列の混入を自動検出
