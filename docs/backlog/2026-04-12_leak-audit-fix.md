# データリーク修正 (Leak Audit Fix)

**日付:** 2026-04-12
**対象ブランチ:** main (2e3e3e0 → a8e3219)
**設計書:** `docs/superpowers/specs/2026-04-11-leak-audit-fix-design.md`
**計画書:** `docs/superpowers/plans/2026-04-11-leak-audit-fix-plan.md`

---

## 概要

バックテストとペーパートレードの乖離原因となっていた全データリーク (POST_RACE情報の混入) を修正。TDD で既存動作を保護しながら、11タスク・13コミットで対応。

## 修正前後のバックテスト比較

| 指標 | 修正前 (リークあり) | 修正後 (リークなし) | 変化 |
|------|-------------------|-------------------|------|
| ROI | 214.0% | **164.3%** | -49.7pt |
| ベット数 | ~2,400 | **2,747** | +347 |
| 純利益 | +¥247,550 | **+¥176,500** | -¥71,050 |
| 最大DD | 0.6% | **1.7%** | +1.1pt |
| 投資額 | ~¥240,000 | **¥274,700** | — |

※ 2021-2024学習 / 2025テスト / flat ¥100 / --ensemble

---

## 修正内容一覧

### C1: JockeyTrainerComboFeatures searchsorted修正
- **ファイル:** `src/features/jockey_trainer_combo.py`
- **問題:** `entry_df["race_date"].max()` で全体フィルタ → 早いレースが遅いレースの結果を見える
- **修正:** 行ごと `searchsorted(target_date, side="left")` に変更
- **コミット:** `beb11ca`

### C2: running_style マッピング削除
- **ファイル:** `feature_engine.py`, `wide_pair_builder.py`, `wide_two_stage_model.py`, `jvlink_fetcher.py`, `domain/models.py`
- **問題:** `kyakusitukubun` (脚質) はレース後判定の POST_RACE フィールド
- **修正:** `running_style` → `kyakusitukubun_cd` (過去走データ) に差し替え
- **コミット:** `3a1c523`

### H2: compute_flb_slope をオッズ歪度に変更
- **ファイル:** `src/features/market_bias_features.py`
- **問題:** `kakuteijyuni` (確定着順) を使った FLB 回帰
- **修正:** `odds_skewness` (オッズ分布歪度) + `implied_prob_hhi` (HHI) に変更
- **コミット:** `85ae0cb`

### H1: compute_roi_ema をオッズのみ指標に変更
- **ファイル:** `src/features/odds_dynamics_features.py`
- **問題:** `kakuteijyuni` で ROI 計算 → 未来情報リーク
- **修正:** `favorite_implied_prob_ema`, `overround_ema`, `entropy_ema` (全て tanodds 由来)
- **コミット:** `ab670a6`

### M2: ninki フォールバック修正
- **ファイル:** `src/features/feature_engine.py`
- **問題:** `tanninki` が 0 の時に `ninki` (確定人気) でフォールバック
- **修正:** フォールバック削除、警告ログ追加。`tanninki` 列自体がない場合のみ `ninki` 使用
- **コミット:** `fb20fcb`

### C3: favorite_win_rate expanding再計算
- **ファイル:** `src/pipelines/training_pipeline.py` (`_build_race_level_features`)
- **問題:** `favorite_win_rate` が当該レースの `kakuteijyuni` を含む集約値
- **修正:** `shift(1).expanding(min_periods=10).mean()` で過去レースのみ使用。フォールバック 0.3
- **コミット:** `bcd9e70` ~ `2f9d2b0` (含む Section3 の _build_regime_stats)

### Section3: RegimeDetector統合修正
- **ファイル:** `src/models/regime_detector.py`, `src/pipelines/training_pipeline.py`, `src/backtest/engine.py`
- **問題:** FEATURE_COLS に `favorite_win_rate`, `flb_slope`, `roi_ema` 等の POST_RACE 指標
- **修正:**
  - FEATURE_COLS を 11列→8列に削減 (PRE_RACE 指標のみ)
  - 新列: `overround_rolling`, `entropy_rolling`, `favorite_implied_prob_rolling`, `odds_skewness_rolling`
  - `train()` 教師ラベルを `favorite_implied_prob_rolling` × `overround_rolling` に変更
  - `_build_regime_stats` を rolling 統計ベースに書き直し
  - `engine.py` に per-race 統計蓄積 + `detect()` 呼び出しを統合
- **コミット:** `bcd9e70`, `2f9d2b0`, `8603861`

### M3: POST_RACE列のpredict除外
- **ファイル:** `src/backtest/engine.py`
- **問題:** `predict()` に `kakuteijyuni`, `confirmed_odds` が渡される
- **修正:** `predict()` 呼び出し前にこれらの列を除外。精算用には保持
- **コミット:** `51cb5f3`

### M1: オッズフォールバック時スキップ
- **ファイル:** `src/backtest/engine.py`
- **問題:** 時系列オッズなし時に確定オッズにフォールバック (ルックアヘッド)
- **修正:** `odds_ts_df` が空の場合は全レーススキップ + 警告ログ
- **コミット:** `28fe479`

### P1: ペーパートレード確定オッズ修正
- **ファイル:** `src/paper_trading/predictor.py`, `scripts/run_paper_trading.py`
- **問題:** ペーパートレードで確定オッズを使用
- **修正:** `extract_pre_post_odds` で発走前オッズを優先、フォールバックは許容
- **コミット:** `77a3acc`

---

## テスト結果

- **764 tests passed**, 2 skipped, 1 pre-existing failure (`test_mlflow_logging.py`)
- 新規テスト: ~30件追加 (リーク防止・統合テスト含む)
- 回帰テストなし (既存テスト全て通過)

## 学んだこと

- `popularity_rank` で `tanninki=0` の約37.8%の馬が NaN になる (警告ログ出力済み)
- LightGBM は NaN を自然に処理するため、無理にフォールバックするより NaN のまま渡す方が安全
- バックテストでは「確定オッズで判定 + 確定オッズで精算」は完全なルックアヘッド。「発走前オッズで判定 + 確定オッズで精算」が正しい分離
