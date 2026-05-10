---
spike: data-leak-phase-20-22
status: complete
created: 2026-05-10
verdict: NO direct feature leakage found. ROI 1075% caused by structural overfitting + selection bias.
---

# Data Leak Investigation: Phase 20-22

## Executive Summary

**Verdict: No direct feature leakage (POST_RACE_COLS properly excluded from prediction path)**

ROI 1075% + 勝率 99.6% の原因は、単一のリークバグではなく、**構造的過学習 + 選択バイアス** の複合効果。

3エージェント並行調査で特徴量エンジン(22ファイル)、バックテストエンジン、オッズデータタイミングを
全点検。直接的な POST_RACE_COLS の特徴量混入は確認されず。

---

## Severity Classification

### CRITICAL — Structural Overfitting (not direct leakage)

#### S1: CQR 過学習 → Q_90=0.0000 → EV_lower 過大推定

- **File:** `src/models/conformal_ev_model.py`, `src/pipelines/training_pipeline.py:853-896`
- **Root Cause:** CQR の特徴量に主モデル出力 (ev_win_calibrated, p_hit, e_return 等) を含めた
  「残差学習」設計 (39cbda3)。437列の特徴量で LightGBM quantile regression を学習。
- **Mechanism:**
  1. CQR は主モデルの予測値 (既に actual_ev_win と高相関) を特徴量として受け取る
  2. LightGBM が訓練データに過学習 → 分位点モデルが actual をほぼ完全にカバーする区間を出力
  3. キャリブレーションセットでも非適合度 ≒ 0 → Q_90 = 0.0000
  4. EV_lower = q_low - Q_90 = q_low (Q_90=0 のため補正なし)
  5. ベット選択が「主モデルが自信のある馬」に極端に偏る
- **Evidence:**
  - CQR キャリブレーション量子が全て 0.0000 (turf/dirt とも)
  - n_calib = 13,374 (turf), 14,435 (dirt) — サンプル不足ではない
  - 39cbda3 のコミットメッセージで「437列の生特徴量」と記載
- **Impact:** ベット選択が過度に保守的になり、272ベット/年に圧縮。選択されたベットは
  ほぼ確実に的中する「簡単な」レースに偏り、ROI が過大評価される。

#### S2: Ultra-Selective Betting = Implicit P-Hacking

- **File:** `src/backtest/engine.py`
- **Root Cause:** 複数のカスケードフィルター (OddsBandFilter, QualityScreener, RegimeDetector,
  SelectionGate, EV threshold) が数千レースを 272ベットに絞り込む。
- **Mechanism:** 272ベット = JRA 全レースの ~0.75%/日。この極端な選択性により:
  - テスト期間で偶然勝ったパターンのみが残る
  - engine.py の bet-count guard (1000/年) を大幅に下回るが警告のみで停止しない
  - COLLAPSED regime スキップが「負ける期間」を除外 → 生き残りバイアス
- **Evidence:** 272ベット中 271的中 = 99.63%。place bet の base rate ~18-35% から考えれば異常値。

---

### HIGH — Actual Look-Ahead Bias

#### H1: WideTwoStageModel uses Random Split (Not Temporal)

- **File:** `src/models/wide_two_stage_model.py:15-32`
- **Code:**
  ```python
  perm = np.random.RandomState(seed).permutation(n)
  split = int(n * (1 - valid_ratio))
  train_idx, valid_idx = perm[:split], perm[split:]
  ```
- **Issue:** 時系列データ (競馬結果) でランダム分割を使用。未来のデータが訓練に含まれ、
  過去のデータが検証に含まれる。Win/Place モデルは temporal split を使用するため、
  Wide モデルのみこの問題がある。
- **Impact:** Wide モデルの validation metrics が過大評価。ワイドベット選択の精度が
  バックテストで不当に高くなる。
- **Fix:** `_train_valid_split` を temporal split (first N% for train, rest for valid) に変更。

#### H2: CQR allows train_ratio >= 1.0 (train = calibrate on identical data)

- **File:** `src/models/conformal_ev_model.py:126-133`
- **Issue:** `train_ratio >= 1.0` の場合、同じデータで学習+キャリブレーション。
  非適合度が必ず 0 になる。現在の呼び出しでは train_ratio=0.8 (デフォルト) だが、
  将来の変更で 1.0 に設定されるリスクがある。
- **Current Status:** training_pipeline.py:887 で train_ratio 指定なし (デフォルト 0.8)。
  temporal split は正しいが、437列の特徴量で 200 boost rounds は過学習リスクが高い。

---

### MEDIUM — Design Fragilities

#### M1: Post-race columns survive in result_df (latent risk)

- **File:** `src/features/feature_engine.py:222-227`
- **Issue:** `kakuteijyuni`, `ninki`, `time`, `timediff` 等が `build_all()` の戻り値に残存。
  現在のモデルは明示的 FEATURE_COLS で除外しているが、将来のコード変更で漏れるリスク。
- **Fix:** `build_all()` の最後で `result_df.drop(columns=[c for c in POST_RACE_COLS if c in result_df.columns])` を追加。

#### M2: CQR feature blacklist approach (not whitelist)

- **File:** `src/models/conformal_ev_model.py:118-123`
- **Issue:** 特徴量を「除外リストにない列」で自動抽出。POST_RACE_COLS にないが
  レース後情報を含む列が追加された場合、自動的に特徴量に混入する。
- **Fix:** CQR も明示的 FEATURE_COLS (whitelist) を使用すべき。

#### M3: EV correction train/test odds inconsistency

- **File:** `src/models/ev_correction_model.py:370`
- **Issue:** 学習時は `confirmed_odds` (確定オッズ) で ev_odds_band_scales を計算。
  バックテスト推論時は `confirmed_odds` が POST_RACE_COLS で drop されるため `"odds"`
  (= tanodds, 発走前オッズ) にフォールバック。学習と推論で異なるオッズを使用。
- **Impact:** odds-band スケーリングが学習時と推論時で不一致。

#### M4: fukuoddslow as both feature and target

- **File:** `src/models/ev_correction_model.py:405`
- **Issue:** `fukuoddslow` が PlaceEVCorrectionModel の特徴量であり、同時に Place Return Model
  のターゲット (`y = hit_df["fukuoddslow"]`) でもある。近恒等写像のリスク。

#### M5: JODDS DataKubun=3/4 not explicitly filtered

- **File:** `src/features/odds_dynamics_features.py`
- **Issue:** JODDS 時系列データに確定オッズ (DataKubun=3/4) が含まれるが、
  明示的な DataKubun フィルタがない。時間オフセットロジックで暗黙的に除外されるが、
  脆弱な設計。

#### M6: popularity_rank fallback to ninki (post-race)

- **File:** `src/features/feature_engine.py:421-463`
- **Issue:** `popularity_rank` の算出で `tanodds` → `tanninki` → `ninki` (確定人気順) の
  フォールバックチェーン。`ninki` は POST_RACE データ。tanodds が利用できない場合に
  確定人気が特徴量に混入する。

---

### LOW — Minor Concerns

| # | Finding | File | Detail |
|---|---------|------|--------|
| L1 | odds_rank depends on call order | `intra_race_features.py:26` | `compute_intra_race_features` は `build_all()` の odds→tanodds 置換後に呼ばれる必要がある |
| L2 | build_features() injects kakuteijyuni | `feature_engine.py:346-358` | 推論パスで `kakuteijyuni`, `ninki` が DataFrame に含まれる (値は 0 のはずだが保護なし) |
| L3 | OddsBandFilter calibrated on training period | `engine.py:406-472` | 訓練期間の内部BTでフィルタを最適化 → 間接的な情報漏れの可能性 |

---

## Root Cause Analysis: Why ROI 1075%?

直接の特徴量リークは存在しない。ROI の異常値は以下の**3層構造**で説明できる:

```
Layer 1: CQR 過学習 (Q_90=0)
  → EV_lower が過大推定
  → ベット選択が極端に保守的に

Layer 2: カスケードフィルター
  → OddsBandFilter + QualityScreener + RegimeDetector + SelectionGate
  → 4000+候補 → 272ベットに圧縮
  → 偶然勝つパターンのみ残存 (implicit p-hacking)

Layer 3: Wide モデルの look-ahead bias
  → ランダム分割による validation metrics の過大評価
  → ワイドベットの精度が不当に高い
```

### 期待される現実的ROI

- 272ベット/年 (全レースの ~0.75%) の超選択的戦略
- Place bet の base rate: ~18-35%/馬
- 健全な ROI は 5-20% 程度が現実的 (市場の takeout 20-30% を考慮)
- 1075% は統計的変動 + 過学習の産物

---

## Recommended Fixes (Priority Order)

1. **[HIGH] WideTwoStageModel temporal split** — `_train_valid_split` を time-sorted に変更
2. **[HIGH] CQR 特徴量削減** — 主モデル出力を除外するか、明示的 whitelist に変更
3. **[MEDIUM] bet-count guard 強化** — 272 < 1000 でバックテスト失敗にする
4. **[MEDIUM] CQR train_ratio 検証** — 0.8 でも 437列では過学習。特徴量選択が必要
5. **[LOW] POST_RACE_COLS の明示的 drop** — `build_all()` の最後で一括削除

---

## Files Investigated

### Feature Engineering (22 files)
- `src/features/feature_engine.py` — メインオーケストレーター
- `src/features/horse_history_features.py` — 過去成績 (searchsorted PIT)
- `src/features/intra_race_features.py` — レース内特徴量
- `src/features/odds_dynamics_features.py` — オッズ動態
- `src/features/odds_deviation_features.py` — オッズ偏差
- `src/features/market_bias_features.py` — 市場バイアス
- `src/features/info_asymmetry_features.py` — 情報非対称性
- `src/features/interaction_features.py` — 相互作用特徴量
- `src/features/jockey_context_features.py` — 騎手コンテキスト
- `src/features/jockey_trainer_combo.py` — 騎手調教師 combo
- `src/features/pace_aptitude_features.py` — ペース適性
- `src/features/course_features.py` — コース特徴量
- `src/features/leakage_validators.py` — リーク検証モジュール
- + others (bloodline, sire, form_cycle, high_odds, trainer_context, race_difficulty, etc.)

### Models
- `src/models/conformal_ev_model.py` — CQR モデル
- `src/models/ev_correction_model.py` — EV 補正モデル
- `src/models/wide_two_stage_model.py` — ワイドモデル (ランダム分割)
- `src/models/two_stage_return_model.py` — Win/Place 2段階モデル

### Pipeline & Backtest
- `src/pipelines/training_pipeline.py` — 学習パイプライン
- `src/backtest/engine.py` — バックテストエンジン

### Data Schema
- `docs/everydb2/04-UMA_RACE.md` — UMA_RACE テーブル (Odds = 確定オッズ)
- `docs/everydb2/14-ODDS_TANPUKUWAKU_HEAD.md` — オッズスナップショットヘッダ
- `docs/everydb2/15-ODDS_TANPUKU.md` — 単勝オッズ
- `docs/everydb2/46-JODDS_TANPUKUWAKU_HEAD.md` — JRA オッズ時系列ヘッダ
- `docs/everydb2/47-JODDS_TANPUKU.md` — JRA オッズ時系列
