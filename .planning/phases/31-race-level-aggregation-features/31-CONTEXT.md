# Phase 31: Race-Level Aggregation Features - Context

**Gathered:** 2026-05-18
**Status:** Ready for planning

<domain>
## Phase Boundary

レース全体の市場構造を表す6特徴量(rl_log_odds_entropy, rl_odds_dispersion, rl_top3_odds_gap, rl_top1_odds, rl_favorite_rank_gap, rl_n_horses)が追加され、既存の未登録特徴量2つ(implied_prob_hhi, odds_skewness)がFEATURE_COLSに昇格し、train/inference両パスで同じ特徴量が計算される。

**In scope:**
- RLF-01~06 の6 race-level特徴量の実装 (新規 `src/features/race_level_features.py`)
- RLF-07 build_all()とbuild_features()の両方でのrace-level特徴量計算 (train/inference parity)
- EFP-01 implied_prob_hhi を全モデルのFEATURE_COLSに昇格
- EFP-02 odds_skewness を全モデルのFEATURE_COLSに昇格
- EFP-03 FEATURE_COLS manifest SHA256更新
- POST_RACE情報漏洩テストの通過確認

**Out of scope:**
- 市場クロス整合性特徴量 (Phase 32)
- バックテスト実行 (Phase 34)
- IC評価・ベースライン比較 (Phase 34)
- Gain per Depth診断 (Phase 33)
- ETL拡張 (Phase 29 — complete)
- モデル構造の変更

</domain>

<decisions>
## Implementation Decisions

### Race-Level特徴量モジュール
- **D-01:** 新規 `src/features/race_level_features.py` を作成。既存 `market_bias_features.py` とは独立。market_biasは市场バイアス(FLB等)に特化、race_levelはレース構造全般
- **D-02:** 6特徴量の定義:
  - `rl_log_odds_entropy`: インプライド確率のシャノンエントロピー `-sum(p * log(p))` where `p = 1/tanodds` normalized per race
  - `rl_odds_dispersion`: レース内tanoddsの標準偏差
  - `rl_top3_odds_gap`: 1番人気と3番人気のtanodds差 (混戦度指標)
  - `rl_top1_odds`: 1番人気のtanodds値をレース内全馬にブロードキャスト (鉄板度)
  - `rl_favorite_rank_gap`: 1番人気と2番人気の対数オッズ差 `log(odds_fav2 / odds_fav1)` (支配度)
  - `rl_n_horses`: 出走頭数 (`field_size`または`umaban`のユニーク数)
- **D-03:** 全特徴量は `tanodds` (pre-race snapshot) のみを使用。`POST_RACE_COLS` に含まれる列は一切使用しない

### build_features() パリティ
- **D-04:** build_features() パリティの実現方法はClaudeの判断に委ねる。推奨: 共通関数を `race_level_features.py` に実装し、build_all()とbuild_features()の両方から呼び出す
- **D-05:** 現在のbuild_features()は `_map_basic_features()` のみ呼び出し、サブモジュールをスキップしている。race-level特徴量のみパリティ対応し、他サブモジュールのパリティは今回のスコープ外

### 特徴量昇格
- **D-06:** implied_prob_hhi と odds_skewness を **全12モデル** のFEATURE_COLSに追加。現在未登録のモデル(AbilityModel, MarketModel, RegimeDetector等)にも追加
- **D-07:** 両特徴量は `market_bias_features.py:compute_flb_slope()` で既に計算済み。追加作業はFEATURE_COLSへの列名追加のみ

### rl_favorite_rank_gap の定義
- **D-08:** `rl_favorite_rank_gap = log(odds_fav2 / odds_fav1)` (対数オッズ差) を採用。ベストプラクティスに基づく選択:
  - 対称性: 逆方向の差が同じ絶対値
  - LightGBMのbinary objectiveと同じlogit空間で動作
  - 高オッズ帯でも低オッズ帯でも同等の感度
  - Kelly基準エッジ計算と直接対応

### Claude's Discretion
- build_features()へのパリティ統合の具体的な実装方法 (共通関数抽出が推奨)
- race_level_features.py の内部関数構成
- 各特徴量のエッジケース処理 (少頭数レース、オッズ欠損等)
- テストケースの具体的な設計

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Feature Engine (中核)
- `src/features/feature_engine.py` — FeatureEngine.build_all() (lines 198-383) と build_features() (lines 385-455)。build_all()のサブモジュール呼び出しパターン(line 319-343)、SAFE-01 POST_RACE stripping (lines 362-371)
- `src/features/market_bias_features.py` — compute_flb_slope() (lines 57-87)。implied_prob_hhi / odds_skewnessの計算箇所。groupby("race_id")パターンの参照実装
- `src/features/intra_race_features.py` — compute_intra_race_features()。既存のrace-level aggregationパターン (bataijyu平均、popularity ranking)

### FEATURE_COLS と Manifest
- `src/models/stage1_ability_model.py` — AbilityModel.FEATURE_COLS (95列)
- `src/models/two_stage_return_model.py` — WinTwoStageModel / PlaceTwoStageModel FEATURE_COLS
- `src/models/ev_correction_model.py` — EVCorrectionModel.FEATURE_COLS (151行目) / PlaceEVCorrectionModel.FEATURE_COLS (405行目)
- `src/models/conformal_ev_model.py` — ConformalEVModel.FEATURE_COLS (81行目)
- `src/models/market_model.py` — MarketModel.FEATURE_COLS (7列)
- `src/models/regime_detector.py` — RegimeDetector.FEATURE_COLS (49行目, 8列)
- `scripts/freeze_feature_manifest.py` — manifest生成スクリプト。12モデルのFEATURE_COLSをSHA256で凍結

### POST_RACE安全性
- `src/domain/types.py` (lines 38-55) — POST_RACE_COLS定義
- `tests/test_post_race_leakage.py` — 3層テストアーキテクチャ。Layer 1 (build_all出力)、Layer 2 (FEATURE_COLS whitelist)、Layer 3 (EV correction odds検証)

### 要件定義
- `.planning/REQUIREMENTS.md` §Race-Level Aggregation Features — RLF-01~07, §Existing Feature Promotion — EFP-01~03

### パイプライン統合
- `src/pipelines/training_pipeline.py` — _train_submodel() でのFeatureEngine呼び出し
- `src/backtest/race_predictor.py` — RacePredictor.predict() でのbuild_features()使用

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/features/market_bias_features.py::compute_flb_slope()`: groupby("race_id")でのレース単位集計パターン。implied_prob_hhi / odds_skewness はここで計算済み
- `src/features/intra_race_features.py::compute_intra_race_features()`: df.groupby("race_id")でのbataijyu平均、odds rankingパターン
- `src/features/odds_dynamics_features.py`: レースレベルrolling統計のパターン (odds_volatility, overround, entropy)
- `src/features/interaction_features.py` (lines 157-207): race_mean_odds / race_std_odds のgroupby transformパターン
- `feature_engine.py::_compute_popularity_rank_from_tanodds()` (lines 133-154): tanoddsからの人気順位計算 (POST_RACE安全)

### Established Patterns
- サブモジュールパターン: 独立した`compute_*()`関数 → build_all()内でTimingContext付き呼び出し → pd.concat で結果マージ
- groupby("race_id")パターン: `.transform("mean")` でbroadcast、`.agg()` で集約
- FEATURE_COLSパターン: クラス属性として定義、`df[self.FEATURE_COLS]` で列選択
- Feature cacheパターン: compute_code_hash() が src/features/ 内の全.pyをハッシュ → コード変更でキャッシュ無効化

### Integration Points
- `src/features/feature_engine.py::build_all()` line ~340: 新しい `compute_race_level_features()` の呼び出し追加先 (compute_difficulty_scoreの後)
- `src/features/feature_engine.py::build_features()` line ~455: race-level特徴量のパリティ追加先
- 全12モデルクラスの `FEATURE_COLS` クラス属性: implied_prob_hhi / odds_skewness の追加先
- `scripts/freeze_feature_manifest.py`: manifest再生成の実行

</code_context>

<specifics>
## Specific Ideas

- rl_favorite_rank_gap に対数オッズ差を採用することで、LightGBMのshallow depth (3-5) で効果的に機能する可能性が高い (Phase 33 Gain per Depthでの検証予定)
- build_features()パリティの実現には、race_level_features.py内でrace_idのgroupbyに依存しない単一レース対応版の関数も用意する必要がある可能性がある (build_features()は1レースのみ処理)
- implied_prob_hhi / odds_skewness は build_all() で既に計算済みのため、FEATURE_COLSへの追加のみでモデルに利用可能。追加計算不要

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 31-Race-Level Aggregation Features*
*Context gathered: 2026-05-18*
