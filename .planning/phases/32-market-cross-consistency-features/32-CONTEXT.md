# Phase 32: Market Cross-Consistency Features - Context

**Gathered:** 2026-05-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Harville理論オッズによる馬券種クロス整合性特徴量が追加され、単勝×ワイド×三連複の市場構造矛盾を捉えられるようになる。

**In scope:**
- MCF-01 Harville公式による理論ワイドオッズ計算機能の実装
- MCF-02 rl_favorite_in_wide_top1 — 1番人気がワイドTOP1組合せに含まれるか (0/1)
- MCF-03 rl_trio_overlap — 三連複1組合せが単勝上位3頭に含まれる馬数 (0-3)
- MCF-04 rl_market_consistency — 1番人気が三連複1組合せに含まれるか (0/1)
- MCF-05 rl_trio_odds_ratio — 実三連複1オッズ / Harville理論三連複オッズ
- MCF-06 rl_wide_harville_ratio — 実ワイドTOP1オッズ / Harville理論ワイドオッズ
- MCF-07 ワイドオッズmergeをbuild_all()に統合 (training/backtest重複排除)
- DataRepositoryにload_wide_odds()メソッド追加
- POST_RACE情報漏洩テストの通過確認
- 全12モデルFEATURE_COLSに5特徴量追加 + manifest更新

**Out of scope:**
- 三連単オッズを使用した特徴量 (Phase 29でETL済みだがMCF要件に含まれない)
- 上位複数組合せの乖離評価 (ninki=1のみがスコープ)
- バックテスト実行 (Phase 34)
- IC評価・ベースライン比較 (Phase 34)
- Gain per Depth診断 (Phase 33)
- モデル構造の変更
- 枠連オッズベース特徴量 (Future: MCF-08)

</domain>

<decisions>
## Implementation Decisions

### ワイドオッズ単一値導出
- **D-01:** ワイドオッズのレンジ(oddslow/oddshigh)から単一値は **中間値 `(oddslow + oddshigh) / 2`** を使用。最も偏りのない導出方法
- **D-02:** rl_wide_harville_ratio の計算対象は **ninki=1の組合せのみ**。Harville理論は高確率(1番人気)の組合せで最も精度が高く、低人気では独立性仮定の誤差が拡大するため
- **D-03:** rl_trio_odds_ratio も同様に ninki=1の三連複組合せのみを使用 (MCF-05要件に忠実)

### データマージ設計
- **D-04:** ワイド・三連複オッズのロード・マージ方法はClaudeの判断に委ねる。以下を満たすこと:
  - MCF-07: training/backtestでの重複コード排除
  - アーキテクチャの一貫性 (既存パターンとの整合)
  - build_all() と build_features() のパリティ
- **D-05:** データアクセスは **DataRepository (Phase 29)** を使用。既存の `load_trio_odds()`, `load_exacta_odds()` に加えて `load_wide_odds()` を追加し、全オッズアクセスをDataRepositoryに統一

### 欠損データ戦略
- **D-06:** ワイド・三連複オッズが欠損(2015-2017年等)の場合、特徴量は **NaNのまま** とする。LightGBMはNaNをネイティブに処理可能
- **D-07:** 欠損データの監視・対応はClaudeの判断に委ねる。ログ出力等でカバレッジを可視化することを推奨

### Harville公式実装
- **D-08:** Harville公式の実装は標準的な定式化に従う:
  - ワイド(馬連): `P(i,j) = P(i) × P(j) / (1 - P(i))` (順序なし組合せ)
  - 三連複: `P(i,j,k) = P(i) × P(j)/(1-P(i)) × P(k)/(1-P(i)-P(j))` (順序なし組合せ)
  - `P(i) = (1/tanodds_i) / sum(1/tanodds_j)` (インプライド勝率)
- **D-09:** 理論オッズ = 1 / 理論確率。実オッズ/理論オッズの比率が1.0から離れるほど市場非効率性が大きい

### モジュール構成
- **D-10:** 新規 `src/features/market_cross_features.py` を作成。Phase 31の `race_level_features.py` パターンを踏襲:
  - メインエントリポイント: `compute_market_cross_features(df, wide_df, trio_df)`
  - MCF_COLS リストをエクスポート (テスト用)
  - build_all()/build_features() パリティ対応

### Claude's Discretion
- build_all()へのワイド/三連複データマージ統合の具体的な実装方法
- market_cross_features.py の内部関数構成
- 各特徴量のエッジケース処理 (少頭数レース、オッズ欠損、kumi文字列パース等)
- テストケースの具体的な設計
- 欠損データの監視方法 (ログ出力等)
- Harville計算の数値安定性処理 (P(i)の合計≠1の場合等)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Feature Engine (中核)
- `src/features/feature_engine.py` — FeatureEngine.build_all() と build_features()。race-level features呼び出しの直後にmarket-cross featuresを追加 (lines ~345-348, ~460-462)
- `src/features/race_level_features.py` — Phase 31で作成。compute_race_level_features()のパターンを踏襲。RL_COLS エクスポート、_compute_for_single_race/_compute_for_multi_race パターン

### Data Access (Phase 29成果物)
- `src/db/repository.py` — DataRepository。load_trio_odds(), load_exacta_odds(), load_trifecta_odds() の既存メソッド。load_wide_odds() を追加
- `src/db/readers.py` — load_wide_odds() の既存実装 (line 251)。DataRepository追加時の参照

### ワイド/三連複オッズデータ構造
- `config/etl_tables.yaml` — 103テーブル定義。n_odds_wide, n_odds_sanren のスキーマ
- Wide odds: kumi="0102"(4桁), oddslow/oddshigh (odds100=/100), ninki
- Trio odds (sanren): kumi="010203"(6桁), odds (odds10=/10), ninki

### FEATURE_COLS と Manifest
- `src/models/stage1_ability_model.py` — AbilityModel.FEATURE_COLS (95列)
- `src/models/two_stage_return_model.py` — WinTwoStageModel / PlaceTwoStageModel FEATURE_COLS
- `src/models/ev_correction_model.py` — EVCorrectionModel / PlaceEVCorrectionModel FEATURE_COLS
- `src/models/conformal_ev_model.py` — ConformalEVModel.FEATURE_COLS
- `src/models/market_model.py` — MarketModel.FEATURE_COLS
- `src/models/regime_detector.py` — RegimeDetector.FEATURE_COLS
- `scripts/freeze_feature_manifest.py` — manifest生成スクリプト

### POST_RACE安全性
- `src/domain/types.py` (lines 38-55) — POST_RACE_COLS定義
- `tests/test_post_race_leakage.py` — 3層テストアーキテクチャ。Phase 32の新特徴量にも適用

### 要件定義
- `.planning/REQUIREMENTS.md` §Market Cross-Consistency Features — MCF-01~07

### Prior Phase Context
- `.planning/phases/31-race-level-aggregation-features/31-CONTEXT.md` — パターンの参照元
- `.planning/phases/29-etl-expansion/29-CONTEXT.md` — DataRepository設計とオッズデータ構造

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/features/race_level_features.py::compute_race_level_features()`: Phase 31で確立されたパターン。RL_COLSエクスポート、groupby("race_id")/単一レース分岐、POST_RACE安全なtanoddsのみ使用
- `src/features/market_bias_features.py::compute_flb_slope()`: groupby("race_id")でのレース単位集計パターン。implied_prob計算の参照
- `src/db/repository.py::DataRepository`: Phase 29で作成。load_trio_odds()/load_exacta_odds()/load_trifecta_odds()。load_wide_odds()を追加する拡張点
- `src/db/readers.py::load_wide_odds()`: 既存のワイドオッズリーダー。DataRepository追加時の実装参照
- `feature_engine.py::_compute_popularity_rank_from_tanodds()` (lines 133-154): tanoddsからの人気順位計算。Harville P(i)計算に利用可能

### Established Patterns
- サブモジュールパターン: 独立したcompute_*()関数 → build_all()内でTimingContext付き呼び出し → 結果マージ
- groupby("race_id")パターン: `.transform("mean")` でbroadcast、`.agg()` で集約
- FEATURE_COLSパターン: クラス属性として定義、全12モデルに一括追加
- POST_RACE安全性: AST source scan + FEATURE_COLS whitelist + build_all出力検証の3層
- DataRepositoryパターン: ParquetStore使用、date_filters + coerce_types

### Integration Points
- `src/features/feature_engine.py::build_all()` line ~348: compute_race_level_features()の直後にcompute_market_cross_features()を追加
- `src/features/feature_engine.py::build_features()` line ~462: race-level featuresの直後にmarket-cross featuresを追加
- `src/db/repository.py`: load_wide_odds()メソッドの追加先
- 全12モデルクラスのFEATURE_COLS: 5新特徴量の追加先
- `scripts/freeze_feature_manifest.py`: manifest再生成の実行

</code_context>

<specifics>
## Specific Ideas

- Harville理論オッズは「市場が完全に効率的ならどうなるか」のベースライン。実オッズ/理論オッズ > 1.0 = 市場が過小評価、< 1.0 = 過大評価
- rl_wide_harville_ratio と rl_trio_odds_ratio は異なる馬券種間の乖離を見るため、同じレースで異なるシグナルを提供する可能性がある
- Phase 33 (Gain per Depth) で、これらの市場クロス整合性特徴量がshallow depth (3-5) で機能することを検証予定
- ワイドオッズの中間値 (oddslow+oddshigh)/2 は10倍未満のオッズではレンジ幅が狭く、中間値で十分な精度

</specifics>

<deferred>
## Deferred Ideas

- 上位3組合せの個別乖離評価 (ninki=1,2,3別列出力) — スコープ外。将来フェーズで三連単ベース特徴量(MCF-08)とともに検討

</deferred>

---

*Phase: 32-Market Cross-Consistency Features*
*Context gathered: 2026-05-18*
