# Phase 27: Feature Interactions - Context

**Gathered:** 2026-05-15
**Status:** Ready for planning

<domain>
## Phase Boundary

最終ベース特徴量セット上にドメイン知識に基づく交互作用項を生成し、高カーディナリティカテゴリ変数をターゲットエンコーディングで処理する。Phase 26で追加された新特徴量（血統、相対比較、mining、record）を含む最終特徴量セットが前提。

**In scope:**
- INTER-01: オッズ・能力値等のレース内相対ランク特徴量の生成。Phase 26のrelative_features.pyの拡張（残り4個のStage1/2追加 + オッズ相対・能力値相対の新規生成）
- INTER-02: 10-15個のドメイン知識に基づく条件付き交互作用項の生成。既存3個 + 新規追加で合計10-15個に
- INTER-03: 高カーディナリティカテゴリ変数（血統コード、騎手コード、調教師コード）のターゲットエンコーディング。OOFリーク完全防止
- FEATURE_COLS更新（Stage1AbilityModel + WinTwoStageModel + PlaceTwoStageModel）
- 既存テスト全通過確認 + POST_RACE漏洩テスト通過確認

**Out of scope:**
- 最終ROI検証・特徴量凍結（Phase 28）
- モデル再学習・ハイパーパラメータ調整
- 複勝/ワイドモデルの変更
- 新しいベース特徴量の追加（Phase 26完了済み）
- レースコンテキスト/ペース投影特徴量の追加（interaction_features.pyに既存）

</domain>

<decisions>
## Implementation Decisions

### 相対特徴量の拡張 (INTER-01)
- **D-01:** Phase 26の残り4個の相対特徴量（`rel_haron_vs_mean`, `rel_blood_quality_rank`, `rel_sire_quality_rank`, `rel_weight_zscore`）をStage1AbilityModel + WinTwoStageModelのFEATURE_COLSに追加する
- **D-02:** オッズ相対特徴量（人気順位の相対位置、複勝オッズ相対zscore等）+ 能力値相対特徴量（`p_ability_win`の相対ランク、`odds_to_ability_ratio`の相対偏差値等）を新規生成してWinTwoStageModelに追加する。ベストプラクティスを追求
- **D-03:** Stage1AbilityModelの既存`race_rank`系5個（intra_race_features.py）と新規relative_features.pyの特徴量は両方維持する。LightGBMが不要な方を自動的にgain=0にするため安全
- **D-04:** 新しい相対特徴量は`relative_features.py`の`_BASE_FEATURES`リストに追加して拡張する。新規モジュールは作成しない

### 交互作用項の設計 (INTER-02)
- **D-05:** 既存3個の扱い（INTER-02の10-15個にカウントするか）+ 新規追加数はClaudeの判断に委ねる。合計10-15個の範囲に収める
- **D-06:** 交互作用項の表現方法は最新のベストプラクティスを追求する。カテゴリ積（文字列結合→category型）と数値積の混合アプローチが推奨。特徴量の性質に応じて適切な表現を選択
- **D-07:** 実装場所はClaudeの判断に委ねる

### ターゲットエンコーディング (INTER-03)
- **D-08:** ターゲットエンコーディングの対象変数は血統系統コード（`blood_keito_cd`）+ 騎手コード + 調教師コード。既存の騎手/調教師コンテキスト特徴量（`jockey_wr_overall`等）とは異なる情報（ターゲットとの直接関係）を提供する
- **D-09:** OOFリーク防止は最適な手法を選択する。リークは絶対に許容しない。時系列データ（race_date順序）を考慮した手法が必須。TimeSeriesSplitベースまたはexpanding windowベースが最も安全
- **D-10:** ターゲットエンコーディング列を追加するモデルはClaudeの判断に委ねる。Phase 25のD-02決定（騎手/調教師コンテキストはStage2のみ）との整合性を考慮
- **D-11:** ターゲットエンコーディングの実装場所はClaudeの判断に委ねる。TEは学習データのfold分割に依存する前処理ステップであり、他の特徴量計算とは性質が異なる点に注意

### Claude's Discretion
- INTER-02: 既存3個のカウント扱い + 新規追加数 + 具体的な交互作用項の選定
- INTER-02: 実装場所（interaction_features.py拡張 vs 新規モジュール）
- INTER-03: TEの追加先モデル（Stage1 + Stage2 + Place のどれに追加）
- INTER-03: TEの実装場所（新規target_encoding.py vs 既存モジュール組み込み）
- INTER-03: 平滑化パラメータ、最小サンプル数閾値の設定
- 各特徴量のFEATURE_COLSへの具体的な挿入位置
- テストの追加・更新内容
- POST_RACE漏洩テストの通過確認方法

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 既存特徴量モジュール（拡張・統合対象）
- `src/features/relative_features.py` — Phase 26の相対特徴量。7特徴量。_BASE_FEATURESに新規追加する（D-04）
- `src/features/interaction_features.py` — 既存3交互作用 + レースコンテキスト + ペース投影。拡張または参照元
- `src/features/intra_race_features.py` — 既存2特徴量(weight_diff_from_mean, odds_rank) + Stage1のrace_rank系5個。D-03で両方維持
- `src/features/feature_engine.py:build_all()` — 特徴量統合ポイント。compute_relative_features()等の呼び出し順序

### FEATURE_COLS定義（変更対象）
- `src/models/stage1_ability_model.py:28` — Stage1AbilityModel.FEATURE_COLS (~89特徴量)。相対特徴量・TE追加先候補
- `src/models/two_stage_return_model.py:48` — WinTwoStageModel.FEATURE_COLS (~67特徴量)。相対特徴量・交互作用・TE追加先
- `src/models/two_stage_return_model.py:289` — PlaceTwoStageModel.HIT_FEATURE_COLS
- `src/models/two_stage_return_model.py:345` — PlaceTwoStageModel.RETURN_FEATURE_COLS

### ターゲットエンコーディング設計参照
- `src/features/jockey_context_features.py` — 騎手コンテキスト特徴量。Beta(1,10)平滑化パターンの参考
- `src/features/bloodline_features.py` — blood_keito_cdを既に使用。TE対象の既存取り扱い
- `src/features/horse_career_stats.py` — PIT-safe累積キャリア統計。expanding().shift(1)パターンの参考
- `src/features/info_asymmetry_features.py` — expanding().shift(1) PIT-safeパターンの参考
- `src/pipelines/training_pipeline.py` — TrainingPipelineV5。OOF fold分割ロジック（TEのfold分割と整合性必須）

### 安全性・監査
- `src/domain/types.py:38-55` — POST_RACE_COLS定義（16列）。漏洩検出の基準
- `tests/test_post_race_leakage.py` — 3層漏洩検出CIテスト。新特徴量追加後も通過必須
- `scripts/analyze_feature_importance.py` — feature importance監査CLI。効果測定に使用可能

### 前フェーズのCONTEXT（決定の連続性）
- `.planning/phases/26-everydb2-new-features/26-CONTEXT.md` — Phase 26決定（血統・相対・mining・record特徴量）
- `.planning/phases/25-quick-win-wire-existing/25-CONTEXT.md` — Phase 25決定（12特徴量配線パターン、Stage2のみ追加）
- `.planning/phases/24-feature-audit-pruning/24-CONTEXT.md` — Phase 24決定（Tier分類、監査パターン）

### 要件定義
- `.planning/REQUIREMENTS.md` — INTER-01, INTER-02, INTER-03の要件定義
- `.planning/ROADMAP.md` — Phase 27 Success Criteria
- `.planning/research/FEATURES.md` — 特徴量ランドスケープ研究。交互作用・相対比較の分析

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **relative_features.py**: groupby("race_id")変換の完成されたパターン。zscore/vs_mean/rank_asc/rank_descの4種変換。_BASE_FEATURESリストに追加するだけで新しい相対特徴量を生成可能
- **interaction_features.py**: カテゴリ積(astype(str) + "_"結合 → category型)と数値積(where() NaN安全)の両方の実装パターンが既存。ドメイン知識交互作用の追加は既存パターンの踏襲
- **bloodline_features.py**: blood_keito_cdの既存取り扱い。TE化の対象となる列の現状確認に必要
- **jockey_context_features.py**: Beta(1,10)平滑化パターン。TEの平滑化戦略の参考
- **horse_career_stats.py / info_asymmetry_features.py**: expanding().shift(1) PIT-safeパターン。時系列TEの参考
- **Phase 23 監査スクリプト**: `scripts/analyze_feature_importance.py`。新特徴量追加後の効果測定に使用可能

### Established Patterns
- **FEATURE_COLSリスト**: モデルクラスにFEATURE_COLS list[str]を定義。Phase 24-26で多数の追加実績あり
- **新規モジュール → build_all()統合**: feature_engine.pyのbuild_all()に新モジュールの呼び出しを追加するパターン
- **mockベーステスト**: 全テストがDB不要。FEATURE_COLS変更に伴うテスト更新はmockのcolumn list更新のみ
- **POST_RACE漏洩検出**: Phase 23の3層CIテスト。新特徴量追加時も自動的に検証される
- **コードハッシュキャッシュ無効化**: Phase 24で導入。特徴量モジュール変更時に自動キャッシュクリア

### Integration Points
- **relative_features.py:_BASE_FEATURES**: 新しい相対特徴量の追加（オッズ相対・能力値相対）
- **relative_features.py:compute_relative_features()**: build_all()から呼び出し済み。オッズ列がDataFrameに存在するかを確認する必要あり
- **interaction_features.py:compute_interaction_features()**: 新しい交互作用項の追加または新規モジュール
- **新規target_encoding.py**: TE計算 → training_pipelineとbacktest/engineの両方に組み込み。OOF fold分割との整合性必須
- **各モデルFEATURE_COLS**: 新特徴量名の追加。Phase 25-26パターンに従う

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・堅牢性を優先
- INTER-01: Phase 26のrelative_features.pyを拡張する方針が確定。オッズ相対特徴量はfukuoddslow等のオッズ列のDataFrame内存在を確認する必要がある
- INTER-02: カテゴリ積と数値積の混合が推奨。LightGBMはcategory型を native supportするためカテゴリ積も有効
- INTER-03: リークは絶対に許容しない。時系列データ(race_date)を考慮した手法が必須
- 騎手/調教師コンテキスト特徴量(jockey_wr等)とTEは異なる情報を提供する。コンテキスト特徴量は過去成績の集約、TEはターゲットとの直接関係
- Phase 25のD-02決定（騎手/調教師コンテキストはStage2のみ追加）との整合性を考慮する必要がある

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 27-Feature Interactions*
*Context gathered: 2026-05-15*
