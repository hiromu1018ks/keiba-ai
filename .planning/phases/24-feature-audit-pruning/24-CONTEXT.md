# Phase 24: Feature Audit & Pruning - Context

**Gathered:** 2026-05-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 23で構築した監査スクリプト（permutation+gain重要度）をOOFデータで実行し、100+特徴量の有効性を定量化する。Tier 1（Gain=0 AND Perm≤0）のノイズ特徴量を各モデルのFEATURE_COLSから個別に除外し、除外前後のOOF品質比較とフルバックテストでROI改善を確認する。特徴量モジュール変更時の自動キャッシュクリア機構を導入する。

**In scope:**
- AUDIT-01: 全モデルのpermutation重要度をOOFデータで計算し、Tier 1（確実なノイズ）+ Tier 2（低重要度フラグ）の多段階レポートを出力
- AUDIT-02: Tier 1ノイズ特徴量をモデル別にFEATURE_COLSから除外し、OOF logloss/AUC比較で安全性確認後、フルバックテストでROI検証
- AUDIT-03: src/features/配下の全.pyファイルのコードハッシュをキャッシュキーに含め、モジュール変更時に自動キャッシュ無効化 + 古いキャッシュ自動削除
- 除外後のフルバックテスト実行（v1.5ベースライン ROI 84.4%との比較）
- ROI悪化時のロールバック + 原因分析レポート出力

**Out of scope:**
- 新しい特徴量の追加（Phase 25-26）
- 特徴量の交互作用・変換（Phase 27）
- 最終ROI検証・特徴量凍結（Phase 28）
- モデル再学習・ハイパーパラメータ調整
- 複勝/ワイドモデルの変更
- Phase 23の漏洩修正（完了済み）

</domain>

<decisions>
## Implementation Decisions

### プルーニング基準と設計
- **D-01:** 多段階プルーニングを採用。Tier 1（Gain=0 AND Permutation≤0）を自動除外、Tier 2（低重要度）をレポート出力してユーザー判断に委ねる。単一閾値の一律除外ではなく、確実性に応じた段階的アプローチ。
- **D-02:** 適用単位はモデル別個別プルーニング。各モデルのFEATURE_COLSを独立に最適化する。ある特徴量がWinでノイズでもPlaceで有効なら、WinのFEATURE_COLSからのみ除外する。
- **D-03:** Tier 1除外の安全性確認はOOF logloss/AUC比較で実施。フルバックテストの前に高速な品質チェックを行い、性能低下がある場合は除外を取り消す。

### ROI検証フロー
- **D-04:** 段階的ROI検証 — Step 1: OOF logloss/AUCで安全性確認 → Step 2: 通過したらフルバックテストでROI比較。v1.5ベースライン（ROI 84.4%）を流用。再実行なし。
- **D-05:** フルBTでROI悪化が確認された場合は即座にロールバック（元のFEATURE_COLSに復元）し、原因分析レポートを出力してPhase 24を完了する。別パターンの自動再試行は行わない。

### キャッシュ無効化（AUDIT-03）
- **D-06:** コードハッシュ方式を採用。キャッシュキー計算に`src/features/`配下の全.pyファイルの内容ハッシュを結合して含める。特徴量モジュールの変更を自動検出してキャッシュを無効化する。
- **D-07:** 無効化された古いキャッシュファイルは自動削除。ディスク容量の無駄な消費を防止する。

### Claude's Discretion
- Tier 1/Tier 2の具体的な閾値設定（Tier 2の「低重要度」の定義 — 例: 全特徴量の下位10%、パーセンタイル基準等）
- 各モデルでどの特徴量がTier 1に該当するかの特定（監査スクリプトの実行結果に基づく）
- 監査レポートの出力形式とファイル配置
- OOF logloss/AUC比較の具体的な実装（既存のvalidate_noise_removal拡張 vs 新規関数）
- キャッシュキー計算の具体的なハッシュ対象ファイルリスト
- ロールバック時の原因分析レポートのフォーマット
- プルーニング適用後のフルバックテスト実行コマンド構成

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 監査スクリプト（Phase 23で構築済み — AUDIT-01の中核）
- `scripts/analyze_feature_importance.py` — 全モデル対応のfeature importance監査CLI。--all-models, --format both, --n-repeats等の引数
- `src/features/win_feature_analysis.py` — compute_permutation_importance(), compute_all_model_importance(), identify_noise_features(), validate_noise_removal()
- `src/features/win_feature_analysis.py:103` — compute_permutation_importance() — sklearn permutation_importanceラッパー
- `src/features/win_feature_analysis.py:171` — compute_all_model_importance() — 全モデルgain+permutation計算、pivot DataFrame返却

### FEATURE_COLS定義（プルーニング対象）
- `src/models/stage1_ability_model.py:28` — Stage1AbilityModel.FEATURE_COLS (89特徴量)
- `src/models/two_stage_return_model.py:48` — WinTwoStageModel.FEATURE_COLS (41特徴量)
- `src/models/two_stage_return_model.py:289` — PlaceTwoStageModel.HIT_FEATURE_COLS (51特徴量)
- `src/models/two_stage_return_model.py:345` — PlaceTwoStageModel.RETURN_FEATURE_COLS (65特徴量)
- `src/models/ev_correction_model.py:151` — EVCorrectionModel.FEATURE_COLS (25特徴量)
- `src/models/ev_correction_model.py:405` — PlaceEVCorrectionModel.FEATURE_COLS (29特徴量)
- `src/models/conformal_ev_model.py:81` — ConformalEVModel.FEATURE_COLS (~120特徴量、最大)

### キャッシュ機構（AUDIT-03の変更対象）
- `src/features/feature_engine.py:37` — compute_cache_key() — 現在はソースParquetパス+日付範囲のみ。コードハッシュ追加対象
- `src/features/feature_engine.py:55` — is_cache_valid() — タイムスタンプベースのキャッシュ検証
- `src/features/feature_engine.py:137` — build_all() — キャッシュ読み書きのメインフロー

### バックテスト・ROI検証
- `scripts/run_backtest.py` — バックテストCLI。--calibration-bt, --report, --ensemble等
- `src/backtest/engine.py` — BacktestEngine。run()でフルBT実行
- `src/backtest/validation_report.py` — generate_validation_report() ROI PASS/FAIL判定

### 前フェーズのCONTEXT（決定の連続性）
- `.planning/phases/23-safety-gate/23-CONTEXT.md` — Phase 23決定（漏洩修正、監査スクリプト設計）
- `.planning/phases/22-integrated-validation/22-CONTEXT.md` — Phase 22決定（バックテスト検証、v1.4ベースライン比較）

### 要件定義
- `.planning/REQUIREMENTS.md` — AUDIT-01, AUDIT-02, AUDIT-03の要件定義
- `.planning/ROADMAP.md` — Phase 24 Success Criteria

### テストパターン
- `tests/test_post_race_leakage.py` — Phase 23で追加。3層漏洩検証テスト（プルーニング後も通過必須）
- `tests/test_backtest_engine.py` — BacktestEngine既存テスト（ROI検証時の回帰テスト）

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **compute_all_model_importance()** (`src/features/win_feature_analysis.py:171`): Phase 23で実装済み。全モデルのgain+permutation重要度を一括計算。CSV pivot table + JSON metadata出力。Phase 24ではこれを実行して結果を分析する。
- **identify_noise_features()** (`src/features/win_feature_analysis.py`): SHAP閾値+gain閾値でノイズ検出。Phase 24ではpermutation閾値も加えた拡張が必要。
- **validate_noise_removal()** (`src/features/win_feature_analysis.py`): 時系列分割で除外前後のlogloss/AUC比較。D-03の安全性確認に直接利用可能。
- **analyze_feature_importance.py CLI** (`scripts/analyze_feature_importance.py`): --all-models --format both で全モデル監査を実行可能。Phase 24の出発点。

### Established Patterns
- **FEATURE_COLSリスト**: モデルクラスにFEATURE_COLS list[str]を定義。Phase 23のCQR whitelist化で統一パターン確立済み。
- **mockベーステスト**: 全テストがDB不要。unittest.mock使用。
- **OOF予測**: TrainingPipelineのバリデーション分割（race_date後方20%）で生成。logloss/AUC計算に直接利用可能。
- **フルバックテスト**: `run_backtest.py --ensemble --calibration-bt --report`で実行。所要時間~57分/年。

### Integration Points
- **feature_engine.py:compute_cache_key()** — コードハッシュ計算の追加（AUDIT-03）
- **feature_engine.py:is_cache_valid()** — 古いキャッシュ自動削除ロジックの追加
- **各モデルのFEATURE_COLS** — Tier 1ノイズ特徴量の除外（AUDIT-02）
- **win_feature_analysis.py** — Tier 1/Tier 2判定ロジックの追加または拡張
- **scripts/run_backtest.py** — プルーニング後のフルBT実行

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・堅牢性を優先
- 監査スクリプトはPhase 23で構築済み。Phase 24は「実行して結果に基づきアクションする」実践フェーズ
- v1.5ベースライン ROI 84.4% が比較基準。プルーニング単体でROI改善できれば理想的だが、少なくともROI悪化は防ぐ
- AbilityModelは89特徴量、ConformalEVModelは~120特徴量と最多。ノイズ特徴量の絶対数も大きい可能性
- ConformalEVModelのFEATURE_COLSは~120と最多 — プルーニング効果が最も期待できる
- キャッシュ無効化はPhase 25（Quick Win Wire）の前に入れておく必要がある — 新特徴量配線時に確実にキャッシュが再計算される

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 24-Feature Audit & Pruning*
*Context gathered: 2026-05-12*
