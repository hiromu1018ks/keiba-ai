# Phase 23: Safety Gate - Context

**Gathered:** 2026-05-11
**Status:** Ready for planning

<domain>
## Phase Boundary

全特徴量パイプラインからレース後情報漏洩を排除し、特徴量品質監査の基盤を構築する。Spike調査（data-leak-phase-20-22）で特定された5件の構造的リスク（M1-M6）を一括解消し、Phase 24（Feature Audit & Pruning）で使用するfeature importance監査スクリプトを構築する。

**In scope:**
- SAFE-01: build_all()出口でPOST_RACE_COLSを確実にドロップするリーク修正
- SAFE-02: permutation重要度 + gain重要度を計算するfeature importance監査スクリプト
- Spike M1: build_all()戻り値のPOST_RACE_COLS残存修正
- Spike M2: CQR特徴量抽出のブラックリスト→whitelist化
- Spike M3: EV correction学習/推論オッズ不一致の修正
- Spike M6: popularity_rank の ninki フォールバック防止
- POST_RACE漏洩検出のCIテスト（3層検証）
- 全モデル(Ability/Win2Stage/Place2Stage/EVCorrection)のfeature importance監査スクリプト拡張

**Out of scope:**
- 特徴量の実際の削減・追加（Phase 24）
- モデル再学習・ハイパーパラメータ調整
- バックテストROI検証（Phase 28）
- WideTwoStageModelのtemporal split修正（H1 — Phase 22以降の別タスク）
- CQR過学習の根本修正（Phase 21 D-11のTODO、別フェーズで対応）
- 複勝/ワイドモデルの修正

</domain>

<decisions>
## Implementation Decisions

### 漏洩修正の適用範囲 (SAFE-01)
- **D-01:** Spike M1-M6の全5件の構造的リスクをPhase 23で一括修正する。ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。将来の漏れリスクを根本的に排除する。
- **D-02:** build_all()のキャッシュ書き込み前にPOST_RACE_COLSをドロップする。キャッシュにはクリーンなDataFrameのみ保存され、以後の読み出しは常に漏洩なし。既存のengine.py:819, run_paper_trading.py:80のdropは安全ネットとして残す（冗長だが防御的）。
- **D-03:** CQR（conformal_ev_model.py）の特徴量抽出をブラックリスト方式（437列の全numeric - 除外セット）から明示的FEATURE_COLS（whitelist）に変更する。他のモデルと同じ設計パターンに統一し、将来の列追加時に自動的に特徴量に混入するリスクを排除。
- **D-04:** M3（EV correction odds不一致）を修正 — 学習時に`confirmed_odds`（確定オッズ）でev_odds_band_scalesを計算し、推論時に`odds`（発走前オッズ）にフォールバックする不一致を解消する。学習時も発走前オッズ（tanodds）を使用するよう変更。
- **D-05:** M6（popularity_rank ninki fallback）を修正 — popularity_rankの算出で`tanodds`→`tanninki`→`ninki`（確定人気順）のフォールバックチェーンから、`ninki`へのフォールバックを除去する。`ninki`はPOST_RACEデータ。

### 監査スクリプトの設計 (SAFE-02)
- **D-06:** 既存の`scripts/analyze_feature_importance.py`を拡張して全モデル対応にする。新規スクリプトは作成しない。既存のWinTwoStageModel.hit_model分析機能は維持し、新たに全モデルのpermutation+gain重要度計算を追加。
- **D-07:** 監査対象は全モデル — Stage1AbilityModel, WinTwoStageModel(hit/return), PlaceTwoStageModel, EVCorrectionModel。Phase 24での包括的監査に必要。ただしStage1以外はOOF予測に依存するため、計算順序の設計に注意。
- **D-08:** 出力形式はCSV + JSONの両方。CSVはピボットテーブル形式（各特徴量×各モデルの重要度スコア）で人間の確認用。JSONは構造化データ（メタデータ込み）でPhase 24の自動フィルタリング処理用。

### CIテストの網羅性
- **D-09:** 3層検証を実装 — ① build_all()の出力にPOST_RACE_COLSが含まれないことを検証 ② 各モデルのFEATURE_COLSにPOST_RACE_COLSが含まれないことを検証 ③ predict()の入力にPOST_RACE_COLSが含まれないことを検証。3層保護で漏洩を完全に遮断。
- **D-10:** CIテストは新規ファイル `tests/test_post_race_leakage.py` に配置する。漏洩検出に特化した独立テストファイル。既存テストに影響なし。

### Claude's Discretion
- build_all()内のPOST_RACE_COLS dropの具体的な実装箇所（キャッシュ書き込み前の適切な行位置）
- キャッシュキー計算にPOST_RACE_COLS除外が影響するかどうかの判定（おそらく影響なし）
- CQRのFEATURE_COLSの具体的な列選定（既存437列から意味のある列を選定する基準）
- M3の具体的な修正方法（confirmed_odds→tanoddsの変更箇所、学習時のodds列の取り扱い）
- M6のフォールバック先の代替（ninkiを使わない場合の人気順の計算方法）
- 監査スクリプトのCLI引数設計（--model, --all-models, --output, --top-n等）
- permutation重要度の計算パラメータ（n_repeats, scoring metric）
- テストのfixtureデータとモック構成
- predict()入力検証のテスト実装方法（mockでRacePredictorを構築して検証）

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Spike調査結果（必読 — リスクの全体像）
- `.planning/spikes/data-leak-phase-20-22.md` — Spike M1-M6の5件の構造的リスクの詳細。Severity分類・Root Cause・Fix内容

### POST_RACE定義（SAFE-01の中心）
- `src/domain/types.py:38-55` — POST_RACE_COLS定義（16列）。漏洩検出の基準
- `src/features/feature_engine.py:135-300` — FeatureEngine.build_all()。POST_RACE_COLS dropの追加箇所
- `src/features/feature_engine.py:295-300` — キャッシュ書き込み箇所。dropはこれより前に挿入

### CQR whitelist化（M2）
- `src/models/conformal_ev_model.py:55-70` — CQR特徴量抽出のブラックリスト定義。whitelist化対象
- `src/models/conformal_ev_model.py:18` — POST_RACE_COLS import
- `src/pipelines/training_pipeline.py:902` — CQR学習時の除外セット定義

### EV correction odds不一致（M3）
- `src/models/ev_correction_model.py:370` — confirmed_oddsでev_odds_band_scalesを計算する箇所
- `src/backtest/engine.py:819` — 推論前にconfirmed_oddsをdropする箇所（oddsへのフォールバック発生）

### popularity_rank fallback（M6）
- `src/features/feature_engine.py:421-463` — popularity_rank算出。tanodds→tanninki→ninkiのフォールバックチェーン

### バックテスト・ペーパートレードの既存drop（安全ネット）
- `src/backtest/engine.py:819-834` — 推論前のPOST_RACE_COLS drop + 精算後の復元
- `scripts/run_paper_trading.py:57-80` — ペーパートレードのPOST_RACE_COLS drop

### 監査スクリプト（SAFE-02の拡張対象）
- `scripts/analyze_feature_importance.py` — 既存のSHAP/gain特徴量重要度分析スクリプト。WinTwoStageModel.hit_model限定

### 既存リーク検証モジュール
- `src/features/leakage_validators.py` — expanding特徴量専用のリーク検証。新しいCIテストの参考

### モデルFEATURE_COLS（CIテスト検証対象）
- `src/models/stage1_ability_model.py:28-107` — Stage1AbilityModel.FEATURE_COLS
- `src/models/two_stage_return_model.py:289-404` — WinTwoStageModel HIT/RETURN FEATURE_COLS
- `src/models/ev_correction_model.py` — EVCorrectionModel.FEATURE_COLS

### 要件定義
- `.planning/REQUIREMENTS.md` — SAFE-01, SAFE-02の要件定義
- `.planning/ROADMAP.md` — Phase 23 Success Criteria

### 前フェーズのCONTEXT（決定の連続性）
- `.planning/phases/22-integrated-validation/22-CONTEXT.md` — Phase 22決定（バックテスト検証、v1.4ベースライン比較）

### テストパターン
- `tests/test_leakage_validators.py` — 既存リーク検証テスト。新しいテストのパターン参考
- `tests/test_backtest_engine.py:312` — POST_RACE_COLSを使った既存テスト

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **POST_RACE_COLS** (`src/domain/types.py:38-55`): 16列のpost-raceカラム定義。build_all()のdrop、CIテストの検証基準としてそのまま使用
- **leakage_validators.py** (`src/features/leakage_validators.py`): expanding特徴量専用だが、API設計（validate_→issues list返却）は新しいCIテストの参考パターン
- **analyze_feature_importance.py** (`scripts/analyze_feature_importance.py`): CLI引数設計、SHAP/gain計算ロジック、CSV出力パターンの基盤
- **build_all()のキャッシュ機構** (`src/features/feature_engine.py:155-200`): キャッシュ書き込み前のdropはキャッシュ無効化トリガーになる可能性あり

### Established Patterns
- **mockベーステスト**: 全テストがDB不要。unittest.mock使用。新しいCIテストもこのパターン
- **FEATURE_COLSリスト**: モデルクラスにFEATURE_COLS list[str]を定義。CQRもこのパターンに統合
- **POST_RACE_COLS一元管理**: domain/types.pyで定義。全モジュールがこれをimport
- **3段階ベットフィルター**: COLLAPSED skip → 動的EV_lower → OddsBandFilter。POST_RACE漏洩はこのフィルターの前段で防止

### Integration Points
- **feature_engine.py:build_all()末尾** — POST_RACE_COLS dropの追加（キャッシュ書き込み前）
- **conformal_ev_model.py** — FEATURE_COLS whitelistへの変更
- **ev_correction_model.py:370** — odds列の修正（confirmed_odds→tanodds）
- **feature_engine.py:421-463** — popularity_rank fallback chainの修正
- **scripts/analyze_feature_importance.py** — 全モデル対応の拡張
- **tests/test_post_race_leakage.py** — 新規CIテストファイル

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・堅牢性を優先
- Spike調査の結果、直接的な特徴量リークは見つかっていないが、構造的リスク（M1-M6）が残存。これらをPhase 23で一括解消することで、Phase 24以降の特徴量改善作業を安全に進められる
- CQRの437列ブラックリストは最もリスクが高い — 新しい列が追加されるたびに自動的に特徴量に混入する可能性がある
- 監査スクリプトはPhase 24で「重要度ゼロ/負の特徴量をFEATURE_COLSから除外」する際の判断基準として直接使用される
- M3（EV correction odds不一致）はsubtleだが重要 — 学習時と推論時で異なるオッズを使うと、odds-bandスケーリングが実運用と乖離する

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 23-Safety Gate*
*Context gathered: 2026-05-11*
