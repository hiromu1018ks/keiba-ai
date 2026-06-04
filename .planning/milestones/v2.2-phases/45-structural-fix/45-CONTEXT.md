# Phase 45: Structural Fix - Context

**Gathered:** 2026-05-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 43/44で特定されたROI劣化原因に基づき、MAWC (MarketAwareWinCalibrator) 単一コンポーネントの構造的修正を適用し、その汎化性をOOF指標で確認する修正フェーズ。

**Phase 44ビセクションの主要知見:**
- MAWCのlogit_market係数支配 (beta=0.90) がECE劣化の主因
- odds_band 1-3でECE 3x悪化 (shadow 0.1444 vs baseline 0.0447)
- MAWCがfavorite/低オッズ帯の確率を抑制 → EVがselection gate閾値を下回る → bet_count -22%
- Rankerはdormant/shadow_only (investment_score non-NaN = 0%) であり、bet_count低下の原因ではない
- OBFは非因果（上流変更の波及効果のみ）
- selection thresholdはMAWCが作ったEV低下の結果を受けているだけ

**In scope:**
- FIX-01: MAWC保守的仕様への構造変更（再学習、交互作用項削除、強正則化C探索）
- FIX-02: 修正版MAWCのOOF汎化確認（品質ゲート + favorite band guard）
- 修正版MAWCの新規variant保存（既存モデル不変更）
- manifest生成（source_model_dir, mawc_fix_version, C_grid, removed_interactions, guard結果）

**Out of scope:**
- Ranker修正・閾値調整（dormantのため対象外）
- OddsBandFilter再学習・閾値調整
- Selection gate閾値変更
- 新特徴量追加
- BT再実行・Shadow Comparison再実行（Phase 46）
- ROI評価（Phase 46）
- レジーム別分析・パラメータ調整

</domain>

<decisions>
## Implementation Decisions

### 修正スコープ

- **D-01:** Phase 45の修正スコープは **MAWC単一コンポーネント**。Ranker/OBF/selection閾値は一切変更しない。
  - 理由: Rankerはdormant、OBFは非因果、selection thresholdはMAWC EV低下の結果。変更範囲を最小化し因果追跡を容易にするため。
  - selection閾値を緩めるとMAWCの確率歪みを隠すだけになる。

### MAWC修正手法

- **D-02:** MAWC保守的仕様への構造変更（OOF再学習ベース）。
  - 係数事後clamping / RacePredictor側サンドイッチ補正は不採用。学習済み係数を後から手で曲げると確率モデルとしての整合性が崩れ、2024/2025固有チューニングに近づくため。
  - LogisticRegressionのC grid探索: [0.003, 0.005, 0.01, 0.03]（強正則化側）
  - 削除する高リスク交互作用項:
    - logit_model × popularity top/favorite
    - logit_model × low odds band (1-2, 2-3, 1-3)
    - 必要に応じて odds_band × logit_model 系を全削除
  - 保持する main effects:
    - logit_model
    - logit_market
    - log_odds
    - popularity_rank_pct
    - p_win_race_rank_pct
    - segment one-hot main effects (odds_band, popularity_band, probability_rank_band)
  - デプロイ条件に favorite band guard を追加:
    - odds 1-3 の ECE がbaselineより悪化しない
    - odds 1-3 の bet_countが大幅に落ちない
    - year-level APRがbaselineより大きく悪化しない
  - LogisticRegression単一構造（LightGBMや追加補正層は使わない）

### 修正版保存戦略

- **D-03:** 新規variant保存。既存モデル不変更。
  - 既存 `data/models-backtest/` は読み取り専用
  - 修正版は `data/models-backtest-mawc-conservative/{year}/` に新規作成
  - ディレクトリ内は元モデルコピー + `market_aware_win_calibrator_{surface}.joblib` のみ修正版置換
  - manifestに以下を記録:
    - source_model_dir: コピー元パス
    - mawc_fix_version: 修正版バージョン識別子
    - C_grid: 探索したC値リスト
    - removed_interactions: 削除した交互作用項リスト
    - favorite_band_guard_result: guard条件の結果
  - Phase 46では baseline vs mawc_conservative として比較
  - ModelLoader.load_from_dir()が年ディレクトリ前提のため、root差し替え方式で対応

### モデル選択基準

- **D-04:** 最小C選択（品質ゲート通過候補の中で最小Cを採用）。
  1. 全体Brier/logloss/ECEがbaseline MAWCより悪化しない
  2. 年度別Brier/logloss/ECEが大きく悪化しない
  3. odds 1-3 favorite band guard を満たす（ECE非悪化、bet_count維持、APR非悪化）
  4. odds 1-3のp過度圧縮チェック（mean(p_conservative/p_model)が極端に低くならない）
  5. 複数候補あり → 最小C採用
  6. 全候補不適格 → 既存MAWC維持（shadow_only/not_deployed としてmanifest記録）
  - Brier最小・ROI最大では選ばない。汎化性優先。

### 汎化確認戦略

- **D-05:** Phase 45の汎化確認は OOF品質 + 軽量proxy確認に限定。Phase 46で全fold検証。
  - Phase 45で確認する項目:
    1. OOF予測のBrier/logloss/ECE
    2. 年度/surface/odds_band別（特にodds 1-3 favorite band guard）
    3. odds 1-3のmean(p_conservative/p_model)過度圧縮チェック
    4. odds 1-3のEV>=1.0通過率比較
    5. Year-level APR非悪化確認
    6. 全候補不適格 → not_deployed記録
  - Phase 45では実施しない: BT再実行、Shadow Comparison再実行、ROI評価
  - Phase 46で実施: baseline vs mawc_conservative 全fold Shadow Comparison + DeploymentGateEvaluator

### Claude's Discretion

- MAWC再学習の具体的OOFデータ分割方法（既存oof_predictions.parquetからの特徴量抽出方法）
- 削除交互作用項の正確な特定（FEATURE_COLSからの該当列名特定）
- C grid評価の実装詳細（LogisticRegression CV vs 手動OOF評価）
- conservative variantのディレクトリ構造・ファイル命名
- テスト構造・命名（既存規約に従う）
- JSON manifestのスキーマ設計
- favorite band guardの閾値の具体的数値（ECE差分許容幅等）

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 44 成果物（主入力）
- `data/backtest/shadow/shadow_comparison_result.json` — Phase 41 baseline vs shadow メトリクス。ECE悪化・bet_count低下の詳細データ。
- `data/backtest/shadow/shadow_horse_diff.parquet` — 馬単位 diff。p_win, p_win_market_aware 列。MAWC効果分析の主要入力。
- `data/backtest/shadow/shadow_race_diff.parquet` — レース単位 diff。bet_count分析用。
- `data/backtest/shadow/diagnosis/shadow_diagnosis_result.json` — Phase 43 診断結果。セグメント別ECE/APR。
- `data/backtest/shadow/gates/deployment_gate_result.json` — Phase 42 gate評価結果。

### MAWC コンポーネント定義
- `src/models/market_aware_win_calibrator.py` — Phase 39 MAWC。LogisticRegression 51-dim。FEATURE_COLS定義、coef_構造、fit/predict メソッド。Phase 45の主修正対象。
- `src/domain/models.py` — SubmodelSet。market_aware_win_calibrator フィールド。

### OOFデータ
- `data/oof/oof_predictions.parquet` — OOF予測。MAWC再学習の入力データソース。
- `src/pipelines/training_pipeline.py` — TrainingPipelineV5。OOF生成パス。MAWC fitの呼び出し箇所。

### Pipeline Integration Points
- `src/backtest/race_predictor.py` — RacePredictor。MAWC呼び出し (lines 269-277)。enable_market_aware_calibrator フラグ。
- `src/backtest/shadow_comparison.py` — ShadowComparisonFramework。N-way variant対応。Phase 46で mawc_conservative variant追加用。
- `src/backtest/deployment_gates.py` — DeploymentGateEvaluator。Phase 46で使用。
- `scripts/run_shadow_comparison.py` — Phase 41 CLI。Phase 46で --shadow-root data/models-backtest-mawc-conservative を指定可能。

### 診断基盤
- `src/backtest/component_attribution.py` — Phase 44 ComponentAttribution。MAWC 51-dim係数分析。analyze_mawc_coefficients()。
- `src/backtest/shadow_diagnosis.py` — Phase 43 ShadowDiagnosis。セグメント定数 POPULARITY_BAND_EDGES 等。
- `src/backtest/historical_bisect.py` — Phase 44 HistoricalBisect。

### Model Loading
- `src/db/model_loader.py` — ModelLoader。load_from_dir()。年ディレクトリ前提。
- `data/models-backtest/` — 既存学習済みモデル（読み取り専用）。

### Requirements
- `.planning/REQUIREMENTS.md` — FIX-01, FIX-02 (Phase 45 requirements)。
- `.planning/ROADMAP.md` — Phase 45 success criteria (3 items)。
- `.planning/PROJECT.md` — Key Decisions (MAWC replaces WinBenterGate+WinSegmentCalibrator, 配備条件=確率品質)。

### Prior Phase Context
- `.planning/phases/44-roi-bisect/44-CONTEXT.md` — Phase 44 ビセクション設計。MAWC/Ranker係数分析結果。
- `.planning/phases/43-shadow-diagnosis/43-CONTEXT.md` — Phase 43 診断設計。セグメント定義。
- `.planning/phases/42-feature-routing-audit-safety-gates/42-CONTEXT.md` — Phase 42 GatePolicy定義。
- `.planning/phases/39-marketawarewincalibrator/39-CONTEXT.md` — Phase 39 MAWC設計。51-dim特徴量定義。

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **MAWC LogisticRegression** (`src/models/market_aware_win_calibrator.py`): 既存のfit/predict実装。FEATURE_COLS定義から51-dim特徴量を特定可能。coef_で係数分析済み。
- **ShadowComparisonFramework** (`src/backtest/shadow_comparison.py`): N-way variant対応済み。Phase 46で --shadow-root data/models-backtest-mawc-conservative を指定してbaseline vs conservative比較可能。
- **ComponentAttribution** (`src/backtest/component_attribution.py`): analyze_mawc_coefficients() で51-dim係数分析済み。MAWC交互作用項の特定に活用。
- **OOFデータ** (`data/oof/oof_predictions.parquet`): MAWC再学習用OOF予測データ。既存のoof_predictionsからMAWC入力特徴量を再構築可能。
- **TrainingPipelineV5** (`src/pipelines/training_pipeline.py`): MAWC fit呼び出しパス。OOF生成ロジック。MAWC再学習はTrainingPipeline外で独立実行する方が安全。

### Established Patterns
- **Model variant保存**: data/models-backtest-{variant_name}/{year}/ パターン。Shadow Comparison Frameworkがroot差し替え対応。
- **JSON + manifest出力**: Phase 41/42/43/44 パターン。manifestにSHA256 + メタデータ記録。
- **Segment定数**: POPULARITY_BAND_EDGES [1-3, 4-6, 7-9, 10-14, 15+], ODDS_BAND_EDGES [1-3, 3-5, 5-10, 10-30, 30+], PROB_RANK_BAND_EDGES [top1, 2-3, 4-6, 7+]。

### Integration Points
- **入力**: data/models-backtest/ (既存モデル読み取り), data/oof/oof_predictions.parquet (OOFデータ), data/backtest/shadow/ (Phase 41/43/44 成果物)
- **出力**: data/models-backtest-mawc-conservative/{year}/ (修正版MAWCのみ置換), manifest JSON
- **消費者**: Phase 46 (Quality Gate Verification) が mawc_conservative variant を Shadow Comparison で評価

</code_context>

<specifics>
## Specific Ideas

- MAWC再学習はTrainingPipelineV5のfitパスを流用せず、独立したスクリプト/モジュールとして実装する。理由: TrainingPipelineは全体パイプライン（12モデル学習+OOF生成）を担当しており、MAWCだけを再学習する用途には大きすぎる。既存MAWCのfitロジック（feature構築→LogisticRegression fit→保存）を抽出した軽量モジュールが適切。
- 削除対象の交互作用項はComponentAttributionの係数分析結果（analyze_mawc_coefficients()）から正確に特定する。odds_band × logit_model 系がECE悪化に寄与している係数を特定し、削除リストを作成。
- C grid [0.003, 0.005, 0.01, 0.03] は各C値でOOF上にfit→predict→品質ゲート評価を繰り返す。最も正則化が強い（最小C）ゲート通過モデルを採用。
- p過度圧縮チェック: mean(p_conservative / p_model) がodds 1-3帯で0.9以下に落ちないことを確認。0.9未満は過度な抑制と判断。
- EV>=1.0通過率: odds 1-3帯で p_conservative * odds >= 1.0 となる馬の割合が、既存MAWC比で大きく低下しないことを確認。

</specifics>

<deferred>
## Deferred Ideas

- **Ranker修正 (investment_score重み・閾値調整)**: Phase 45対象外。Rankerはdormant。必要ならPhase 46後の候補。
- **OddsBandFilter再学習・閾値調整**: Phase 45対象外。OBFは非因果。Phase 46後の候補。
- **Selection gate閾値調整**: Phase 45対象外。MAWC修正の結果として自然に改善されることを期待。
- **全12モデルSHAP/gain比較**: Phase 44範囲外。Phase 46後の候補。
- **レジーム別分析・レジーム別パラメータ調整**: REQUIREMENTS.mdで除外。v2.3+で検討。
- **新特徴量追加**: REQUIREMENTS.mdで除外。v2.3+で検討。
- **デプロイゲート自動判定 (DEP-01)**: v2.3+に延期。
- **Optuna 19次元パラメータ最適化 (DEP-02)**: v2.3+に延期。

</deferred>

---

*Phase: 45-Structural Fix*
*Context gathered: 2026-05-31*
