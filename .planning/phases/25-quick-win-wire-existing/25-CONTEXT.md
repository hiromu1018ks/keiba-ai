# Phase 25: Quick Win Wire Existing - Context

**Gathered:** 2026-05-12
**Status:** Ready for planning

<domain>
## Phase Boundary

既に実装・テスト済みのJockey/Trainer/Combo合計12特徴量をMLモデルのFEATURE_COLSに追加し、フルバックテストでROI改善を確認する。特徴量の計算自体はtraining_pipeline.pyとbacktest/engine.pyに既に実装済み。欠けているのは各モデルのFEATURE_COLSリストへの追加と、paper_tradingパスへのJockeyTrainerComboFeatures追加のみ。

**In scope:**
- WIRE-01: JockeyContextFeatures(4特徴量)のWinTwoStageModel + PlaceTwoStageModel FEATURE_COLSへの追加
- WIRE-02: TrainerContextFeatures(4特徴量)のWinTwoStageModel + PlaceTwoStageModel FEATURE_COLSへの追加
- WIRE-03: JockeyTrainerComboFeatures(4特徴量)のWinTwoStageModel + PlaceTwoStageModel FEATURE_COLSへの追加
- paper_trading/predictor.pyへのJockeyTrainerComboFeatures計算追加
- フルバックテスト実行によるROI改善確認（v1.5ベースライン ROI 84.4%またはPhase 24後ベースラインとの比較）
- 既存テスト全通過確認

**Out of scope:**
- Stage1AbilityModelへの追加（Stage2 onlyの設計方針を維持）
- 新しい特徴量の追加（Phase 26）
- 特徴量の交互作用・変換（Phase 27）
- 最終ROI検証・特徴量凍結（Phase 28）
- モデル再学習・ハイパーパラメータ調整
- 複勝/ワイドモデルの変更
- feature_engine.pyのbuild_all()への統合（現状の独立計算パターンを維持）

</domain>

<decisions>
## Implementation Decisions

### Win Model Feature Selection (WIRE-01/02/03)
- **D-01:** WinTwoStageModel.FEATURE_COLSに12特徴量すべてを追加する。Phase 24で構築した監査スクリプト（permutation+gain重要度）で効果を後評価可能。LightGBMは不要特徴量を自動的にgain=0にするため、全追加は安全なアプローチ。

### Stage1 & Place Model Scope
- **D-02:** Stage1AbilityModelには追加しない。モジュールdocstringが「Stage2のみ」を想定しており、Stage1は馬自身の能力評価に集中する設計方針を維持する。
- **D-03:** PlaceTwoStageModelのHIT_FEATURE_COLS（残り9個）とRETURN_FEATURE_COLS（12個すべて）に追加する。EVCorrectionModel/PlaceEVCorrectionModel/ConformalEVModelは既に全12特徴量を含んでいるため、2段階モデルも完全配線に統一する。ベストプラクティスを追求する方針。

### ROI Verification
- **D-04:** Phase 25でフルバックテストを実行し、配線前後のROIを比較する。Phase 26-27の新特徴量追加前に効果を確認する。バックテストコマンド: `run_backtest.py --ensemble --calibration-bt --report`（所要時間~57分/年）。

### Paper Trading Path
- **D-05:** paper_trading/predictor.pyにJockeyTrainerComboFeaturesの計算を追加する。現状はJockeyContextFeaturesとTrainerContextFeaturesのみ計算しており、jt_combo特徴量が欠落している。Win modelがjt_combo特徴量を使用するため、paper tradingパスでも必須。

### Claude's Discretion
- 各FEATURE_COLSへの具体的な挿入位置（既存特徴量グループの論理的な箇所）
- バックテスト実行の具体的なコマンド構成と比較ベースライン（Phase 24後ベースライン vs v1.5ベースライン）
- テストの追加・更新内容（FEATURE_COLS変更に伴うmockテスト調整）
- POST_RACE漏洩テスト（Phase 23で追加）が配線後も通過することの確認方法

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 特徴量モジュール（実装済み — 配線対象）
- `src/features/jockey_context_features.py` — JockeyContextFeatures。4特徴量: jockey_wr_overall, jockey_wr_distance, jockey_wr_venue, jockey_prize_log。x_KISYU_SEISEKIからSetYear < race_yearの最新年を使用。
- `src/features/trainer_context_features.py` — TrainerContextFeatures。4特徴量: trainer_wr_overall, trainer_wr_distance, trainer_wr_venue, trainer_prize_log。x_CHOKYO_SEISEKIからSetYear < race_yearの最新年を使用。
- `src/features/jockey_trainer_combo.py` — JockeyTrainerComboFeatures。4特徴量: jt_combo_wr, jt_combo_place_rate, jt_combo_starts, jt_combo_prize_log。過去出走データから騎手-調教師コンビの実績を計算。

### FEATURE_COLS定義（変更対象）
- `src/models/two_stage_return_model.py:48-102` — WinTwoStageModel.FEATURE_COLS (40特徴量)。12特徴量の追加先。
- `src/models/two_stage_return_model.py:289-340` — PlaceTwoStageModel.HIT_FEATURE_COLS。現状3/12、残り9個を追加。
- `src/models/two_stage_return_model.py:345-400` — PlaceTwoStageModel.RETURN_FEATURE_COLS。0/12、12個すべてを追加。
- `src/models/ev_correction_model.py:134-141` — EVCorrectionModel.FEATURE_COLS。既に全12特徴量を含む（参考）。
- `src/models/conformal_ev_model.py:134-141` — ConformalEVModel.FEATURE_COLS。既に全12特徴量を含む（参考）。
- `src/models/stage1_ability_model.py:28-128` — Stage1AbilityModel.FEATURE_COLS (89特徴量)。追加しない（D-02）。

### 既存の計算パス（確認用 — 変更不要）
- `src/pipelines/training_pipeline.py:549-569` — 学習パス。既に3モジュール計算＋df_oofにmerge済み。
- `src/backtest/engine.py:654-675` — バックテストパス。既に3モジュール事前計算済み。

### Paper Trading Path（D-05で変更）
- `src/paper_trading/predictor.py:55-56,100-104` — setup()でJockeyContext + TrainerContextを計算。JockeyTrainerComboFeaturesの追加が必要。

### テスト（既存 + 更新確認）
- `tests/test_jockey_context_features.py` — JockeyContextFeatures単体テスト（既存）
- `tests/test_trainer_context_features.py` — TrainerContextFeatures単体テスト（既存）
- `tests/test_jockey_trainer_combo.py` — JockeyTrainerComboFeatures単体テスト（既存）
- `tests/test_post_race_leakage.py` — Phase 23で追加。3層漏洩検証テスト。配線後も通過必須。
- `tests/test_two_stage_return_model.py` — Win/Place 2段階モデルテスト。FEATURE_COLS変更の影響確認。

### バックテスト・ROI検証（D-04）
- `scripts/run_backtest.py` — バックテストCLI。--calibration-bt, --report, --ensemble等
- `src/backtest/engine.py` — BacktestEngine。run()でフルBT実行
- `src/backtest/validation_report.py` — generate_validation_report() ROI PASS/FAIL判定

### 前フェーズのCONTEXT（決定の連続性）
- `.planning/phases/24-feature-audit-pruning/24-CONTEXT.md` — Phase 24決定（プルーニング基準、Tier分類、キャッシュ無効化）
- `.planning/phases/23-safety-gate/23-CONTEXT.md` — Phase 23決定（漏洩修正、監査スクリプト設計）

### 要件定義
- `.planning/REQUIREMENTS.md` — WIRE-01, WIRE-02, WIRE-03の要件定義
- `.planning/ROADMAP.md` — Phase 25 Success Criteria

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **特徴量計算ロジック**: training_pipeline.py (L549-569) と backtest/engine.py (L654-675) で既に3モジュールの計算が実装済み。FEATURE_COLSへの追加のみでモデルが特徴量を利用可能になる。
- **Beta(1,10) smoothing**: 全モジュールで統一された勝率平滑化 (wins+1)/(total+11)。小サンプルでの安定性を確保。
- **Phase 23 監査スクリプト**: `scripts/analyze_feature_importance.py`。--all-modelsで全モデルのpermutation+gain重要度を計算。配線後の効果測定に使用可能。

### Established Patterns
- **FEATURE_COLSリスト**: モデルクラスにFEATURE_COLS list[str]を定義。Phase 23のCQR whitelist化、Phase 24のTier 1プルーニングで確立済みパターン。
- **mockベーステスト**: 全テストがDB不要。FEATURE_COLS変更に伴うテスト更新はmockのcolumn list更新のみ。
- **独立計算パターン**: JockeyContext/TrainerContext/JockeyTrainerComboはfeature_engine.pyのbuild_all()外で独立計算。build_all()キャッシュに依存しない設計。Phase 24のコードハッシュキャッシュ無効化の影響を受けない。

### Integration Points
- **WinTwoStageModel.FEATURE_COLS** (L48-102): 12特徴量の追加先。現在40特徴量→52特徴量に。
- **PlaceTwoStageModel.HIT_FEATURE_COLS** (L289-340): 残り9特徴量の追加先。現在42特徴量→51特徴量に。
- **PlaceTwoStageModel.RETURN_FEATURE_COLS** (L345-400): 12特徴量の追加先。現在56特徴量→68特徴量に。
- **paper_trading/predictor.py:setup()** (L100-104): JockeyTrainerComboFeaturesの計算追加。

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・堅牢性を優先
- 「Quick Win」の名前の通り、FEATURE_COLSへの文字列追加が主なコード変更。計算ロジックは既に実装済み
- Phase 24でプルーニング後のベースラインROIが未確定（フルBT実行が必要）。D-04のフルバックテストでPhase 24後ベースラインも同時に確立できる
- Win modelがROI改善の主戦。WinTwoStageModelに12特徴量を追加することが最大のインパクト
- Stage1除外の理由: Stage1は馬自身の能力（過去成績、血統、ペース適性等）に集中し、騎手・調教師コンテキストはStage2で評価する2段階設計に合致

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 25-Quick Win Wire Existing*
*Context gathered: 2026-05-12*
