---
status: resolved
trigger: "sire_x_cushion_band が _prepare_features のカテゴリカル変換リストに含まれておらず、LightGBM が9つ目のカテゴリカル特徴量を認識できない"
created: 2026-06-07
updated: 2026-06-07
---

# Debug Session: cat-feature-list-incomplete

## Symptoms

- **Expected behavior:** 全カテゴリカル特徴量が推論時にも正しく category dtype に変換され、学習時と同じカテゴリカルセットが LightGBM に渡される
- **Actual behavior:** sire_x_cushion_band が NaN 埋めで float64 になり、LightGBM が9つ目のカテゴリカルを認識できない
- **Error messages:** ValueError: train and valid dataset categorical_feature do not match
- **Timeline:** 前回セッション(pt-cat-feature-mismatch)のNaN埋め修正適用後も継続発生
- **Reproduction:** PT predict 実行時

## Root Cause (user-provided)

- sire_x_cushion_band（種牡馬系統×クッションバンド交互作用）は学習時は category dtype
- _prepare_features のカテゴリカル変換リスト（ハードコード8列）に含まれていない
- PT 推論時は NaN 埋めで float64 になる → LightGBM が9つ目のカテゴリカルを認識できない
- より堅牢な修正として、FEATURE_COLS 内の全 category/object 列を動的に検出して変換することが推奨される

## Current Focus

- **hypothesis:** ハードコードされたカテゴリカル列リストが不完全。sire_x_cushion_band が漏れているだけでなく、他の漏れもある可能性
- **test:** 全モデルの _prepare_features のカテゴリカルリストと、学習データ FEATURE_COLS 内の実際の category/object 列を比較
- **expecting:** FEATURE_COLS 内の全 category/object 列が網羅されること
- **next_action:** 全モデルの _prepare_features を調査し、動的カテゴリカル検出または完全なリストで修正
- **reasoning_checkpoint:** ハードコードリストは脆弱。学習データの dtype に基づく動的検出が最も堅牢

## Evidence

- (to be collected by session manager)

## Eliminated

- (none yet)

## Resolution

- **root_cause:** sire_x_cushion_band (Phase 48 トラック条件交互作用特徴量) は FeatureEngine で category dtype として生成されるが、6モデルの _prepare_features ハードコードカテゴリカルリストに含まれていなかった。PT推論時に NaN 埋めで float64 になり、LightGBM が categorical_feature do not match エラーを発生させていた。
- **fix:** 全6モデル(7箇所)の _prepare_features に sire_x_cushion_band をハードコードリストに追加。さらに、FeatureEngine で category/object dtype として生成された列を動的に検出して category 変換するフォールバックロジックを追加し、将来の新規カテゴリカル特徴量の漏れを防止。
- **verification:** 261 model-related tests passed (0 failures)。既存テストへの回帰なし。
- **files_changed:**
  - src/models/stage1_ability_model.py (_prepare_features)
  - src/models/two_stage_return_model.py (WinTwoStageModel._prepare_features, PlaceTwoStageModel._prepare_features)
  - src/models/ev_correction_model.py (EVCorrectionModel._prepare_features, PlaceEVCorrectionModel._prepare_features)
  - src/models/place_ability_model.py (train/predict 2箇所)
  - src/models/wide_two_stage_model.py (train_hit_model/train_return_model/predict_score 3箇所)
