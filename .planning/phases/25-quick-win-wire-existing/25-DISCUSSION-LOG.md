# Phase 25: Quick Win Wire Existing - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-12
**Phase:** 25-quick-win-wire-existing
**Areas discussed:** Win model feature selection, Stage1 & Place wiring, ROI verification scope

---

## Win Model Feature Selection

| Option | Description | Selected |
|--------|-------------|----------|
| All 12 features | 12特徴量すべてをWinTwoStageModel.FEATURE_COLSに追加。Phase 24の監査スクリプトで重要度を評価 | ✓ |
| Curated subset | ドメイン知識で厳選（4-6個に限定）。ノイズ最小化 | |
| Same 3 as Place HIT | jockey_wr_overall, trainer_wr_overall, jt_combo_place_rateの3個のみ。一貫性重視 | |

**User's choice:** All 12 features
**Notes:** LightGBMが不要特徴量を自動的に無視できることと、Phase 24の監査スクリプトで後評価可能なことを考慮。

---

## Stage1 Ability Model Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Skip Stage1 | Stage1は「馬の能力評価」に集中。モジュールdocstring「Stage2のみ」に従う | ✓ |
| Include in Stage1 | 騎手・調教師の実力は馬の能力評価に影響するという観点 | |

**User's choice:** Skip Stage1 (Recommended)
**Notes:** Stage1の設計方針（馬自身の能力に集中）を維持。

---

## PlaceTwoStageModel Wiring

| Option | Description | Selected |
|--------|-------------|----------|
| Full 12 to both | HIT_FEATURE_COLSに残り9個 + RETURN_FEATURE_COLSに12個を追加 | |
| Only fill HIT gaps | HIT_FEATURE_COLSの残り9個のみ。Return modelはオッズ情報重視 | |
| Defer to post-audit | Win modelのみ対応。Placeは監査結果後に判断 | |

**User's choice:** Other — "実装難易度は問わないのでベストプラクティスを追求"
**Notes:** ユーザーの自由入力。ベストプラクティスに従い、Full 12 to both として扱う。EVCorrection/ConformalEVと同じ完全配線に統一。

---

## ROI Verification Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Defer to Phase 28 | Phase 25ではコード変更とテスト確認のみ。ROI検証はPhase 28に一任 | |
| Quick OOF check only | OOF logloss/AUCで簡易確認。フルBTはPhase 28 | |
| Full backtest now | Phase 25でフルバックテスト（~57分）を実行。Phase 26-27の前に効果確認 | ✓ |

**User's choice:** Full backtest now
**Notes:** Phase 26-27の新特徴量追加前に配線効果を確認する意図。Phase 24後のベースラインも同時に確立可能。

---

## Paper Trading Path (discovered during analysis)

JockeyTrainerComboFeaturesがpaper_trading/predictor.pyで未計算であることを分析中に発見。D-05として暗黙的に決定（ユーザー承認済み）。

---

## Claude's Discretion

- FEATURE_COLSへの具体的な挿入位置
- バックテストコマンド構成とベースライン比較方法
- テストの追加・更新内容
- POST_RACE漏洩テストの通過確認方法

## Deferred Ideas

None — discussion stayed within phase scope
