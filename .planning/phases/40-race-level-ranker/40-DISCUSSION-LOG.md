# Phase 40: Race-Level Ranker - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-28
**Phase:** 40-Race-Level Ranker
**Areas discussed:** Ranker model architecture, Training target / label design, Existing selector relationship, Feature set & combination

---

## Ranker model architecture

| Option | Description | Selected |
|--------|-------------|----------|
| LightGBM (両方) | RelevanceもValueもLightGBM。非線形捕捉力が高いが過学習リスク。 | |
| Ridge (両方) | RelevanceもValueもRidge回帰。安定・解釈性高。 | |
| Ridge primary + LightGBM shadow | デフォルトはRidge、LightGBMはshadow比較用。Phase 39のD-02パターン。 | ✓ |

**User's choice:** Ridge/linear primary + LightGBM LambdaRank shadow benchmark. RelevanceとValue各Ridgeモデル。LightGBMはshadowのみ、デプロイはRidge。「simplest model that passes gates」方針。v2.1の主リスクはoverfit ranking layer。
**Notes:** Fixed validated combination formula from OOF/WF diagnostics, not 2024/2025 coefficient tuning.

| Option | Description | Selected |
|--------|-------------|----------|
| 2-model split | RelevanceとValueは別モデル。RNK-01/02に直接対応。 | ✓ |
| 1 unified model | 1つのRidgeモデル。シンプルだがRelevance/Value分離不可。 | |

**User's choice:** 2-model split. relevance_scorer + value_scorer. 結合: investment_score = a*relevance + b*value + c*calibrated_log_ev - d*uncertainty_penalty.
**Notes:** Report each component separately in shadow diagnostics so selection changes are explainable.

| Option | Description | Selected |
|--------|-------------|----------|
| WF grid search | OOF/WF diagnosticsで重みを探索。 | |
| Equal weights | a=b=1, c=d=0.5。シンプル。 | |

**User's choice:** 標準化(z-score/robust percentile)後に固定重み: a=0.35, b=0.35, c=0.20, d=0.10。2024/2025でチューニングしない。
**Notes:** Grid-searching investment_score weights on ROI/HR will recreate the overfitting loop we are trying to escape. The ranker models should learn the signal; the final blend should be simple, stable, and explainable.

| Option | Description | Selected |
|--------|-------------|----------|
| Same C-grid as Phase 39 | [0.03, 0.1, 0.3, 1.0, 3.0] + logloss primary。 | |
| Ranking-specific config | 異なるC_gridやNDCG, Spearman metric。 | ✓ |

**User's choice:** Alpha grid [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]. Relevance: NDCG@3 / top1 win relevance. Value: rank correlation + top1/top3 value capture. Tie-breaker: larger alpha. Loglossは確率モデル用、ランカーには不適切。
**Notes:** Do not tune on 2024/2025 fixed test folds. Use chronological WF folds only.

---

## Training target / label design

| Option | Description | Selected |
|--------|-------------|----------|
| is_win (binary) | シンプル。2着と殿馬の区別不可。 | |
| Position-based | finishing_positionの逆数や指数減衰。RNK-01に完全合致。 | ✓ |
| Hybrid | is_winとfinishing_positionを両方活用。 | |

**User's choice:** Graded relevance target: 1.00(1st), 0.55(2nd), 0.30(3rd), 0.10(4-5th), 0.00(otherwise). kakuteijyuniはlabelのみ。
**Notes:** Pure is_win is too sparse and ignores near-winner information. A graded relevance target is stable and matches the race-level ranking objective.

| Option | Description | Selected |
|--------|-------------|----------|
| OOF EV residual | calibrated_EVをOOF予測で計算。 | |
| Actual vs expected return | (actual_payout - expected_payout) / expected_payout。 | |
| Mispricing signal | p_model > p_market の二値分類。 | ✓ |

**User's choice:** Composite OOF-safe value target: clipped_log_ev + mispricing_bonus - uncertainty_penalty. actual returnはdiagnosticのみ。
**Notes:** Realized return is too sparse and turns the ranker into an ROI-label overfit machine. The value ranker should learn expected mispricing from OOF-calibrated probabilities.

| Option | Description | Selected |
|--------|-------------|----------|
| Extend existing OOF gen | Phase 39のgenerate_win_oof_predictions()を拡張。 | ✓ |
| Dedicated OOF generation | ranker専用の新しいOOF生成関数。 | |

**User's choice:** Phase 39のOOF生成を拡張。同じfold定義。OOFHealthValidator検証済み + IFF train-mode + MAWC OOF/shadowから構築。
**Notes:** Do not create a separate ranker-only OOF generator. Probability-derived features must be recomputed from OOF probabilities.

---

## Existing selector relationship

| Option | Description | Selected |
|--------|-------------|----------|
| Ranker replaces Policy+Profit | Rankerがinvestment_scoreを出力し既存置き換え。 | |
| Parallel shadow | Rankerはshadow、既存も並行動作。Phase 41で比較。 | ✓ |
| Full replacement | Rankerのみ。WinSelectionGateも削除。 | |

**User's choice:** Parallel shadow first. Ranker produces investment_score in shadow mode. Existing selectors remain functional behind feature flags. No deletion in v2.1.
**Notes:** Removal only considered in later cleanup milestone after stable shadow/backtest evidence. If gates pass: ranker replaces WinSelectionPolicy, WinProfitSelector stays shadow-only, WinSelectionGate remains as fallback.

| Option | Description | Selected |
|--------|-------------|----------|
| After calibrator | MAWCの直後。全ランナーにスコアリング。 | ✓ |
| After WinSelectionGate | ゲート通過馬のみ。計算効率良いがバイアス継承。 | |

**User's choice:** After MarketAwareWinCalibrator / InvestmentFeatureFrame, before final selection sorting. Score all runners — do not restrict to gate-passed horses.
**Notes:** Avoid inheriting gate selection bias. In shadow mode, add investment_score columns to diagnostics for all runners.

---

## Feature set & combination

| Option | Description | Selected |
|--------|-------------|----------|
| Curated subset | 各ランカーに6-12個の主要特徴量。解釈性重視。 | ✓ |
| All IFF features | 全94 specsを投入。L2正則化に任せる。 | |
| Category-based split | カテゴリレベルでRelevance/Value分担。 | |

**User's choice:** Curated feature subsets. Relevance: ~12-16 features (p_win, ability, form, closing, blood, jockey, class, course). Value: ~14-18 features (logit_gap, EV, odds, late_money, uncertainty, market).
**Notes:** Feature names must match Phase 38 schema registry. Use registered missing/default behavior if unavailable. No ad hoc aliases.

| Option | Description | Selected |
|--------|-------------|----------|
| Race-level z-score | (x - mean) / std。直感的。外れ値に弱い。 | |
| Robust percentile | rank / field_size。外れ値に強い。[0,1]範囲。 | ✓ |

**User's choice:** Race-level robust percentile ranks. Deterministic tie handling: rank(method="average" or "first" with stable sort by race_id/umaban). Z-scores logged as diagnostics only.
**Notes:** Small field sizes and outliers can destabilize z-scores. Percentile ranks are field-size-independent.

---

## Claude's Discretion

- Exact feature matrix construction and missing-feature handling within IFF schema rules.
- LightGBM LambdaRank shadow training configuration.
- SubmodelSet field naming for ranker models.
- Test structure and naming within existing conventions.
- Model serialization format (joblib).
- Exact integration code in RacePredictor.predict() and get_win_candidates().

## Deferred Ideas

None — discussion stayed within phase scope.
