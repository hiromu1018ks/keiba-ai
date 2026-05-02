# Phase 3: Selection Gate, Confidence & Betting - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-02
**Phase:** 3-Selection Gate, Confidence & Betting
**Areas discussed:** WinSelectionGate設計, Conformal信頼性推定, JRA控除率とエッジ閾値

---

## WinSelectionGate設計

### Q1: WinSelectionGateのアーキテクチャ

| Option | Description | Selected |
|--------|-------------|----------|
| PlaceSelectionGateの完全踏襲 | OOF walk-forward score tables + smoothed scoring + add-second reranker + soft_pass_mask全て再現 | ✓ |
| スコアリングのみ踏襲 | score tables + walk-forward閾値最適化は踏襲、add-second rerankerは省略 | |
| 軽量アプローチ | score tablesを使わず、EV/edgeの単純閾値 + RegimeDetector連動のみ | |

**User's choice:** PlaceSelectionGateの完全踏襲 (Recommended)
**Notes:** 単勝も複勝と同等の厳密な選択が必要

### Q2: WinSelectionGateの入力変数

| Option | Description | Selected |
|--------|-------------|----------|
| 同一構造 | prob/edge/oddsの3次元binning。place_selection_* → win_selection_*, fukuoddslow → tanoddslow | |
| Win特化入力追加 | JRA控除率調整後エッジ、Benterブレンド度合い、オッズバケット細分化 | |

**User's choice:** ベストプラクティス（同一構造を採用）
**Notes:** 3次元binningを維持、Win特化入力は追加しない

### Q3: オッズソース

| Option | Description | Selected |
|--------|-------------|----------|
| tanoddslow | 最終単勝オッズ。バックテストで既存列 | ✓ |
| tanoddshigh | 最高単勝オッズ。データ欠損リスク | |

**User's choice:** tanoddslow (Recommended)
**Notes:** PlaceSelectionGateがfukuoddslowを使うのと同じパターン

### Q4: add-second reranker

| Option | Description | Selected |
|--------|-------------|----------|
| 実装する | 1レース2頭目のベット候補をスコアリング。ゲートが学習して判断 | |
| 実装しない | 1レース1頭のみ選択。分散リスク最小化 | |

**User's choice:** ベストプラクティス（実装する）
**Notes:** ゲートがOOFデータから2頭目の有効性をデータ駆動で評価

---

## Conformal信頼性推定

### Q5: Conformal実装方式

| Option | Description | Selected |
|--------|-------------|----------|
| 既存拡張 | RobustConfidenceEstimatorをWinSelectionGateと連携するよう拡張 | |
| 新規クラス作成 | WinConformalEstimator新規作成。コード重複リスク | |
| ゲート内蔵 | 信頼性推定をWinSelectionGateの内部機能として組み込む | |

**User's choice:** ベストプラクティス（既存拡張）
**Notes:** CP quantileの精度向上（race-condition-dependent calibration）を行う

### Q6: 信頼性→ゲート連携

| Option | Description | Selected |
|--------|-------------|----------|
| EV下限をedgeに | EV_lower_win_correctedをWinSelectionGateの入力edgeとして使用 | |
| 独立軸として追加 | bin軸を4次元(prob/edge/odds/confidence)に拡張 | |

**User's choice:** ベストプラクティス（EV下限をedgeに）
**Notes:** 3次元binningを維持

### Q7: 低信頼レースの扱い

| Option | Description | Selected |
|--------|-------------|----------|
| 閾値で除外 | 信頼区間幅が閾値以上なら完全除外。SELC-02要件に合致 | ✓ |
| 賭け金調整のみ | 低信頼レースは有効だが賭け金を減らす | |

**User's choice:** 閾値で除外 (Recommended)
**Notes:** SELC-02要件「低信頼度レースを除外する」に合致

---

## JRA控除率とエッジ閾値

### Q8: エッジ計算方式

| Option | Description | Selected |
|--------|-------------|----------|
| p差分エッジ | edge = p_model - p_market。p_marketは控除率込み | ✓ |
| 控除率除去後エッジ | fair_odds = odds/(1-0.25)で控除率除去後にエッジ計算 | |

**User's choice:** p差分エッジ (Recommended)
**Notes:** p_market = 1/tanoddslowは既に控除率込みなので、p_modelとの差分が真のエッジ

### Q9: エッジ閾値への控除率反映

| Option | Description | Selected |
|--------|-------------|----------|
| 閾値を25%に引き上げ | GateKeeperのedge閾値を0.03→0.25以上に | |
| レジーム別閾値 | RegimeDetectorのedge_thresholdに控除率を組み込む | ✓ |
| ゲートに学習させる | WinSelectionGateのscore tablesが自動最適化 | |

**User's choice:** レジーム別閾値 (Recommended)
**Notes:** AGGRESSIVE/CONSERVATIVE/COLLAPSED別に控除率を考慮した閾値

### Q10: Kelly計算への反映

| Option | Description | Selected |
|--------|-------------|----------|
| 既存Kelly維持 | 簡易Kelly(edge/(odds-1), cap=25%)を維持 | ✓ |
| 控除率反映Kelly | fair_odds = odds/(1-0.25)で控除率除去後のfair oddsを使う | |

**User's choice:** 既存Kelly維持 (Recommended)
**Notes:** WinSelectionGateがベット可否を判定、Kellyは賭け金計算のみ担当

---

## Claude's Discretion

- RegimeDetectorの具体的なedge_threshold値（控除率考慮後の最適値）
- RobustConfidenceEstimatorのrace-condition-dependent calibrationの詳細実装
- WinSelectionGateのsmoothed scoreのprior_weight等ハイパーパラメータ
- add-second rerankerの閾値グリッドの範囲・粒度

## Deferred Ideas

なし — 議論は全てフェーズスコープ内に留まりました
