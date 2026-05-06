# Phase 15: EV Filter Enhancement - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-06
**Phase:** 15-EV Filter Enhancement
**Areas discussed:** 動的閾値の計算方式, EV診断の深度, Conformal再キャリブレーション

---

## 動的閾値の計算方式

### 閾値のアルゴリズム

| Option | Description | Selected |
|--------|-------------|----------|
| Percentile方式 | OOF winnersのEV分布から分位点を計算。シンプルで解釈しやすい | |
| ROC最適探索方式 | 各カットオフでバックテストROIを計算し最大化点を探索 | |
| 複合方式 | Percentile(25th)初期値 + Phase 17 Optuna 15次元最適化 | ✓ |

**User's choice:** "難易度は問わないのでベストプラクティスを追求"
**Notes:** 複合方式を推奨し確定。Percentileで統計的に根拠のある初期値を設定し、Optunaでデータ駆動最適化。

### 閾値の適用粒度

| Option | Description | Selected |
|--------|-------------|----------|
| Global (1閾値) | 全レースに1つの閾値 | |
| Surface別 (2閾値) | 芝/ダート別に計算。Phase 14 D-03パターン | ✓ |
| Regime別 (2閾値) | AGGRESSIVE/CONSERVATIVEで異なる閾値 | |

**User's choice:** "ベストプラクティスを追求"
**Notes:** Surface別を推奨し確定。芝/ダートのEV分布差異を反映。Optuna 16次元。

### NaNフォールバック

| Option | Description | Selected |
|--------|-------------|----------|
| NaN → 通す | fillna(1.0)の既存方針踏襲 | |
| NaN → 除外 | 厳格な対応 | |
| NaN → デフォルト閾値フォールバック | サーフェス別デフォルト閾値で判定 | ✓ |

**User's choice:** "ベストプラクティスを追求"
**Notes:** サーフェス別フォールバックを推奨し確定。Conformal未学習レースでも一貫した動作。

---

## EV診断の深度

### 診断の深度

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 14パターン踏襲 | EV予測vs実際払戻の相関/RMSE + オッズバンド別 | |
| 深度診断 (学術的) | Phase 14 + ECE + Brier score分解 + Reliability diagram + 時系列追跡 | ✓ |
| 段階的 | Phase 15:基本、Phase 18:深掘り | |

**User's choice:** "ベストプラクティスを追求"
**Notes:** 深度診断を推奨し確定。学術的精度評価でEV推定の過大/過小評価を完全に定量化。

### 診断の実行方法

| Option | Description | Selected |
|--------|-------------|----------|
| パイプライン統合 | run_backtest.py --ensembleで自動実行 | ✓ |
| 独立スクリプト | scripts/run_ev_diagnostics.py | |

**User's choice:** パイプライン統合 (Phase 14踏襲)
**Notes:** Phase 14のdrift_diagnostics.pyパターンに準拠。

---

## Conformal再キャリブレーション

### 再キャリブレーション要否

| Option | Description | Selected |
|--------|-------------|----------|
| 再キャリブレーションする | アンサンブルOOF残差でRobustConfidenceEstimatorを再calibrate | ✓ |
| スコープ外として後回し | 閾値の動的化だけで対応 | |

**User's choice:** 再キャリブレーションする (推奨)
**Notes:** EVF要件外だがEV_lower精度に直結。二段階根本解決(Conformal精度改善 + 動的閾値)。

---

## Claude's Discretion

- EV診断モジュールのJSONスキーマ設計
- Percentile計算の実装詳細(正のエッジ勝利馬の定義)
- Brier score分解の実装方法
- Reliability diagramのビン数と表示形式
- 時系列ドリフト追跡の粒度
- サーフェス別フォールバック閾値の計算方法
- テスト戦略(モックベース)

## Deferred Ideas

None — discussion stayed within phase scope
