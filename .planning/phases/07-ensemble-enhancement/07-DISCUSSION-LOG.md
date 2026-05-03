# Phase 7: Ensemble Enhancement - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-03
**Phase:** 07-Ensemble Enhancement
**Areas discussed:** ハイパーパラメータ差別化, Early stopping適用範囲, 特徴量サブセット分割, 多様性検証手法

---

## ハイパーパラメータ差別化

| Option | Description | Selected |
|--------|-------------|----------|
| 固定分散 | 各モデルに異なる固定パラメータを設定。シンプルで再現性が高い | |
| Optuna個別最適化 | Optunaで各モデルのハイパーパラメータを個別最適化 | |
| 固定→Optuna フォールバック | 固定分散ベースで実装し、不十分な場合のみOptuna | |

**User's choice:** ベストプラクティスを追求、実装難易度は問わない
**Notes:** ユーザーは一貫して品質・精度最優先の姿勢。Optuna個別最適化 + 探索空間分離で決定

| Option (探索空間) | Description | Selected |
|-------------------|-------------|----------|
| 探索空間分離 | 各モデルに異なるパラメータ範囲を設定して多様性を強制 | ✓ |
| 共通探索空間 | 全モデルで同じ探索空間を共有 | |
| Claude discretion | Claudeに判断を任せる | |

**User's choice:** ベストプラクティスを追求
**Notes:** 探索空間分離で決定

| Option (実装箇所) | Description | Selected |
|-------------------|-------------|----------|
| StackedEnsemble内完結 | StackedEnsembleクラス内にチューニングを完結 | ✓ |
| OptunaTuner拡張 | 既存OptunaTunerに拡張して統合 | |

**User's choice:** ベストプラクティスを追求
**Notes:** StackedEnsemble内完結で決定

---

## Early Stopping適用範囲

| Option | Description | Selected |
|--------|-------------|----------|
| 全フェーズ適用 | K-fold OOF内 + finalモデルの両方でearly stopping | ✓ |
| Finalのみ | finalモデルのみearly stopping | |
| Claude discretion | Claudeに判断を任せる | |

**User's choice:** ベストプラクティスを追求
**Notes:** 全フェーズ適用で決定

| Option (validation確保) | Description | Selected |
|--------------------------|-------------|----------|
| OOF内 80/20分割 | 各foldで学習データを80/20に分割してvalidation確保 | ✓ |
| Claude discretion | Claudeに判断を任せる | |

**User's choice:** ベストプラクティスを追求
**Notes:** OOF内80/20分割で決定

---

## 特徴量サブセット分割

| Option | Description | Selected |
|--------|-------------|----------|
| 固定異なる比率 | 各モデルのfeature_fractionを異なる固定値に設定 | |
| Optuna最適化 | Optunaチューニングでfeature_fractionも最適化 | ✓ |
| 意味的グループ分け | 時系列/オッズ/展開等のグループに分割して割り当て | |

**User's choice:** ベストプラクティスを追求
**Notes:** Optuna最適化でHPと統合して決定

---

## 多様性検証手法

| Option | Description | Selected |
|--------|-------------|----------|
| OOF相関チェック + 警告 | ペアワイズ相関を計算し閾値超過で警告 | |
| OOF相関 + Importance相関 | 予測相関 + feature importanceのSpearman順位相関の多角評価 | ✓ |
| Claude discretion | Claudeに判断を任せる | |

**User's choice:** ベストプラクティスを追求
**Notes:** OOF相関 + Importance相関の多角評価で決定

---

## Claude's Discretion

- Optunaの試行回数(n_trials)、タイムアウト設定
- 各モデルの具体的な探索空間パラメータ範囲(lr, depth, leaves, rounds)
- OOF内80/20分割の実装詳細
- feature_fractionの探索範囲とステップ
- Ridgeメタラーナーのalpha値
- 多様性評価のログフォーマット
- Optunaのobjective関数設計

## Deferred Ideas

None — discussion stayed within phase scope
