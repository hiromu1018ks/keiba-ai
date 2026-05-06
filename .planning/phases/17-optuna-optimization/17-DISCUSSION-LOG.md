# Phase 17: Optuna Optimization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-06
**Phase:** 17-Optuna Optimization
**Areas discussed:** WF fold構成, 計算時間とトライアル数, Multi-seed安定性検証, ベット数制約の調整

---

## WF fold構成

### Walk-Forward foldの実装アプローチ

| Option | Description | Selected |
|--------|-------------|----------|
| 軽量WF: モデル共有 | 学習済みモデル1セットを全foldで共有。テスト期間のみ変更。高速(既存設計の延長) | ✓ |
| 完全WF: fold毎再学習 | fold毎にモデルを再学習。真のWFだが計算コストが膨大(44min×4fold×100trials)。実行不可能 | |
| ハイブリッド: 事前学習済みスナップショット | 事前に4セットのモデルスナップショットを生成。1回のみの再学習コスト(~3h) | |

**User's choice:** 軽量WF: モデル共有 (Recommended)
**Notes:** Phase 13の「独自軽量WFループ」設計を踏襲。戦略パラメータ最適化が目的のため、モデルは固定して戦略パラメータのみ変動させる

### 4foldのテスト期間構成

| Option | Description | Selected |
|--------|-------------|----------|
| 年次4fold | 2022/2023/2024/2025の4年テスト。各fold十分なベット数(年間1000+) | ✓ |
| 半年4fold (OOS) | 2024-H1, 2024-H2, 2025-H1, 2025-H2。完全OOSだがベット数半減 | |
| ハイブリッド4fold | 2023/2024-H1/2024-H2/2025。バランス型 | |

**User's choice:** 年次4fold (Recommended)
**Notes:** 2022-2023は学習期間と重なるが、戦略パラメータのロバスト性評価としては有効。最終OOS検証はPhase 18

---

## 計算時間とトライアル数

### Optunaトライアル数

| Option | Description | Selected |
|--------|-------------|----------|
| 100試行維持 | Phase 13 D-11の決定を維持。TPEで16次元空間を十分探索可能 | ✓ |
| 50試行に削減 | 4fold化による計算量増を相殺。実行時間約半分 | |
| 200試行に増加 | 更に安定した結果。実行時間約4倍 | |

**User's choice:** 100試行維持 (Recommended)
**Notes:** 4fold化でもTPEの効率性で16次元空間を十分探索可能

### モデルロード最適化

| Option | Description | Selected |
|--------|-------------|----------|
| trial内ロード1回 | モデルロードtrial内1回 + training_bet_historyキャッシュ | ✓ (ベストプラクティス) |
| study全体で1回ロード | 最速だがregime_overrides上書きの安全性懸念 | |
| 現状維持 | fold毎ロード。シンプルだが実行時間長い | |

**User's choice:** ベストプラクティスを追求
**Notes:** ユーザーの一貫した方針「ベストプラクティス追求」「実装難易度は問わない」に基づく選択

### MedianPruner設定

| Option | Description | Selected |
|--------|-------------|----------|
| MedianPruner維持 | 4foldの2fold目でpruning | |
| 1fold目で早期pruning | 75%節約だが誤pruningリスク | |
| Claude's discretion | 研究者・プランナーが決定 | ✓ |

**User's choice:** Claude's discretion

---

## Multi-seed安定性検証

### Seed数構成

| Option | Description | Selected |
|--------|-------------|----------|
| 2 seeds追加 | 主実行(seed=42, 100trials) + 安定性確認(seed=43,44, 各50trials) | ✓ |
| 1 seed + 事後解析 | 上位10trial分布分析。最速だがseed間再現性未検証 | |
| 5 seeds (完全検証) | 5回の独立実行。実行時間5倍 | |

**User's choice:** 2 seeds追加 (Recommended)
**Notes:** 実行時間の現実性を考慮。主実行1回 + 安定性確認2回 = 計3seed実行

### 安定性判定基準

| Option | Description | Selected |
|--------|-------------|----------|
| CV閾値0.3 | 3seed間のCV>0.3で不安定判定 | |
| stddev比で評価 | stddev/|default| > 0.5で不安定 | |
| Claude's discretion | 研究者・プランナーが設計 | ✓ |

**User's choice:** Claude's discretion

### 不安定次元の対応

| Option | Description | Selected |
|--------|-------------|----------|
| 固定化して再実行 | 不安定次元をデフォルト値に固定→探索空間縮小→再最適化。CONF-03に相当 | ✓ (ベストプラクティス) |
| 報告のみ | Phase 18で人間が判断。再実行なし | |
| 探索範囲を縮小 | 固定化より柔軟だが複雑 | |

**User's choice:** ベストプラクティスを追求
**Notes:** 固定化して再実行(CONF-03自動縮小に相当)を選択

---

## ベット数制約の調整

### ベット数制約設計

| Option | Description | Selected |
|--------|-------------|----------|
| 現状維持(1000) | min_bets=1000, ハードカットオフ(ROI=-1.0) | ✓ |
| 軟制約(連続ペナルティ) | TPE探索効率向上だが実装複雑 | |
| Claude's discretion | 研究者・プランナーが設計 | |

**User's choice:** 現状維持(1000) (Recommended)
**Notes:** Phase 13 D-09を踏襲。1000件は統計的有意性の妥当な目安

---

## Claude's Discretion

- `_generate_folds()`の具体的な実装(コンストラクタ引数から動的生成)
- MedianPrunerの設定(n_startup_trials, n_warmup_steps等)
- 安定性判定の具体的な手法と閾値(CV, stddev比, rank相関等)
- 安定性レポートのJSONスキーマ
- モデルロード最適化の具体的な実装

## Deferred Ideas

None — discussion stayed within phase scope
