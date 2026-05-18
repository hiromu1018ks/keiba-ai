# Phase 34: Validation and Manifest Update - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-18
**Phase:** 34-Validation and Manifest Update
**Areas discussed:** バックテスト設定, ベースラインIC比較方法, 検証失敗時の対応

---

## バックテスト設定

### 対象年度

| Option | Description | Selected |
|--------|-------------|----------|
| 2023/2024/2025 (v1.6同等) | 3年分、v1.6との直接比較可能 | |
| 2024/2025のみ | 2年分、実行時間短縮 | |
| 2025のみ | 最速、比較は2024年のみ | |

**User's choice:** 2024年のみ（自由入力：「２０２４だけにしよう」）
**Notes:** ユーザーが明確に2024年単年を指定。v1.6の2024年結果との比較が可能

### Strategy Manifest

| Option | Description | Selected |
|--------|-------------|----------|
| manifestなしでBT (Recommended) | デフォルトパラメータ、~41分/年、v1.6と同じ条件 | ✓ |
| 先にOptuna最適化 | ~2.5h/trial追加、ROI最大化 | |

**User's choice:** manifestなしでBT (Recommended)

### Betting Mode

| Option | Description | Selected |
|--------|-------------|----------|
| flat (100円固定) (Recommended) | v1.6と同じ条件、純粋なモデル改善測定 | ✓ |
| kelly | Fractional Kelly、未最適化パラメータ | |

**User's choice:** flat (100円固定) (Recommended)

### Calibration BT

| Option | Description | Selected |
|--------|-------------|----------|
| スキップ (Recommended) | 実行時間短縮、影響小 | ✓ |
| 実行 | ~16分追加、OddsBandFilter再キャリブレーション | |

**User's choice:** スキップ (Recommended)

---

## ベースラインIC比較方法

### ベースラインIC未記録時の対応

| Option | Description | Selected |
|--------|-------------|----------|
| v1.7 IC値のみ記録 (Recommended) | 将来のベースラインとして機能 | ✓ |
| v1.6相当モデルを別学習して比較 | フェアな比較、追加学習時間必要 | |
| IC比較なし（ROIのみ） | IC評価をPhase 34スコープ外に | |

**User's choice:** v1.7 IC値のみ記録 (Recommended)
**Notes:** v1.6学習時にOOF save hookが存在しなかったためベースラインICが未記録

### IC評価の実行タイミング

| Option | Description | Selected |
|--------|-------------|----------|
| BT後にIC評価 (Recommended) | BT保存OOFを使用、自然なフロー | ✓ |
| BTとは別に学習してIC評価 | 追加学習時間必要 | |

**User's choice:** BT後にIC評価 (Recommended)

### IC値のサーフェス別記録

| Option | Description | Selected |
|--------|-------------|----------|
| turf/dirt/all全て記録 (Recommended) | 詳細なベースライン | ✓ |
| allのみ | 簡潔だが芝・ダート別の改善追跡不可 | |

**User's choice:** turf/dirt/all全て記録 (Recommended)

---

## 検証失敗時の対応

### 目標未達時の対応方針

| Option | Description | Selected |
|--------|-------------|----------|
| 結果記録してmanifest凍結 (Recommended) | Phase 34スコープを守る、次マイルストーンで改善 | ✓ |
| Phase 34内でOptuna再最適化 | ~2.5h追加 | |
| 不要特徴量を削除して再BT | 時間かかるが品質保証 | |

**User's choice:** 結果記録してmanifest凍結 (Recommended)

### GPD判定方法

| Option | Description | Selected |
|--------|-------------|----------|
| GPDレポート実行 + 人間判定 (Recommended) | 異常あればレポートに記録 | |
| GPDは参考情報のみ | 成功/失敗判定基準なし | |

**User's choice:** Claudeが判定（自由入力：「claudeが判定して」）
**Notes:** ユーザーがClaude判定を希望。MDR > 0 and FAD <= 5を基準として採用

### GPD成功基準

| Option | Description | Selected |
|--------|-------------|----------|
| MDR > 0 and FAD <= 5 (Recommended) | Marketがshallow優位、Fundamentalがdepth 5以下で活性化 | ✓ |
| Claudeの判断で基準設定 | FEATURE_CATEGORY_MAPに基づく分析 | |

**User's choice:** MDR > 0 and FAD <= 5 (Recommended)

---

## Claude's Discretion

- BT結果のROI値とv1.6(85.7%)の比較解釈
- IC値の「良い/悪い」の判定（ベースラインがないため）
- GPD結果のMDR/FAD値に基づくPASS/WARN判定
- 各検証ステップ間のエラーハンドリング
- 検証結果レポートの形式

## Deferred Ideas

- Optuna戦略パラメータ最適化 — 将来フェーズで実行
- 2023/2025年度の追加BT — Phase 34最小スコープは2024年のみ
