# Phase 22: 統合検証とバックテスト - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-09
**Phase:** 22-統合検証とバックテスト
**Areas discussed:** テスト年度・学習設定, ベースライン比較と評価指標, 検証スコープとWF検証, 不達時の対応とテスト戦略

---

## テスト年度・学習設定

### テスト年度

| Option | Description | Selected |
|--------|-------------|----------|
| 2024単年 | 学習2020-2023、テスト2024。~41分。v1.4ベースラインと直接比較可能 | ✓ |
| 2024+2025の2年 | マルチ年度モード。最新2年で汎用性確認。~80分 | |
| 2023+2024+2025の3年 | 3年で最も厳格な検証。~120分 | |

**User's choice:** 2024単年
**Notes:** v1.4のROI 83.1%は2024テスト結果なので直接比較可能

### Strategy manifest

| Option | Description | Selected |
|--------|-------------|----------|
| 既存manifest使用 | Phase 17最適化済みパラメータ。キャリブレーションBT有効~57分 | |
| manifestなし | デフォルトパラメータ。~41分。純粋なモデル改善効果を測定 | |
| 再最適化 | Optuna再実行。~2.5h/trial | |

**User's choice:** ユーザー修正済みの既存manifestを使用（自由入力）
**Notes:** ユーザー指摘: "strategyはキャリブレーションと関係ないはずだ。変更したぞ。" manifestと--calibration-btは独立した概念

### Betting mode

| Option | Description | Selected |
|--------|-------------|----------|
| flat固定 (推奨) | 100円固定。Kelly影響除外で純粋なモデル改善を測定 | |
| kelly可変 | Fractional Kelly。実際の運用に近いが改善要因が複合的 | |

**User's choice:** "最適を選んでくれ。ただ、キャリブレーションのオプションがついてないぞ。ちゃんと調べろよ"
**Notes:** flat固定を採用。--calibration-btフラグの追加指摘を受けて構成を修正

### BT構成確定

| Option | Description | Selected |
|--------|-------------|----------|
| この構成で確定 | flat + calibration-bt + report。~57分 | ✓ |
| flat + kellyの両方 | 2パターン比較。~114分 | |

**User's choice:** この構成で確定

---

## ベースライン比較と評価指標

### 比較指標

| Option | Description | Selected |
|--------|-------------|----------|
| 主要指標のみ | ROI + 高オッズ帯ROI(20+)のみ。Success Criteriaに直接対応 | |
| 包括的セグメント分析 | ROI/EV過大評価/レジーム別/オッズバンド別/的中率/平均オッズ | ✓ |

**User's choice:** 包括的セグメント分析

### ベースライン取得方法

| Option | Description | Selected |
|--------|-------------|----------|
| v1.4を再実行 | ~41分追加で同一条件比較 | |
| 既存数値を使用 | ROI 83.1%、EV過大評価2.42倍。追加時間なし | ✓ |

**User's choice:** 既存数値を使用

### レポート出力

| Option | Description | Selected |
|--------|-------------|----------|
| 既存レポート機構 | --reportフラグで自動生成。BacktestReportGeneratorがセグメント別内訳を含む | ✓ |
| 追加の差分比較スクリプト | v1.4差分テーブルや改善要因分解を自動生成するスクリプトを作成 | |

**User's choice:** 既存レポート機構

---

## 検証スコープとWF検証

### WF検証

| Option | Description | Selected |
|--------|-------------|----------|
| スキップ | Phase 22はバックテスト単発に集中。WF検証は別セッション | ✓ |
| WF検証も実行 | run_wf_validation.pyで過学習検出。~4時間追加 | |

**User's choice:** スキップ

### EV診断

| Option | Description | Selected |
|--------|-------------|----------|
| 既存診断のみ | ECE/Brier/Reliability/CQRカバレッジ自動計算 | ✓ |
| TODO指標も実装 | validation_suite.pyのlogloss/spearman_rhoを実装 | |

**User's choice:** 既存診断のみ

---

## 不達時の対応とテスト戦略

### ROI不達時

| Option | Description | Selected |
|--------|-------------|----------|
| 現状で完了 | 分析レポート出力してv1.5完了。改善は次マイルストーン | ✓ |
| 反復改善 | セグメント分析から改善方策を特定し追加フェーズで対応 | |
| Optuna再実行 | 16次元パラメータ再最適化。~2.5h/trial | |

**User's choice:** 現状で完了

### テストカバレッジ

| Option | Description | Selected |
|--------|-------------|----------|
| 既存テストのみ | 1,393テスト通過確認。Phase 19/20/21単体テスト48個含む | ✓ |
| 統合E2Eテスト追加 | Isotonic→CQR→バックテストのフルチェーンmockテスト | |

**User's choice:** 既存テストのみ

---

## Claude's Discretion

- バックテスト実行の具体的な手順（スクリプト実行順序、結果の検証方法）
- レポート結果の解釈とサマリの提示方法
- セグメント別分析の具体的な出力形式

## Deferred Ideas

None — discussion stayed within phase scope
