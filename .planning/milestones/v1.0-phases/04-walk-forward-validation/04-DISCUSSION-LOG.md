# Phase 4: Walk-Forward Validation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-03
**Phase:** 4-Walk-Forward Validation
**Areas discussed:** Walk-forwardウィンドウ設計, 過学習検出基準, 加重平均ROI計算方法, 結果レポート・PASS/FAIL判定

---

## Walk-forwardウィンドウ設計

| Option | Description | Selected |
|--------|-------------|----------|
| Expanding window | 学習期間が拡張される方式。データを捨てない標準的アプローチ | ✓ |
| Rolling window | 常に直近N年だけ学習。古いデータを捨てる | |

**User's choice:** Expanding window (Recommended)
**Notes:** データ量を最大化する標準的アプローチ

| Option | Description | Selected |
|--------|-------------|----------|
| train=4, test=1 | Phase 1-3と同一条件 | ✓ |
| train=3, test=1 | 最近データに集中 | |

**User's choice:** train=4, test=1
**Notes:** 既存パイプラインの運用実績と同一条件

| Option | Description | Selected |
|--------|-------------|----------|
| 2フォールド (2024, 2025) | Success Criteriaに対応、実行時間~2時間 | ✓ |
| 5フォールド (2021-2025) | 統計的信頼性が高いが実行時間~5時間 | |

**User's choice:** 2フォールド (2024, 2025)
**Notes:** Success Criteriaの「2024-2025」に対応

| Option | Description | Selected |
|--------|-------------|----------|
| 既存WalkForwardCV利用 | バックテスト実行は--years対応済み | |
| WalkForwardCV拡張 | 過学習分析・加重平均ROIを統合 | |
| ベストプラクティス追求 | 実装難易度問わず最善を選択 | ✓ |

**User's choice:** ベストプラクティス追求 → WalkForwardCV拡張
**Notes:** 品質優先方針に沿って拡張アプローチを採用

| Option | Description | Selected |
|--------|-------------|----------|
| 既存BacktestResult | total_roi等をそのまま参照 | |
| 新規データクラス | WFValidationResult作成 | |
| ベストプラクティス追求 | 最善の方法を選択 | ✓ |

**User's choice:** ベストプラクティス追求 → 新規データクラス
**Notes:** train ROI / test ROI / gap / 過学習スコアを明示的に保持する新データクラス

| Option | Description | Selected |
|--------|-------------|----------|
| train期間もバックテスト | 過学習検出に必須、実行時間倍 | ✓ |
| OOFメトリクスで代用 | 高速だがROI比較にならない | |

**User's choice:** train期間もバックテスト
**Notes:** 過学習検出にtrain ROIとtest ROIの正確な比較が不可欠

| Option | Description | Selected |
|--------|-------------|----------|
| 新スクリプト run_wf_validation.py | 単一責任、run_backtest.pyと独立 | |
| run_backtest.py拡張 | --wf-validationフラグ追加 | |
| ベストプラクティス追求 | 最善の方法を選択 | ✓ |

**User's choice:** ベストプラクティス追求 → 新スクリプト
**Notes:** 単一責任の原則に従い独立エントリポイント

---

## 過学習検出基準

| Option | Description | Selected |
|--------|-------------|----------|
| ROI gap閾値 | シンプルで解釈しやすい | |
| 統計的有意性検定 | t検定、統計的厳密性 | |
| 複合判定 | ROI gap + 全年度ROI > 100% | |
| ベストプラクティス追求 | 最善の方法を選択 | ✓ |

**User's choice:** ベストプラクティス追求 → 複合判定アプローチ
**Notes:** ROI gap + 両年度一貫性 + feature importance安定性の3観点で総合評価

| Option | Description | Selected |
|--------|-------------|----------|
| 20%ポイント閾値 | train-test gapが20%以上で過学習兆候 | |
| 15%ポイント閾値 | より厳しい判定 | |
| 結果を見て調整 | 初回20%で実行し結果次第で厳密化 | ✓ |

**User's choice:** 結果を見て調整
**Notes:** 初回20%ポイントで実行、結果を見て調整

| Option | Description | Selected |
|--------|-------------|----------|
| 3観点の総合評価 | ROI gap + 年度別ROI + feature importance比較 | ✓ |
| ROI gapのみ | シンプルだがfeature importance変化は検出不可 | |

**User's choice:** 3観点の総合評価
**Notes:** feature importanceの年度間比較で特定特徴量への過度な依存を検出

---

## 加重平均ROI計算方法

| Option | Description | Selected |
|--------|-------------|----------|
| ベット数加重 | ベット数が多い年度ほど信頼性が高いことを反映 | |
| 単純平均 | 年度間で平等 | |
| プールROI (総払戻/総投資) | 全ベットをプールしたROI。最も直接的 | |
| ベストプラクティス追求 | 最善の方法を選択 | ✓ |

**User's choice:** ベストプラクティス追求 → プールROIを主要指標、ベット数加重ROIを併記
**Notes:** 金融バックテストで最も誠実な指標はプールROI。参考指標としてベット数加重も併記

---

## 結果レポート・PASS/FAIL判定

| Option | Description | Selected |
|--------|-------------|----------|
| JSON + MLflow + 自動判定 | 年度別ROI、過学習スコア、feature importance比較を含む | |
| JSONのみ (手動確認) | MLflow記録なし | |
| JSON + HTMLレポート | グラフ付き、実装コスト高 | |
| ベストプラクティス追求 | 最善の方法を選択 | ✓ |

**User's choice:** ベストプラクティス追求 → JSON + MLflow + 自動判定
**Notes:** JSON形式 + MLflow記録 + 自動PASS/FAIL判定

| Option | Description | Selected |
|--------|-------------|----------|
| 3基準の自動PASS/FAIL | 各年度ROI確認、過学習兆候評価、加重平均ROI>100% | ✓ |
| 判定なし (結果のみ) | ユーザーが手動判断 | |

**User's choice:** 3基準の自動PASS/FAIL
**Notes:** 3つのSuccess Criteriaを機械的にチェック

---

## Claude's Discretion

- ROI gap閾値の初期値と調整ロジックの詳細（初回20%で実行）
- WFValidationResultデータクラスのフィールド設計
- Feature importance安定性評価の具体的計算方法（Spearman順位相関等）
- MLflow記録のメトリクス名とパラメータ
- PASS/FAIL判定のロジック詳細（WARNING/FAILの区分）
- JSONレポートのスキーマ設計
- 既存WalkForwardCV拡張 vs ラッパークラスの選択

## Deferred Ideas

None — discussion stayed within phase scope
