# Phase 11: Bet Selection Filters - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-04
**Phase:** 11-Bet Selection Filters
**Areas discussed:** EV下限フィルター戦略, OddsBandFilterのバンド特定, フィルター連鎖とベット数ガード

---

## EV下限フィルター戦略

### フィルター実装方式

| Option | Description | Selected |
|--------|-------------|----------|
| ハードフィルター追加 | get_win_candidates() で EV_lower >= 1.0 をハードフィルターとして追加。既存 edge>0 は残し二重フィルターにする | ✓ |
| 既存フィルター置き換え | win_selection_edge > 0 を EV_lower >= 1.0 に置き換える | |
| Claude裁量 | Claudeの判断で最適な方法を選択 | |

**User's choice:** ベストプラクティスを追求 → ハードフィルター追加（二重フィルター構成）

### フィルター適用場所

| Option | Description | Selected |
|--------|-------------|----------|
| 候補選択内 | get_win_candidates() の最初のフィルタリング段階で適用。後続のスコア計算が除外分のみに集中 | ✓ |
| ベット生成内 | select_bets() 内で候補選択後に追加フィルター | |
| Claude裁量 | Claudeの判断で最適な場所を選択 | |

**User's choice:** ベストプラクティスを追求 → 候補選択内で早期適用

### EV_lower 欠損時の動作

| Option | Description | Selected |
|--------|-------------|----------|
| フォールバック: edgeのみ | EV_lower が NaN の場合、edge>0 のみで判定。安全なフォールバック | ✓ |
| 厳格: 全て除外 | EV_lower が NaN のベットは全て除外 | |
| Claude裁量 | Claudeの判断で最適な方法を選択 | |

**User's choice:** ベストプラクティスを追求 → フォールバック（edgeのみ）

### ログ出力

| Option | Description | Selected |
|--------|-------------|----------|
| 除外統計ログ付き | 除外件数・EV_lower < 1.0 の割合をログ出力、レポートにも反映 | ✓ |
| 最小ログ | debug レベルで最低限出力 | |
| Claude裁量 | Claudeの判断で最適な方法を選択 | |

**User's choice:** ベストプラクティスを追求 → 除外統計ログ付き

---

## OddsBandFilter のバンド特定

### バンド特定方法

| Option | Description | Selected |
|--------|-------------|----------|
| 動的解析 | バックテスト実行時にトレーニング期間データから各バンドROIを自動計算。ルックアヘッドバイアスなし | ✓ |
| ハードコード | 事前解析結果から赤字バンドを設定ファイルに固定 | |
| ハイブリッド | 初期値ハードコード + 動的解析オプション | |
| Claude裁量 | Claudeの判断で最適な方法を選択 | |

**User's choice:** ベストプラクティスを追求 → 動的解析

### 配置場所

| Option | Description | Selected |
|--------|-------------|----------|
| 独立クラス + Engine | 新規 OddsBandFilter クラス + BacktestEngine.run() から呼び出し | ✓ |
| Predictor内蔵 | RacePredictor.get_win_candidates() 内に組み込み | |
| Claude裁量 | Claudeの判断で最適な場所を選択 | |

**User's choice:** ベストプラクティスを追求 → 独立クラス + Engine

### 赤字判定条件

| Option | Description | Selected |
|--------|-------------|----------|
| ROI < 100% で除外 | トレーニング期間ROIが100%未満のバンドを除外 | ✓ |
| ROI < 閾値(例:105%)で除外 | より保守的な閾値 | |
| Claude裁量 | バックテスト結果で調整可能 | |

**User's choice:** ベストプラクティスを追求 → ROI < 100%

### ログ出力

| Option | Description | Selected |
|--------|-------------|----------|
| 除外統計ログ付き | 除外バンド名・件数・各バンドROIをログ出力、レポートに反映 | ✓ |
| 最小ログ | debug レベルで最低限出力 | |
| Claude裁量 | Claudeの判断で最適な方法を選択 | |

**User's choice:** ベストプラクティスを追求 → 除外統計ログ付き

---

## フィルター連鎖とベット数ガード

### フィルター適用順序

| Option | Description | Selected |
|--------|-------------|----------|
| レベル順: Race→Candidate→Odds | COLLAPSEDスキップ → EV下限 → OddsBandFilter。レース全体除外を先に実行 | ✓ |
| Claude裁量 | Claudeの判断で最適な順序を選択 | |

**User's choice:** ベストプラクティスを追求 → レベル順

### ベット数ガード

| Option | Description | Selected |
|--------|-------------|----------|
| ログ監視 + WARNING | 残存ベット数 < 1,000/年 で WARNING。自動緩和なし（Phase 13 Optunaで対応） | ✓ |
| 自動緩和 | ベット数不足時に最も影響の小さいフィルターを自動緩和 | |
| Claude裁量 | Claudeの判断で最適な方法を選択 | |

**User's choice:** ベストプラクティスを追求 → ログ監視 + WARNING

### COLLAPSED スキップ実装

| Option | Description | Selected |
|--------|-------------|----------|
| Engine early-return + logging | BacktestEngine.run() レースループ内で regime 検出直後に early-return。get_strategy_params() 拡張 | ✓ |
| Predictor.should_bet()拡張 | RacePredictor.should_bet() に COLLAPSED 判定を追加 | |
| Claude裁量 | Claudeの判断で最適な方法を選択 | |

**User's choice:** ベストプラクティスを追求 → Engine early-return + logging

---

## Claude's Discretion

- EV_lowerフィルターの具体的なpandasフィルター条件の実装
- OddsBandFilterのインターフェース設計（calibrate() + filter() メソッド等）
- バンド境界定義（Phase 9レポートと同じ 1.0-3.0/3.0-10.0/10.0-30.0/30.0+）
- 除外統計ログのフォーマット（INFO レベル、構造化ログ）
- WARNING の出力条件とフォーマット
- レポート拡張の具体的なコード変更
- テスト戦略

## Deferred Ideas

None — discussion stayed within phase scope
