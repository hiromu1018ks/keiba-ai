# Phase 8: Win Backtest Core - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-04
**Phase:** 8-Win Backtest Core
**Areas discussed:** 単勝決済精度, 候補選択基準, betting-target設計, WF検証の単勝化

---

## 単勝決済精度

### Q1: 単勝ベットの決済方法

| Option | Description | Selected |
|--------|-------------|----------|
| 実際の払戻金 | paytansyopay1/100を使用。JRA公式払戻金、丸め処理含む | ✓ |
| オッズベース | tanoddslow/100を使用。常に利用可能、再現性が高い | |
| ハイブリッド | paytansyopay1優先、欠損時tanoddslowフォールバック | |

**User's choice:** 実際の払戻金 (Recommended)
**Notes:** JRA公式払戻金が最も正確

### Q2: ETL SQLに単勝払戻し列を追加するか

| Option | Description | Selected |
|--------|-------------|----------|
| SQLに追加 | get_payouts() SQLにpaytansyoumaban1, paytansyopay1を追加 | ✓ |
| 変更しない | 既存Parquetに列が既にあるためSQLはそのまま | |

**User's choice:** SQLに追加 (Recommended)
**Notes:** 将来の再ETLに備えた予防措置

### Q3: final_win_odds_mapのオッズ参照ソース

| Option | Description | Selected |
|--------|-------------|----------|
| tanoddslow | 最終オッズ(確定オッズ)。entries.parquet内に存在 | ✓ |
| オッズスナップショット | 複数スナップショットから決済時点オッズを構築 | |

**User's choice:** tanoddslow (Recommended)
**Notes:** JRA公式確定オッズ

### Q4: paytansyopay1欠損時のフォールバック

| Option | Description | Selected |
|--------|-------------|----------|
| tanoddslowフォールバック | stake * tanoddslow/100。WARNINGログ出力 | ✓ |
| ベット無効化 | 決済額=0 | |
| ステーク返済 | 決済額=stake(投資額返済) | |

**User's choice:** tanoddslowフォールバック (Recommended)
**Notes:** 保守的で実用的なフォールバック

---

## 候補選択基準

### Q5: 単勝候補選択の基本フィルタ

| Option | Description | Selected |
|--------|-------------|----------|
| 保守的 | win_gate_pass=True AND conformal_score > 0.5 | |
| バランス型 | win_selection_edge > 0 AND tanoddslow >= 1.0 | ✓ |
| 積極的 | win_selection_ev > 1.0のみ | |

**User's choice:** バランス型 (Recommended)
**Notes:** 精度とベット数のバランスが良い

### Q6: WinSelectionGateとConformalの統合方法

| Option | Description | Selected |
|--------|-------------|----------|
| スコアランキング | win_gate_score降順ソート。win_gate_passはログのみ | ✓ |
| ゲートフィルタ | win_gate_pass=Trueをフィルタに追加 | |
| 3段階フィルタ | 基本フィルタ→conformal足切り→scoreランキング | |

**User's choice:** スコアランキング (Recommended)
**Notes:** ゲート未学習時でも候補選択が機能する設計

### Q7: 1レースあたりの最大候補数

| Option | Description | Selected |
|--------|-------------|----------|
| 1頭/レース | 最も保守的 | |
| 2頭/レース | バランス型。place選択と同じ | ✓ |
| 上限なし | edge>0の全候補にベット | |

**User's choice:** 2頭/レース (Recommended)

---

## betting-target設計

### Q8: --betting-targetのモード設計

| Option | Description | Selected |
|--------|-------------|----------|
| 排他型 | win, place, wideのいずれか1つのみ | ✓ |
| 加法型 | 複数同時指定可能 | |

**User's choice:** 排他型 (Recommended)
**Notes:** シンプルで理解しやすい

### Q9: --betting-targetのデフォルト値

| Option | Description | Selected |
|--------|-------------|----------|
| デフォルト=win | v1.2は単勝マイルストーン | ✓ |
| デフォルト=place | 既存動作を維持 | |

**User's choice:** デフォルト=win (Recommended)

### Q10: ディスパッチ実装方法

| Option | Description | Selected |
|--------|-------------|----------|
| if/else分岐 | BacktestEngine内で条件分岐 | |
| RacePredictor経由 | get_win_candidates()追加、責務分散 | ✓ |
| Orchestrator統合 | ライブ用Orchestratorをbacktestでも使用 | |

**User's choice:** 実装難易度は問わないのでベストプラクティスを追求せよ → RacePredictor経由を採用
**Notes:** ユーザーは一貫して品質優先の方針

---

## WF検証の単勝化

### Q11: run_wf_validation.pyの修正範囲

| Option | Description | Selected |
|--------|-------------|----------|
| 最小修正 | --betting-target追加のみ。~10行変更 | ✓ |
| 拡張診断付き | 単勝特有診断追加(オッズバンド別ROI等)。Phase 9と重複リスク | |

**User's choice:** 最小修正 (Recommended)
**Notes:** Phase 9でレポート機能を実装するため、WF検証は最小限に

### Q12: フォールド定義

| Option | Description | Selected |
|--------|-------------|----------|
| 既存フォールド | 2フォールド(Fold 0: 2020-2023→2024, Fold 1: 2021-2024→2025) | ✓ |
| カスタムフォールド | --folds引数でカスタマイズ | |

**User's choice:** 既存フォールド (Recommended)

---

## Claude's Discretion

- build_win_payout_map()の具体的実装詳細
- get_win_candidates()の返り値の型設計
- select_bets()へのwin path追加方法
- BacktestEngine.run()内のmap構築タイミング
- _settle_bet()のwin対応詳細
- MLflowログフォーマット

## Deferred Ideas

None — discussion stayed within phase scope
