# Spike Manifest

## Idea

run_backtest を WIN モードで実行した際に出る "Win payout missing for {race_id} umaban=X, using odds fallback" 警告の原因調査。

## Requirements

- 警告のノイズと実データ欠損を区別できること
- JRAレースのみを対象とすること（engine.py の jyocd フィルタと整合）

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | win-payout-missing | standard | "Win payout missing" 警告の根因を3つに分類・定量化できる | VALIDATED | backtest, payouts, data-quality |
| 002 | backtest-perf | standard | バックテスト実行時間のボトルネック6領域を特定し、P0+P1で43-53%短縮を達成する改善案を提示 | VALIDATED | backtest, performance, profiling, pandas |
