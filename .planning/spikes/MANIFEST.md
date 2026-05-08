# Spike Manifest

## Idea

バックテスト実行時間（~3h）を精度を損なわず半減させる。Spike 002で特定したP0/P1改善案を実検証し、GPU/CPU/メモリの効率利用を探索する。

## Requirements

- 精度（ROI, bet_count, HitRate）の劣化を5%未満に抑えること
- 既存テストが全てパスすること
- Windows環境で動作すること
- LightGBMの予測値が最適化前後で一致すること（GPU使用時も）

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | win-payout-missing | standard | "Win payout missing" 警告の根因を3つに分類・定量化できる | VALIDATED | backtest, payouts, data-quality |
| 002 | backtest-perf | standard | バックテスト実行時間のボトルネック6領域を特定し、P0+P1で43-53%短縮を達成する改善案を提示 | VALIDATED | backtest, performance, profiling, pandas |
| 003 | calibration-shorten | standard | キャリブレーションBT期間を4年→6-12ヶ月に短縮してもROI/bet_count差異<5%を維持 | PARTIAL | backtest, calibration, accuracy |
| 004 | gpu-cpu-memory | standard | GPU有効化 + ThreadPoolExecutor最適化 + Categorical化で訓練時間を有意に短縮 | PARTIAL | lightgbm, gpu, threading, categorical |
| 005 | p1-quick-wins | standard | MLflow pip指定 + odds_tsデータ受け渡し + _coerce_types早期returnで~158s短縮 | VALIDATED | mlflow, caching, parquet |
