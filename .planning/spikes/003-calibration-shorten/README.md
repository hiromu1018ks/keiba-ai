---
spike: 003
name: calibration-shorten
type: standard
validates: "Given 4年学習期間のキャリブレーションBT, when 直近6-12ヶ月に短縮, then ROI/bet_count差異<5%で精度保持、実行時間-3000〜3500s"
verdict: PARTIAL
related: [002]
tags: [backtest, calibration, accuracy, odds-band-filter]
---

# Spike 003: キャリブレーションBT期間短縮の精度検証

## What This Validates

Given: キャリブレーションBT（Phase B）が全体の60%（~4900s）を占める
When: キャリブレーション期間を短縮またはスキップ
Then: テスト結果（ROI, bet_count）に有意な差異が生じない（<5%）

## Research

### OddsBandFilter キャリブレーションフロー

```
run_backtest.py (_collect_training_bet_history)
  └→ BacktestEngine(target=win, default_strategy)
       └→ _calibrate_odds_band_filter(training_bet_history=None)
            └→ _generate_training_bet_history()
                 └→ BacktestEngine(target=place) [再帰防止]
                 └→ place-based bet_history を生成
            └→ OddsBandFilter.calibrate(place_bet_history)
                 └→ ROI per band = sum(result)/sum(stake)
                 └→ roi_threshold=1.0 → 全バンド ROI<1.0 → 全バンド除外
       └→ 外部エンジン: 全バンド除外 → 0 bets → 空のbet_history
  └→ 空のbet_history を返す

test_engine.run(..., training_bet_history=[])
  └→ _calibrate_odds_band_filter(training_bet_history=[])
       └→ if training_bet_history: → False (空リスト)
       └→ キャリブレーションスキップ → フィルター無効
  └→ テスト実行: OddsBandFilterによる除外なし
```

### 実測データからの確認

テスト期間（2024年）のbet_history.json（2642件）には全オッズ帯のベットが存在:
- 1.0-3.0: 5件, 3.0-10.0: 140件, 10.0-30.0: 738件, 30.0+: 1759件
- フィルターが有効なら30.0+帯の1759件は除外されているはず → **除外されていない** = フィルター無効を確認

### 期間窓比較（2024-2025年テストデータ）

| Window | Bets | Excluded Bands | Match Full? |
|--------|------|----------------|-------------|
| full (24mo) | 3661 | ALL 4 bands | baseline |
| 12mo | 1019 | 3 bands (10-30除外せず) | DIFF |
| 6mo | 468 | 3 bands (10-30除外せず) | DIFF |
| 3mo | 243 | 3 bands (10-30除外せず) | DIFF |

10.0-30.0バンドがサンプル不足で不安定（月次ROI: 0.0〜6.7と高分散）。

## How to Run

```bash
python .planning/spikes/003-calibration-shorten/verify_calibration_shorten.py
```

## What to Expect

- フル期間で全4バンドがROI<1.0（除外対象）
- テストbet_historyに全オッズ帯が存在 → フィルター実質無効を確認
- 10.0-30.0バンドの月次ROIは高分散（サンプルサイズ不足）

## Investigation Trail

1. **bet_history.json分析**: テスト期間2642件、全オッズ帯にベット存在 → フィルター無効確認
2. **OddsBandFilter.calibrate()コード追跡**: ROI per bandのみ計算、時間加重なし
3. **engine.pyフロー追跡**: calibration chain発見 — place→win→test の3層構造
4. **退化現象特定**: `roi_threshold=1.0` + 競馬の控除率 → 全バンドROI<1.0 → 空bet_history → フィルター無効化
5. **期間窓比較**: 短縮すると10.0-30.0バンドの判定が変わる（サンプル不足）だが、現状のフィルター自体が無効

## Results

**Verdict: PARTIAL ⚠**

### 主要発見

1. **退化現象**: デフォルト戦略（`roi_threshold=1.0`）では、キャリブレーションBTが**全バンドを除外**し、結果的に0件のbet_historyを返す。テストエンジンは空リストを受け取りフィルターを無効化。**Phase Bの4900sは完全に無駄な計算。**

2. **スキップ可能**: デフォルト戦略使用時、キャリブレーションBT全体をスキップしても**テスト結果は完全に同一**（フィルターが無効だから）。これだけで **-4900s（-60%）** の短縮。

3. **Optuna最適化時**: `--strategy-manifest`使用時は異なる`roi_threshold`が設定される可能性。この場合はキャリブレーションが意味を持つ。しかし12ヶ月分のデータで十分（月次ROIは分散が高いが傾向は安定）。

### 推奨アクション

| 優先度 | アクション | 効果 | リスク |
|--------|-----------|------|--------|
| **P0** | `--skip-calibration`フラグ追加 | **-4900s (-60%)** | 無（デフォルト戦略では退化） |
| P1 | `--calibration-months N`パラメータ追加 | -2000〜3000s | 低 |
| P2 | OddsBandFilter のroi_threshold最適化 | 不明 | 中（戦略依存） |

### 期待効果

**現在: ~8239s → Phase Bスキップ: ~3338s（-59%）**

Phase Bをスキップするだけで「半減」目標を達成。他の最適化（Spike 004/005）は追加の短縮。
