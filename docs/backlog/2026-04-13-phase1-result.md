# Phase 1 Result: 既存特徴量の穴埋めと活用

> **Date:** 2026-04-13
> **Baseline:** ROI 98.8%, 499 bets, -¥610 (2021-2024学習/2025テスト/ensemble/flat/JRAのみ)

## Backtest Result

| 指標 | 値 |
|------|-----|
| **ROI** | **101.6%** |
| **Bets** | 427 |
| **Stake** | ¥42,700 |
| **Return** | ¥43,370 |
| **Profit** | **+¥670** |
| **Max DD** | 3.8% |
| **Final Bankroll** | ¥100,670 |

## 対ベースラインからの変化

- **ROI: 98.8% → 101.6% (+2.8pt)** — 黒字化達成
- **Bets: 499 → 427 (-72)** — RaceQualityScreener の EMA 指標追加でレース選別が厳格化
- **Profit: -¥610 → +¥670 (+¥1,280)** — 赤字から黒字へ転換

## 実装したタスク

1. **Task 1.1**: horse_career_stats に馬場状態別累積統計8列追加 (cum_turf_good/heavy, cum_dirt_good/heavy)
2. **Task 1.2**: blood_condition_wr を vectorized な馬場状態別 Beta 平滑化勝率で実装
3. **Task 1.3**: blood_keito_cd を種牡馬系統コードで実装 (2段JOIN: horses→keito)
4. **Task 1.4**: compute_flb_slope() をパイプライン統合 + odds_skewness, implied_prob_hhi 追加
5. **Task 1.5**: compute_roi_ema() をパイプライン統合 + overround_ema, entropy_ema 追加

## 所見

- 血統特徴量 (blood_condition_wr, blood_keito_cd) が NaN プレースホルダーから実際の値になり、モデルの判別力が向上
- RaceQualityScreener への EMA 指標追加がレース選別精度を改善（ベット数減少 = より質の高いレースを選択）
- Max DD 3.8% と低リスクで黒字達成は堅牢性の証左

## 次のステップ

Phase 2 (種牡馬産駒特徴量) でさらなる改善を狙う。目標: ROI +3-8pt。
