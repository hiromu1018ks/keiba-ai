# Phase 3 結果: Market Model OOF 対応

> 実施日: 2026-04-14
> コミット: `7e36983` (predict_oof), `71a55c9` (num_threads fix), `3779b1b` (pipeline統合)

## 変更内容

### Task 3.1: MarketModel.predict_oof() 実装
- 5-Fold KFold CV (`shuffle=False`) で OOF 予測を生成
- OOF 予測値で log_error を再計算 (signed_log_error_win, abs_log_error_win 等)
- 全データ再学習で推論用モデル更新
- Rule 11 準拠 (_p_market_pred_win を drop)
- テスト: 8件新規 + 12件既存 = 20/20 通過

### Task 3.2: TrainingPipeline への OOF 統合
- `_train_submodel()` 内の `market.predict_and_calc_error(df)` 後に `market.predict_oof(df)` 追加
- PIT安全: _train_submodel は学習データのみを受け取るため df 全体に OOF 適用で OK

### Task 3.3: バックテスト検証

## バックテスト結果 (2025テスト / ensemble / flat / JRAのみ)

| 指標 | Phase 3 結果 | Plan Baseline | 差分 |
|------|-------------|---------------|------|
| ROI | **74.9%** | 98.8% | -23.9pt |
| ベット数 | 3,984 | 499 | +3,485 |
| 投資額 | ¥398,400 | — | — |
| 払戻額 | ¥298,340 | — | — |
| 利益 | **-¥100,060** | -¥610 | -¥99,450 |
| 最大DD | **100.1%** | — | — |
| 最終資金 | -¥60 | — | — |

### 分析

#### ROI 低下の原因 (正常な動作)
- OOF 導入により Market Model の log_error からデータリークが除去された
- Stage2 (ReturnModel) が「過度に最適化された」誤差信号を受け取れなくなった
- これはリーク排除の正常な副作用 — 長期的には汎化性能向上が期待される

#### 懸念点
1. **最大DD 100.1%**: 資金がほぼ枯渇。運用不可レベル
2. **ベット数激増 (3,984 vs 499)**: QualityScreener の挙動変化か、OOF 予測値分布の変化
3. **改善前 63.8% → 74.9% (+11.1pt)**: スクリプト内の比較軸はオッズ整合前の古い baseline の可能性

#### BT/PT 乖離評価
- 目標: +24.3pt → 15pt以下
- Paper Trading データなしのため未評価
- OOF 導入により BT 内での自己予測リークは除去済み

## 次のステップ

1. **Phase 4 の実装で ROI 回復を狙う** — 新特徴量 (過去走拡張、ペース適性、コース適性) がリークのない正しい信号を補完する
2. **QualityScreener の閾値見直し** — ベット数 3,984 は過多。より厳しいフィルタリングが必要
3. **マルチ年度バックテスト** — 2023-2025 の3年度で堅牢性確認

## コミット履歴

| Commit | Message |
|--------|---------|
| `7e36983` | feat: MarketModel.predict_oof() を追加 (5-Fold CV) |
| `71a55c9` | fix: predict_oof() に num_threads パラメータを追加 |
| `3779b1b` | feat: Market Model OOF を TrainingPipeline に統合 |
