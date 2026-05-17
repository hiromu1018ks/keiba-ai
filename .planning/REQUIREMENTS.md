# Requirements: keiba-ai v1.7 Market-Independent Edge Discovery

**Defined:** 2026-05-17
**Core Value:** 単勝モデルのバックテストROIを100%超えにすること

## v1.7 Requirements

### ETL Expansion

- [x] **ETL-01**: EveryDB2から三連複オッズ (n_odds_sanren) をParquetに抽出するETL拡張
- [x] **ETL-02**: EveryDB2から馬連オッズ (n_odds_umaren) をParquetに抽出するETL拡張
- [x] **ETL-03**: EveryDB2から三連単オッズ (n_odds_sanrentan) をParquetに抽出するETL拡張
- [x] **ETL-04**: ETL抽出データのカバレッジ検証 (2015-2025、欠損率確認)

### Residual IC Evaluation

- [ ] **RIC-01**: B差分IC (Spearman(model - market, y)) の計算機能を実装
- [ ] **RIC-02**: C直交IC (Spearman(orthog(model|market), y)) の計算機能を実装 — OLS残差で市場成分を除去
- [ ] **RIC-03**: E Incremental IC (IC(model, y) - IC(market, y)) の計算機能を実装
- [ ] **RIC-04**: Per-race IC (各レース内Spearmanの平均) の計算機能を実装
- [ ] **RIC-05**: 現行モデル (v1.6) のベースラインIC値を計算・記録
- [ ] **RIC-06**: 4定式化バッテリーの相互チェック機能 (方向一致性の自動検証)

### Gain per Depth Diagnostic

- [ ] **GPD-01**: LightGBM trees_to_dataframe() でdepth別gain寄与率を集計する機能
- [ ] **GPD-02**: Market/Fundamental/Categorical 3分類でdepth別シェアを可視化する機能
- [ ] **GPD-03**: StackedEnsemble内LightGBMモデルへのアクセスと分析機能
- [ ] **GPD-04**: 暗黙的Two-Stage構造 (上位depth=Market, 下位depth=Fundamental) の検証

### Race-Level Aggregation Features

- [ ] **RLF-01**: rl_log_odds_entropy — インプライド確率のシャノンエントロピー (レース難易度)
- [ ] **RLF-02**: rl_odds_dispersion — オッズ標準偏差 (レース分散)
- [ ] **RLF-03**: rl_top3_odds_gap — 1番人気と3番人気のオッズ差 (混戦度)
- [ ] **RLF-04**: rl_top1_odds — 1番人気オッズのrace-levelブロードキャスト (鉄板度)
- [ ] **RLF-05**: rl_favorite_rank_gap — 1番人気と2番人気の順位差 (支配度)
- [ ] **RLF-06**: rl_n_horses — 出走頭数 (フィールドサイズ)
- [ ] **RLF-07**: build_all() と build_features() の両方にrace-level特徴量を追加 (train-inference parity)

### Existing Feature Promotion

- [ ] **EFP-01**: implied_prob_hhi をFEATURE_COLSに昇格 (既存計算済み)
- [ ] **EFP-02**: odds_skewness をFEATURE_COLSに昇格 (既存計算済み)
- [ ] **EFP-03**: 昇格特徴量のFEATURE_COLS manifest SHA256更新

### Market Cross-Consistency Features

- [ ] **MCF-01**: Harville公式による理論ワイドオッズ計算機能の実装
- [ ] **MCF-02**: rl_favorite_in_wide_top1 — 1番人気がワイドTOP1組合せに含まれるか (0/1)
- [ ] **MCF-03**: rl_trio_overlap — 三連複1組合せが単勝上位3頭に含まれる馬数 (0-3)
- [ ] **MCF-04**: rl_market_consistency — 1番人気が三連複1組合せに含まれるか (0/1)
- [ ] **MCF-05**: rl_trio_odds_ratio — 実三連複1オッズ / Harville理論三連複オッズ
- [ ] **MCF-06**: rl_wide_harville_ratio — 実ワイドTOP1オッズ / Harville理論ワイドオッズ
- [ ] **MCF-07**: ワイドオッズmergeをbuild_all()に統合 (training/backtest重複排除)

### Validation

- [ ] **VAL-01**: 新特徴量追加後のマルチ年度バックテスト (ROI測定)
- [ ] **VAL-02**: Residual IC改善の確認 (C直交ICのベースライン比)
- [ ] **VAL-03**: Gain per Depthで新特徴量が適切なdepth (3-5) で機能していることを確認
- [ ] **VAL-04**: FEATURE_COLS manifest凍結 + SHA256ハッシュ更新
- [ ] **VAL-05**: POST_RACE情報漏洩テストの再実行 (新特徴量に対する漏洩チェック)

## Future Requirements

### Deferred to v1.8+

- **ETL-05**: 枠連オッズ (n_odds_umatan) のETL拡張
- **MCF-08**: 枠連ベース市場整合性特徴量
- **GPD-05**: 多次元直交IC (win+wide+umaren同時直交化)
- **RLF-08**: レースグレード/条件戦別のentropy重み付け
- **RIC-07**: 時系列ドリフト検出 (ICの時系列変化追跡)

## Out of Scope

| Feature | Reason |
|---------|--------|
| オッズ特徴量の除去 | Echo Chamber脱却 = 追加アプローチ。除去はC直交ICを悪化させる (実証済み) |
| Stern/Heneryモデル | Harvilleで90%+のシグナルを捕捉。複雑モデルは利益逓減 |
| LSTM/Transformer | 過去5-15走では過学習リスク高 (v1.0からの方針維持) |
| リアルタイムオッズ収集の改善 | 既存機能をそのまま使用 |
| 複勝/ワイドモデルの変更 | 単勝に集中 (v1.0からの方針維持) |
| 自動投票機能 | ペーパートレードまで |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ETL-01 | Phase 29 | Complete |
| ETL-02 | Phase 29 | Complete |
| ETL-03 | Phase 29 | Complete |
| ETL-04 | Phase 29 | Complete |
| RIC-01 | Phase 30 | Pending |
| RIC-02 | Phase 30 | Pending |
| RIC-03 | Phase 30 | Pending |
| RIC-04 | Phase 30 | Pending |
| RIC-05 | Phase 30 | Pending |
| RIC-06 | Phase 30 | Pending |
| RLF-01 | Phase 31 | Pending |
| RLF-02 | Phase 31 | Pending |
| RLF-03 | Phase 31 | Pending |
| RLF-04 | Phase 31 | Pending |
| RLF-05 | Phase 31 | Pending |
| RLF-06 | Phase 31 | Pending |
| RLF-07 | Phase 31 | Pending |
| EFP-01 | Phase 31 | Pending |
| EFP-02 | Phase 31 | Pending |
| EFP-03 | Phase 31 | Pending |
| MCF-01 | Phase 32 | Pending |
| MCF-02 | Phase 32 | Pending |
| MCF-03 | Phase 32 | Pending |
| MCF-04 | Phase 32 | Pending |
| MCF-05 | Phase 32 | Pending |
| MCF-06 | Phase 32 | Pending |
| MCF-07 | Phase 32 | Pending |
| GPD-01 | Phase 33 | Pending |
| GPD-02 | Phase 33 | Pending |
| GPD-03 | Phase 33 | Pending |
| GPD-04 | Phase 33 | Pending |
| VAL-01 | Phase 34 | Pending |
| VAL-02 | Phase 34 | Pending |
| VAL-03 | Phase 34 | Pending |
| VAL-04 | Phase 34 | Pending |
| VAL-05 | Phase 34 | Pending |

**Coverage:**
- v1.7 requirements: 36 total
- Mapped to phases: 36
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-17*
*Last updated: 2026-05-17 after initial definition*
