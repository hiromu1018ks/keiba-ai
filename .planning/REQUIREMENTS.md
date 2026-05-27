# Requirements: keiba-ai v2.0 Investment Pipeline Restructuring

**Defined:** 2026-05-27
**Core Value:** 単勝モデルのバックテストROIを100%超えにすること

## v2.0 Requirements

### OOF Health (Phase 37)

- [ ] **OOF-01**: OOF成果物(予測/選択)の空保存を禁止し、fail-fastで異常終了させる
- [ ] **OOF-02**: race_id単位でtrain/validの重複を検査し、混入があれば停止する
- [ ] **OOF-03**: OOF top1 hit rate > 35% または top1 ROI > 200% を異常として停止する
- [ ] **OOF-04**: OOF行数が期待行数の70%未満の場合に停止する
- [ ] **OOF-05**: fold数 < 3 の場合に停止する
- [ ] **OOF-06**: 同一race_idの複数fold混入を検査する
- [ ] **OOF-07**: OOFにis_oof=Trueとfold列を必須化する
- [ ] **OOF-08**: health manifestを行数、レース数、fold数、fold別race_id一意性、top1 hit rate/ROI、日付範囲、source model hash、artifact versionを含めて生成する

### InvestmentFeatureFrame (Phase 38)

- [ ] **IFF-01**: モデル確率特徴量5列(p_win_pred, p_win_corrected, p_win_final, p_win_oof, win_selection_prob)を生成する。学習用FeatureFrameではin-sampleのp_win_predを使用禁止とし、p_win_oof / p_win_final_oofのみ使用する。p_win_predは推論用のみ
- [ ] **IFF-02**: 市場確率特徴量6列(p_market_win_raw, p_market_win_norm, logit_p_market, market_rank, market_share, overround_proxy)を生成する
- [ ] **IFF-03**: モデル対市場ギャップ特徴量7列(p_diff, logit_diff, p_ratio, rank_diff, rank_vs_popularity, market_value_ratio, market_residual)を生成する
- [ ] **IFF-04**: レース内相対特徴量12列(model/market prob rank, ev/edge rank, gap_to_top, gap_to_runner_up, field_size, entropy model/market, top1/top3 concentration)を生成する
- [ ] **IFF-05**: オッズ帯特徴量7列(odds_band, log_odds, odds_rank, is_favorite, is_longshot, favorite_odds, favorite_gap)を生成する
- [ ] **IFF-06**: late odds特徴量9列(drop_rate_60_10/30_10, velocity, volatility, acceleration, popularity_change_30_10, late_odds_drop_z, rank_change, market_share_change)を生成する
- [ ] **IFF-07**: 能力/フォーム要約特徴量9列(p_ability_win, rel_p_ability_win_zscore/rank, form_trend, form_consistency, weighted_recent_form_finish/time, days_since_last_race, class_move)を生成する
- [ ] **IFF-08**: コース/ペース要約特徴量8列(surface, distance_bin, track_condition_code, pace_scenario_fit, pace_pressure, pace_aptitude, closing_speed_ratio_avg, harontime_last3f_avg)を生成する
- [ ] **IFF-09**: 不確実性特徴量6列(prob_calibration_bin, odds_calibration_bin, prediction_entropy, model_market_disagreement, conformal_width, ev_uncertainty_proxy)を生成する
- [ ] **IFF-10**: 学習用FeatureFrameはOOF-safe列のみから構築し、当該レースのPOST_RACEデータを一切使用しない
- [ ] **IFF-11**: stable schemaとしてfeature_version, source_artifact_versionを含め、POST_RACE列を含めない
- [ ] **IFF-12**: 欠損特徴量は明示的なindicatorで扱い、暗黙に削除しない

### MarketAwareWinCalibrator + Segment Calibration (Phase 39)

- [ ] **MAW-01**: Benter型logitブレンド(logit(p_model) + logit(p_market) + 残差特徴量)を実装し、β係数のfloorを0.20以上に設定する
- [ ] **MAW-02**: OOF予測(p_win_oof)のみで学習し、in-sample予測(p_win_pred)を使用しない
- [ ] **MAW-03**: 配備判定をBrier/logloss/ECEの確率品質で行い、ROI単独では配備しない
- [ ] **MAW-04**: surface/odds帯/prob rankのsegment conditioningを特徴量として組み込み、post-hoc確率乗数として扱わない(WSC代替)
- [ ] **MAW-05**: LogisticRegression(主)とLightGBM binary(対照)の2モデルを比較する
- [ ] **MAW-06**: Optunaで正則化強度をチューニングする
- [ ] **MAW-07**: 最もシンプルなモデル(LogisticRegression)を優先し、LightGBMはBrier/logloss/ECE改善かつ年別actual/predicted悪化なしの場合のみ配備する
- [ ] **MAW-08**: 年別(surface別)actual/predicted比率の最大乖離が既存より改善することを確認する
- [ ] **MAW-09**: 既存のWinBenterGateとWinSegmentCalibratorはフォールバックとして保持する。ただしfeature flagで排他的に切り替え、MarketAwareWinCalibrator配備時に旧補正を追加のpost-hocとして同時適用しない(二重補正禁止)

### Race-Level Ranker (Phase 40)

- [ ] **RLR-01**: LightGBM LambdaRank(objective=lambdarank)をrace_id group強制付きで実装する
- [ ] **RLR-02**: 単勝1頭選択の既存動作を維持し、bet countを減らさない
- [ ] **RLR-03**: 勝率ranker: is_winを目的変数とする1段目rankerを実装する
- [ ] **RLR-04**: value ranker: calibrated_evとCLVを目的変数とする2段目rankerを実装する
- [ ] **RLR-05**: investment_score = calibrated_log_ev + value_ranker_score + market_mispricing_score - uncertainty_penalty の統合スコアを実装する。CLVはv2.0では学習ラベル補助・診断に限定し、推論時特徴量には使用しない
- [ ] **RLR-06**: rankerのラベルに純粋なROIラベルを使用せず、win relevance / calibrated EV / value diagnosticsに分離する

### Cross-Cutting Requirements

- [ ] **XCT-01**: 全新規calibrator/rankerコンポーネントは配備前にshadow modeをサポートする(本番推論で新旧両方の結果を出力し、ベットは旧パイプラインのまま)
- [ ] **XCT-02**: baseline選択(既存パイプライン)をfeature flagで切り替え可能にする
- [ ] **XCT-03**: 配備が1レース1頭baselineのbet countを減らす場合、明示的な承認なしには配備しない
- [ ] **XCT-04**: 検証はOOF、walk-forward 2024/2025、artifact再現性チェックを含む。2024/2025は固定検証foldとして扱い、係数・閾値の直接最適化には使用しない
- [ ] **XCT-05**: 同一入力artifactから決定的な予測を生成することを保証する
- [ ] **XCT-06**: backtest reportはbaseline vs新パイプラインを確率品質、選定馬変更、surface/odds/prob-rank帯、CLV、ROI、hit rate、drawdownで比較する
- [ ] **XCT-07**: 実行時間を測定し、再利用可能frameをキャッシュし、不要な全面再計算を導入しない
- [ ] **XCT-08**: 全永続化model/artifactにversion、schema hash、source OOF manifest path、train日付範囲、deployment gate結果を含める

## Deferred to v2.1+

### Portfolio & Multi-Market

- **PF-01**: Race-Level Portfolio (複数頭購入、制約付き配分)
- **PF-02**: Win/Place/Wide統合セレクタ
- **PF-03**: 複数券種横断選択

### Advanced Features

- **ADV-01**: CLV予測モデル (ranker補助用)
- **ADV-02**: OOF drift detector (継続監視)
- **ADV-03**: Reliability diagram可視化
- **ADV-04**: Feature importance安定性監査

## Out of Scope

| Feature | Reason |
|---------|--------|
| 新規外部依存の追加 | 既存LightGBM 4.6.0 + sklearn 1.8.0 + betacal 1.1.0で全Phase対応可能 |
| 複勝/ワイドパイプライン変更 | v2.0は単勝パイプライン構造改革に集中 |
| LSTM/Transformer時系列モデリング | 過去5-15走では過学習リスク高 |
| ROI直接最適化(Optuna ROI最大化) | 確率品質で配備判定。ROIは結果指標 |
| レジーム検出依存 | レジームに依存しない構造にする |
| ベット数削減によるROI嵩上げ | 同等以上のレースカバー率を維持 |
| 2024/2025固有の係数ハードコード | OOF + walk-forward + 年別安定性で配備判定 |
| 新データ源の導入 | 既存EveryDB2データで十分 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| OOF-01 | Phase 37 | Pending |
| OOF-02 | Phase 37 | Pending |
| OOF-03 | Phase 37 | Pending |
| OOF-04 | Phase 37 | Pending |
| OOF-05 | Phase 37 | Pending |
| OOF-06 | Phase 37 | Pending |
| OOF-07 | Phase 37 | Pending |
| OOF-08 | Phase 37 | Pending |
| IFF-01 | Phase 38 | Pending |
| IFF-02 | Phase 38 | Pending |
| IFF-03 | Phase 38 | Pending |
| IFF-04 | Phase 38 | Pending |
| IFF-05 | Phase 38 | Pending |
| IFF-06 | Phase 38 | Pending |
| IFF-07 | Phase 38 | Pending |
| IFF-08 | Phase 38 | Pending |
| IFF-09 | Phase 38 | Pending |
| IFF-10 | Phase 38 | Pending |
| IFF-11 | Phase 38 | Pending |
| IFF-12 | Phase 38 | Pending |
| MAW-01 | Phase 39 | Pending |
| MAW-02 | Phase 39 | Pending |
| MAW-03 | Phase 39 | Pending |
| MAW-04 | Phase 39 | Pending |
| MAW-05 | Phase 39 | Pending |
| MAW-06 | Phase 39 | Pending |
| MAW-07 | Phase 39 | Pending |
| MAW-08 | Phase 39 | Pending |
| MAW-09 | Phase 39 | Pending |
| RLR-01 | Phase 40 | Pending |
| RLR-02 | Phase 40 | Pending |
| RLR-03 | Phase 40 | Pending |
| RLR-04 | Phase 40 | Pending |
| RLR-05 | Phase 40 | Pending |
| RLR-06 | Phase 40 | Pending |
| XCT-01 | Phase 39, 40 | Pending |
| XCT-02 | Phase 39, 40 | Pending |
| XCT-03 | Phase 40 | Pending |
| XCT-04 | Phase 39, 40 | Pending |
| XCT-05 | All phases | Pending |
| XCT-06 | Phase 40 | Pending |
| XCT-07 | Phase 38, 39, 40 | Pending |
| XCT-08 | All phases | Pending |

**Coverage:**
- v2.0 requirements: 35 total (27 phase-specific + 8 cross-cutting)
- Mapped to phases: 35
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-27*
*Last updated: 2026-05-27 after v2.0 milestone requirements definition*
