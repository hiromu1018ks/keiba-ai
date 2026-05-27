# Requirements: keiba-ai v1.8 Turf Precision Calibration

**Defined:** 2026-05-19
**Core Value:** 単勝モデルのバックテストROIを100%超えにすること

## v1.8 Requirements

### ETL — Data Foundation

- [x] **ETL-01**: HaronTimeL3/L4 (SE table) をfloat64変換してentries Parquetに格納、センチネル値(000/999)をNaN化
- [x] **ETL-02**: LapTime1~25 (RA table) をfloat64変換してraces Parquetに格納、センチネル値(000)をNaN化
- [x] **ETL-03**: Jyuni1c~4c (SE table) コーナー通過順位を数値化してentries Parquetに格納
- [x] **ETL-04**: 全新POST_RACE列を `domain/types.py` の POST_RACE_COLS に登録し、v1.6の3層CI漏洩検出が機能することを確認
- [x] **ETL-05**: HaronTimeL3/L4の相互排他性を検証し、統合ロジック(harontime_last3f)を決定

### HLF — Haron/Lap Feature Computation

- [ ] **HLF-01**: 過走上がり3F/4Fの平均・z-score・トレンド特徴量をPIT-safeに計算 (expanding_stats + race_date < target_date)
- [ ] **HLF-02**: 上がりタイムのレース内相対ランキング (harontime_l3_avg_race_rank) を計算
- [ ] **HLF-03**: LapTime1~25からレースペース特徴量(前半/中盤/後半ペース比)を計算 (過走レースのみ)
- [ ] **HLF-04**: 全HLF特徴量を12モデルのFEATURE_COLSに登録
- [ ] **HLF-05**: TrainingPipeline._train_submodel() と BettingOrchestrator.build_features() の両パスでHLF特徴量が計算されることを確認

### TRF — Turf Relative Features

- [ ] **TRF-01**: form_trend_race_rank, blood_total_wr_race_rank, blood_surface_wr_race_rank を add_race_transforms() に追加
- [ ] **TRF-02**: weighted_recent_form (直近3走重み付き成績) を horse_history_features.py に追加
- [ ] **TRF-03**: 全TRF特徴量を12モデルのFEATURE_COLSに登録

### INT — Conditional Interaction Features

- [ ] **INT-01**: grade_x_form_trend (グレード別調子トレンド) 交互作用を interaction_features.py に追加
- [ ] **INT-02**: distance_x_closing_index (距離別追込力) 交互作用を追加
- [ ] **INT-03**: grade_x_blood_prize_log (グレード別血統賞金) 交互作用を追加
- [ ] **INT-04**: 全INT特徴量を12モデルのFEATURE_COLSに登録

### CAL — Calibration Layers

- [ ] **CAL-01**: 人気帯キャリブレーション (1-3, 4-6, 7-9, 10-12, 13+) のOOF residual ratio スケーリングを ev_correction_model.py に追加
- [ ] **CAL-02**: 人気帯キャリブレーションに拡張ウィンドウOOF計算を適用し、ルックアヘッドバイアスを防止
- [ ] **CAL-03**: EVCorrectionModel.FEATURE_COLS に regime_state, surface_x_popularity, market_entropy_x_surface を追加
- [ ] **CAL-04**: regime_state を RacePredictor → EVCorrectionModel 間で伝播させる仕組みを実装
- [ ] **CAL-05**: レジーム-EVフィードバックループの強制遷移テストを実装

### VAL — Validation & Verification

- [ ] **VAL-01**: v1.6の3層CI漏洩検出テストを全新特徴量(HLF/TRF/INT)に適用し全通過
- [ ] **VAL-02**: 芝Stage1 IC b_difference が正の値に改善されたことを確認
- [ ] **VAL-03**: 芝pop 4-12キャリブレーションratioが0.527から改善されたことを確認
- [ ] **VAL-04**: BT 2024 ROIが100%超えを達成したことを確認
- [ ] **VAL-05**: Turf conservative regime ROIが改善されたことを確認
- [ ] **VAL-06**: Manifest v1.8 凍結 (SHA256特徴量ハッシュ)

## v2 Requirements (Deferred)

### Advanced Haron/Lap Features

- **HLF-06**: コーナー通過順位からの展開特徴量 (逃げ/先行/差し/追込分類)
- **HLF-07**: ペースプロファイル分類 (スロー/ミドル/ハイペース)
- **HLF-08**: 末脚指数 (上がり3F - レース平均上がり)

### E-correction Fundamental Activation

- **EFA-01**: E補正モデルに blood_prize_log × p_ability_win 交互作用を追加
- **EFA-02**: E補正モデルに trainer_wr_venue × class_level_current 交互作用を追加
- **EFA-03**: 芝win_ret fundamental_activation_depth を10→5以下に改善

### Training Data Features

- **TDF-01**: 坂路調教タイム (37-HANRO) のETL・特徴量化
- **TDF-02**: training_pipeline _build_race_level_features() rl_*列処理追加

### Investment Feature Frame (Phase 38)

- **IFF-01**: InvestmentFeatureFrameBuilder.build_frame(df, mode) が9カテゴリ(model_prob, market_prob, model_market_gap, race_relative, odds_band, late_odds, ability_form, course_pace, uncertainty) 90-130列の投資特徴量を生成
- **IFF-02**: train modeはOOF-safe列(p_win_oof等)のみ使用、in-sample列(p_win_pred等)を拒否。infer modeは本番列を使用。両モード同一出力スキーマ
- **IFF-03**: train/infer出力スキーマ同一性: 同一列名・列順・dtype。テストがアサート
- **IFF-04**: InvestmentFeatureSpec frozen dataclassによるスキーマレジストリ。全特徴量にcategory, source columns, train/infer behavior, missing behavior, leakage class, dtype, stable output nameを定義
- **IFF-05**: POST_RACE列非混入。漏洩テスト(VAL-01 scoped to InvestmentFeatureFrame)通過
- **IFF-06**: Parquetキャッシュ + sidecar manifest JSON。決定性: 同一入力+同一builder_version→同一出力
- **IFF-07**: artifact manifest: feature_version, schema_hash, source_artifact_hash, source OOF health manifest path, builder_version, mode, generated_at

### Validation — Investment Frame (Phase 38)

- **VAL-01**: 3層CI漏洩テストをInvestmentFeatureFrameに適用。POST_RACE除外、train mode OOF-safe確認、train/infer同一スキーマ

## Out of Scope

| Feature | Reason |
|---------|--------|
| 複勝/ワイドモデルの変更 | 単勝に集中するため |
| 新データ源の導入 | 既存EveryDB2データで十分 |
| モデル再学習(スタッキング構造変更) | 既存3モデルスタッキングをそのまま使用 |
| LSTM/Transformer時系列モデリング | 過去5-15走では過学習リスク高 |
| 坂路調教タイム(37-HANRO) | 今回はレースハロンタイムに集中 |
| 前処理パイプラインの大規模リファクタ | 既存パターンの拡張で対応可能 |
| WF検証の実行 | PostgreSQL環境依存、別タイミングで実施 |
| Regime-dependent calibration | v2.0全体でスコープ外。過去実験で信頼性未確認 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ETL-01 | Phase 35 | Complete |
| ETL-02 | Phase 35 | Complete |
| ETL-03 | Phase 35 | Complete |
| ETL-04 | Phase 35 | Complete |
| ETL-05 | Phase 35 | Complete |
| HLF-01 | Phase 36 | Pending |
| HLF-02 | Phase 36 | Pending |
| HLF-03 | Phase 36 | Pending |
| HLF-04 | Phase 36 | Pending |
| HLF-05 | Phase 36 | Pending |
| TRF-01 | Phase 36 | Pending |
| TRF-02 | Phase 36 | Pending |
| TRF-03 | Phase 36 | Pending |
| INT-01 | Phase 36 | Pending |
| INT-02 | Phase 36 | Pending |
| INT-03 | Phase 36 | Pending |
| INT-04 | Phase 36 | Pending |
| CAL-01 | Phase 39 | Pending |
| CAL-02 | Phase 39 | Pending |
| CAL-03 | Phase 39 | Pending |
| CAL-04 | Retired | Regime-dependent calibration out of scope for v2.0 |
| CAL-05 | Retired | Regime-dependent calibration out of scope for v2.0 |
| IFF-01 | Phase 38 | Pending |
| IFF-02 | Phase 38 | Pending |
| IFF-03 | Phase 38 | Pending |
| IFF-04 | Phase 38 | Pending |
| IFF-05 | Phase 38 | Pending |
| IFF-06 | Phase 38 | Pending |
| IFF-07 | Phase 38 | Pending |
| VAL-01 | Phase 38 | Pending |
| VAL-02 | Phase 39 | Pending |
| VAL-03 | Phase 39 | Pending |
| VAL-04 | Phase 39 | Pending |
| VAL-05 | Phase 39 | Pending |
| VAL-06 | Retired | Replaced by IFF-07 artifact manifest (v2.0) |

**Coverage:**
- v1.8/v2.0 requirements: 35 total
- Mapped to phases: 35
- Unmapped: 0
- Retired: 3 (CAL-04, CAL-05, VAL-06)

---
*Requirements defined: 2026-05-19*
*Last updated: 2026-05-27 after Phase 38 context — CAL regime retired, IFF requirements added, VAL redistributed*
