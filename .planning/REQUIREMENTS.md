# Requirements: v2.3 Track Condition Feature Integration

**Milestone:** v2.3 Track Condition Feature Integration
**Goal:** 馬場状態の連続値データ（ダート含水率・芝クッション値）を特徴量として統合し、BT ROI 97%+を回復する
**Created:** 2026-06-04

## v1 Requirements

### Data Pipeline (ETL)

- [ ] **ETL-01**: 外部CSV（ダート含水率 `20180728~20260531ダート含水率.csv`）をParquetに変換するprecomputeスクリプトを実装する。エントリ単位ID → race_id (先頭14桁) でrace-level集約、NaN適正処理、データ型検証を含む
- [ ] **ETL-02**: 外部CSV（芝クッション値 `20200912~20260531クッション値.csv`）をParquetに変換するprecomputeスクリプトを実装する。エントリ単位ID → race_id集約、NaN適正処理、データ型検証を含む
- [ ] **ETL-03**: DataRepositoryに含水率・クッション値Parquetのローダーメソッドを追加する。既存の`load_horse_career_stats()`パターンに従う
- [ ] **ETL-04**: POST_RACE分類の確認と適正処理を行う。含水率/クッション値はレース当日JRA発表値（締切前利用可能情報）として扱い、POST_RACE_COLSに含めないことをCI検証で確認する

### Tier 1 — Maximum Edge (P0)

- [ ] **T1-01**: `dirt_moisture_x_kyakusitu` 特徴量を実装する。ダート含水率と脚質コードの交互作用特徴量。含水率上昇時の逃げ馬有利バイアス（物理メカニズムベースの構造的エッジ）を捕捉する
- [ ] **T1-02**: `turf_cushion_track_relative` / `turf_cushion_track_zscore` 特徴量を実装する。芝クッション値のトラック別相対化。生値のコース間差（函館7.6 vs 京都10.2）を正規化し、同じ「良」馬場内の微細な差を捕捉する

### Tier 2 — High Edge (P1)

- [ ] **T2-01**: `dirt_moisture_x_barrier_pos` / `dirt_moisture_high_flag` / `dirt_moisture_dry_flag` 特徴量を実装する。含水率と枠位置の交互作用 + 高含水(>12%)/低含水(<3%)バイナリフラグ。含水率12%超で内枠有利が逆転する現象を捕捉する
- [ ] **T2-02**: `turf_cushion_x_kyakusitu` 特徴量を実装する。芝クッション値と脚質の交互作用。高クッション(硬)で先行有利、低クッション(柔)で差し/追込有利のバイアスを捕捉する
- [ ] **T2-03**: `sire_x_cushion_band` 特徴量を実装する。種牡馬コード × クッション値ビン(5段階)の交互作用。血統データとの結合が必要

### Tier 3 — Derived & Context (P2)

- [ ] **T3-01**: `horse_dirt_wet_hit_rate` / `horse_dirt_dry_hit_rate` 特徴量を実装する。馬個体のダート含水率適性（高含水時/低含水時の着順上位率）。過走履歴からのPIT-safe計算が必要
- [ ] **T3-02**: `horse_cushion_hard_hit_rate` / `horse_cushion_soft_hit_rate` 特徴量を実装する。馬個体の芝クッション値適性（硬/柔時の着順上位率）。過走履歴からのPIT-safe計算が必要
- [ ] **T3-03**: `horse_condition_type` 特徴量を実装する。馬個体の馬場状態適性カテゴリ（湿得意/乾得意/万能）分類
- [ ] **T3-04**: `cushion_season_deviation` / `moisture_season_deviation` 特徴量を実装する。クッション値・含水率のコース別月別偏差。季節特有のバイアスを捕捉する

### Tier 4 — Higher-Order & Race-Level (P3)

- [ ] **T4-01**: `track_front_bias_score` / `kickback_risk_score` / `expected_pace_class` 特徴量を実装する。含水率/クッション値から算出する先行バイアススコア、蹴り返しリスク、ペース予測
- [ ] **T4-02**: `race_condition_match_score` / `race_field_front_bias` 特徴量を実装する。レースフィールド条件マッチスコア（出走各馬の適性最大値）とフィールド先行バイアス
- [ ] **T4-03**: `cushion_anomaly_flag` / `moisture_extreme_flag` 特徴量を実装する。クッション値/含水率の異常値検出（コース平均から2σ逸脱、コース上位/下位5%）
- [ ] **T4-04**: `cushion_x_distance` / `moisture_x_weight` / `cushion_x_age` / `moisture_x_prev_kyakusitu` / `surface_condition_transition` 特徴量を実装する。既存特徴量とのインタラクション拡張

### Feature Registration & Safety

- [ ] **REG-01**: 新特徴量をFeatureEngineに統合し、FEATURE_COLSに12モデル全登録する。既存の`_register_features()`パターンに従う
- [ ] **REG-02**: Feature Routing Audit対応 — 新特徴量の外科的ルーティング（MarketModel/RaceQualityScreener除外等）を検証する。v1.8のPhase36教訓を適用
- [ ] **REG-03**: 新特徴量のPOST_RACE分類が正しいことを3層CI検証で確認する

### Validation

- [ ] **VLD-01**: マルチ年度BT（2024/2025）で新特徴量のROI寄与を検証する。BT ROI 97%+（v1.7レベル回復）を成功基準とする
- [ ] **VLD-02**: 新特徴量のIC評価（C直交IC）を実行し、既存特徴量と独立したシグナルであることを確認する
- [ ] **VLD-03**: クッション値データのWF Fold0（2020-2023学習）でのデータ可用性制約を検証し、NaN率が許容範囲内であることを確認する

## Future Requirements (Deferred)

- Conservative MAWC redesign / selective interaction experiment — v2.4+
- デプロイゲート自動判定 (DEP-01) — v2.4+
- Optuna 19次元パラメータ最適化 (DEP-02) — v2.4+
- WinSegmentCalibrator dead code removal (WRN-01) — v2.4+

## Out of Scope

| Feature | Reason |
|---------|--------|
| MAWCキャリブレーション修正 | v2.3は新特徴量に集中。MAWC問題はv2.4+で扱う |
| 複勝/ワイドモデルの変更 | Core Value: 単勝に集中 |
| 新データ源の追加（EveryDB2以外） | 含水率/クッション値CSVが既に利用可能 |
| レジーム依存ロジック変更 | 構造的エッジ獲得に集中 |
| モデルアーキテクチャ変更 | 特徴量追加アプローチで対応。スタッキング構造は維持 |
| オッズ特徴量の除去 | C直交IC悪化のリスクあり |

## Traceability

| REQ-ID | Phase | Status |
|--------|-------|--------|
| ETL-01 | Phase 47 | Pending |
| ETL-02 | Phase 47 | Pending |
| ETL-03 | Phase 47 | Pending |
| ETL-04 | Phase 47 | Pending |
| T1-01 | Phase 48 | Pending |
| T1-02 | Phase 48 | Pending |
| T2-01 | Phase 48 | Pending |
| T2-02 | Phase 48 | Pending |
| T2-03 | Phase 48 | Pending |
| T3-01 | Phase 49 | Pending |
| T3-02 | Phase 49 | Pending |
| T3-03 | Phase 49 | Pending |
| T3-04 | Phase 49 | Pending |
| T4-01 | Phase 49 | Pending |
| T4-02 | Phase 49 | Pending |
| T4-03 | Phase 49 | Pending |
| T4-04 | Phase 49 | Pending |
| REG-01 | Phase 48 | Pending |
| REG-02 | Phase 50 | Pending |
| REG-03 | Phase 50 | Pending |
| VLD-01 | Phase 50 | Pending |
| VLD-02 | Phase 50 | Pending |
| VLD-03 | Phase 50 | Pending |

---
*Last updated: 2026-06-04*
