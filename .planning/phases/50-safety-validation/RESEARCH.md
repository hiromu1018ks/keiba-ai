# Phase 50: Safety & Validation — Lightweight Research

**Date:** 2026-06-05
**Scope:** Audit registry拡張パターン、CI test実装、IC評価スクリプト、NaN率測定方法

## 1. Feature Routing Audit Registry 拡張パターン

### 現状 (`src/audit/feature_routing_registry.py`)
- `FORBIDDEN_CALIBRATOR_FEATURES`: 50列 (MAWC派生features)
- `FORBIDDEN_RANKER_FEATURES`: 28列 (RLR relevance/value/derived)
- `CRITICAL_TARGET_MODELS`: MarketModel, RaceQualityScreener (FAIL-fast)
- `ADVISORY_TARGET_MODELS`: EVCorrectionModel, PlaceEVCorrectionModel, ConformalEVModel, RegimeDetector, PlaceAbilityModel, AbilityModel, WinTwoStageModel

### Phase 50で必要な拡張
新トラック条件特徴量自体は「leak risk」ではないため、FORBIDDEN_*リストへの追加は不要。
しかし、以下の検証が必要:
1. **除外4モデル** (MarketModel, RaceQualityScreener, RegimeDetector, ConformalEVModel) に トラック条件特徴量が混入していないこと → 既存auditのCRITICAL/ADVISORYチェックで自動検出可能
2. **Surface-awareデータ検証**: dirt系特徴量が芝行でNaN、turf系特徴量がダート行でNaNになることをCIテストで確認

### 拡張方法
- 既存 `run_feature_audit()` は各モデルのFEATURE_COLSとFORBIDDENを交差チェック
- 新特徴量がFEATURE_COLSに追加されているため、除外モデルに誤って追加されていればWARN/FAILとして自動検出
- 追加のCIテストが必要: `tests/test_track_condition_features.py` に surface-aware NaN検証を追加

## 2. 特徴量カラム完全インベントリ

### T1/T2 (8列) — `TRACK_CONDITION_COLS`
```python
dirt_moisture_x_kyakusitu, turf_cushion_track_relative, turf_cushion_track_zscore,
dirt_moisture_x_barrier_pos, dirt_moisture_high_flag, dirt_moisture_dry_flag,
turf_cushion_x_kyakusitu, sire_x_cushion_band
```

### T3-04 + T4-01 + T4-03 + T4-04 (11列) — `TRACK_DERIVED_COLS`
```python
track_front_bias_score, kickback_risk_score, expected_pace_class,
cushion_season_deviation, moisture_season_deviation,
cushion_anomaly_flag, moisture_extreme_flag,
cushion_x_distance, moisture_x_weight, cushion_x_age, surface_condition_transition
```

### T4-02 (4列) — `RACE_CONDITION_COLS`
```python
race_condition_match_score, race_condition_match_max,
race_condition_match_ratio, race_field_front_bias
```

### T3 aptitude (14列) — `APTITUDE_COLS` (モデルFEATURE_COLSには含まれない)
```python
race_id, kettonum,  # keys
horse_dirt_wet_hit_rate, horse_dirt_dry_hit_rate,
horse_cushion_hard_hit_rate, horse_cushion_soft_hit_rate,  # hit rates
horse_dirt_wet_starts_count, horse_dirt_dry_starts_count,
horse_cushion_hard_starts_count, horse_cushion_soft_starts_count,  # counts
horse_condition_versatility, horse_condition_type,  # classification
prev_dirt_moisture, prev_turf_cushion  # previous conditions (T4-04 input)
```

**合計**: 23列がモデルFEATURE_COLSに直接登録 + 14列が中間データ

### モデルルーティング (Phase 48/49実績)
- **6 included**: AbilityModel, WinTwoStageModel, PlaceTwoStageModel, EVCorrectionModel, PlaceEVCorrectionModel, PlaceAbilityModel (+ WideTwoStageModel)
- **4 excluded**: MarketModel, RaceQualityScreener, RegimeDetector, ConformalEVModel

### 生値マージ (2列)
- `dirt_moisture`, `turf_cushion` — FeatureEngine.build_all() でマージ、FEATURE_COLSには含まれない

## 3. POST_RACE CI検証

### 既存CI test (`tests/test_track_condition_data.py`)
- Phase 47で追加: CSV→Parquet変換、物理レンジ検証、集約ロジックのテスト (16 tests)
- Phase 47 D-11でPOST_RACE CI test追加済み

### Phase 50で必要
- Phase 48/49で追加された特徴量（TRACK_CONDITION_COLS + TRACK_DERIVED_COLS + RACE_CONDITION_COLS）がPOST_RACE_COLSに含まれていないことを確認
- `src/domain/types.py` のPOST_RACE_COLSに新特徴量が誤って追加されていないかCIで担保

### 3層CI検証パターン
1. **whitelist**: FEATURE_COLSに登録された特徴量は安全
2. **forbidden**: POST_RACE_COLS (41列) に含まれるものは使用禁止
3. **manual**: カテゴリ列等の個別確認

## 4. IC評価 (`run_ic_eval.py` + `src/models/ic_evaluator.py`)

### 実行方法
```bash
python scripts/run_ic_eval.py data/oof/oof_predictions.parquet --output data/baseline/ic_baseline.json
```

### 出力
- `run_ic_evaluation()` が B差分IC / C直交IC / E Incremental IC / Per-race IC を計算
- JSON形式で各特徴量のIC値、p値、サンプル数を出力
- `console_summary()` でコンソールにサマリ表示

### Phase 50での評価対象
- 全23列（TRACK_CONDITION_COLS + TRACK_DERIVED_COLS + RACE_CONDITION_COLS）
- 各列について: 単変量IC、C直交IC、欠損率、有効サンプル数
- Tier別(T1/T2/T3/T4)およびhorse-level/race-level別の集計
- カテゴリ列(sire_x_cushion_band)はカテゴリ別ターゲット統計として別評価

### 注意点
- ICは情報提供目的（個別FAILなし）
- abs(C直交IC) >= 0.005をsignal分類
- fold間符号反転・有効サンプル不足を診断対象としてフラグ付け

## 5. BT段階判定 (VLD-01)

### コマンド
```bash
# 一次判定: 2025年単独BT (--ensembleのみ)
python scripts/run_backtest.py --years 2025 --train-window 4 --ensemble

# 二次判定: 2024+2025通算BT (--ensembleのみ)
python scripts/run_backtest.py --years 2024 2025 --train-window 4 --ensemble

# 二次確認: safety filter付き
python scripts/run_backtest.py --years 2024 2025 --train-window 4 --ensemble --min-win-ev 1.03 --min-win-odds 3.0
```

### 成功基準
1. 2025年単独ROI >= 97%
2. 2024+2025通算ROI >= 97% AND 各年ROI >= 90%

### 実行時間
~41分/年 (CLAUDE.mdより)

## 6. WF Fold0 NaN率検証 (VLD-03)

### 対象データ
- WF Fold0: train 2020-2023 / test 2024
- クッション値データ: 2020/09開始 → Fold0学習期間の前半(2020/01-08)はNaN
- 含水率データ: 2018/07開始 → Fold0のダートレースはほぼカバー

### 検証方法
1. FeatureEngine.build_all()でWF Fold0期間(2020-2023)の特徴量を生成
2. Surface-aware NaN率計算: 芝系特徴量は芝レース行のみを分母とする
3. 元データNaN vs 派生処理NaNの原因分離

### 閾値 (D-11~D-13)
- < 30%: PASS
- 30-50%: WARN
- >= 50%: FAIL

### 実装アプローチ
- `data/features/horse_features.parquet` をロードして列ごとにNaN率を計算
- またはBT実行時の特徴量Parquetを利用

## 7. タスク構成案

Phase 50は以下の論理グループに分割可能:

### Wave 1: CI検証 (コスト最小)
- REG-02: Feature Routing Audit拡張 + surface-aware CI test
- REG-03: POST_RACE 3層CI検証 (Phase 48/49特徴量追加)

### Wave 2: データ品質検証 (DB必要・中コスト)
- VLD-03: WF Fold0 NaN率検証 + 原因分離レポート

### Wave 3: BT ROI検証 (最も高コスト ~82-123分)
- VLD-01: 段階BT (一次2025 → 二次2024+2025)

### Wave 4: IC評価 (BT完了後)
- VLD-02: OOF IC評価 + Tier別集計

### 依存関係
- Wave 1→2→3→4 の直列依存
- Wave 1のCI FAIL → 修正後に再実行
- Wave 3の一次FAIL → 診断→1回のみ再試行→PASS/FAIL
