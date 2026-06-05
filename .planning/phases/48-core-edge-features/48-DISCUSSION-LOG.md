# Phase 48: Core Edge Features - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-04
**Phase:** 48-Core Edge Features
**Areas discussed:** データマージ戦略, 外科的ルーティング, 特徴量モジュール構成, 正規化・ビニング手法

---

## データマージ戦略

| Option | Description | Selected |
|--------|-------------|----------|
| build_all()で早期マージ | 全特徴量モジュールが利用可能。horse_features.parquetキャッシュに含まれる | |
| _train_submodel()で遅延マージ | 既存パターンと一致。horse_features.parquetには含まれない | |
| ハイブリッド: 生値は早期 + 交互作用は遅延 | 生値はbuild_all()でマージ+キャッシュ保存。交互作用はkyakusitu可用性に従い_train_submodel()で計算 | ✓ |

**User's choice:** ハイブリッド: 生値は早期 + 交互作用は遅延
**Notes:** build_all()でDataRepository(store).load_track_conditions()を呼び出しrace_dfに左結合。horse_features.parquetに生値がキャッシュされ高速化。交互作用は_train_submodel()内でHorseHistoryFeatures後に計算。

---

## 外科的ルーティング

| Option | Description | Selected |
|--------|-------------|----------|
| MarketModel除外ルート | AbilityModel + Win/Place/Wide 2-Stage + EV Correctionに登録。MarketModel/RaceQuality/RegimeDetector除外 | |
| 全モデル一律登録 | シンプルだがPhase 36のMarketModel支配リスクあり | |
| 特徴量ごとに個別ルーティング | 精密だが複雑 | |

**User's choice:** MarketModel除外ルート (自由入力で明示)
**Notes:** 「トラック条件特徴量は外科的ルーティングする。AbilityModel、Win/Place/Wide TwoStage、EVCorrection系には登録するが、MarketModel、RaceQuality、RegimeDetectorには登録しない。Phase 36で強特徴量の全モデル一律登録によりMarketModelが支配されたため、馬場状態特徴量は市場残差モデルから除外する。」

---

## 特徴量モジュール構成

| Option | Description | Selected |
|--------|-------------|----------|
| 新規 track_condition_features.py (推奨) | 専用モジュールに集約。T3/T4拡張も同一モジュール | |
| 既存 interaction_features.py の拡張 | モジュール数増加なし。正規化ロジックが混在 | |
| 分割: T1は既存 + T2は新規 | 責務分担は明確だが関連特徴量が分散 | |

**User's choice:** 新規 track_condition_features.py (自由入力で明示)
**Notes:** 「トラック条件特徴量は新規 src/features/track_condition_features.py に実装する。既存 interaction_features.py は汎用交互作用のまま維持し、馬場連続値に固有の正規化・ビニング・surface別処理・将来のT3/T4拡張は専用モジュールに集約する。公開関数は compute_track_condition_features(df) とし、TRACK_CONDITION_COLS を定義する。」

---

## 正規化・ビニング手法

### cushion track relative/zscore

| Option | Description | Selected |
|--------|-------------|----------|
| 学習データベース z-score (推奨) | train期間のtrackcd別mean/std。ルックアヘッド回避 | |
| 生値 + trackcd カテゴリ | シンプル。コース間差の正規化はモデルに委ねる | |
| 全データベース z-score | テスト時に未来情報を一部含む。WF検証で懸念 | |

**User's choice:** 学習データベース z-score (自由入力で明示)
**Notes:** 「turf_cushion_track_relative / turf_cushion_track_zscore は学習データ期間だけで trackcd 別 mean/std を算出し、検証・テスト期間へ適用する。全期間統計は使わない。std が0またはNaNの場合は zscore をNaNにする。relative は turf_cushion - track_mean とする。」

### sire_x_cushion_band

| Option | Description | Selected |
|--------|-------------|----------|
| 5等分ビン [0,5,7,9,11,inf] | シンプル・解釈性高い。極柔/柔/標準/硬/極硬 | |
| 分位数ベース 5ビン | データ密度に応じたビン。foldごとに境界が変わる | |
| 3ビン (柔/標準/硬) | カテゴリcardinality低いが情報量低下 | |

**User's choice:** 固定5段階ビン (自由入力で明示)
**Notes:** 「5段階の固定ビンにする。ただし境界は実データ範囲に合わせて [0, 7, 8, 9, 10, inf] とし、labels=["very_soft","soft","standard","firm","very_firm"] にする。分位数ビンはfoldごとに意味が変わるため避ける。3ビンは安定するがT2-03要件の5段階を満たさないため採用しない。」元の [0,5] は実データでほぼ空になりやすいため調整。

---

## Claude's Discretion

- テスト構成・テストケースの詳細設計
- TRACK_CONDITION_COLSの具体的な列名定義
- track_mean/track_stdの保存形式
- surface-aware計算の実装詳細
- build_all()へのstore経由DataRepository呼び出しパターン
- ログフォーマット・進捗表示

## Deferred Ideas

なし — 議論は全てPhase 48のスコープ内にとどまった
