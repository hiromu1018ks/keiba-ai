# 学習パイプライン高速化 設計書

**日付**: 2026-04-04
**目標**: run_train.py の実行時間を 2時間+ → 30分以内に短縮 (精度完全維持)

## 現状分析

### PC スペック

| 項目 | 値 |
|------|-----|
| CPU | Intel i5-12600K (10コア/16スレッド) |
| GPU | NVIDIA RTX 3060 12GB VRAM (driver 595.97) |
| RAM | 32GB |
| OS | Windows 11 Pro |

### パイプライン構成

| メトリクス | 値 |
|-----------|-----|
| LightGBM train() 呼び出し回数 | 30回 (サーフェス毎14 + グローバル2) |
| 総ブーストラウンド | 11,400 rounds (両サーフェス合計) |
| 学習データ行数 | ~216,000行 (2020-2023, 馬単位) |
| 特徴量モジュール数 | 10モジュール |
| サーフェス並列化 | ThreadPoolExecutor(max_workers=2) |

### 推定タイムブレークダウン

| フェーズ | 推定割合 | ボトルネック要因 |
|----------|---------|-----------------|
| HorseHistoryFeatures.compute() | 30-50% | iterrows() による行単位処理 |
| 共有特徴量計算 (odds_dynamics等) | 15-25% | groupby().apply() + polyfit |
| LightGBM モデル学習 (30回) | 20-35% | CPUのみ、early stoppingなし |
| Wide pair構築 + その他 | 5-10% | itertools.combinations ループ |

### 主要問題

1. **RTX 3060 が完全に未使用** — LightGBMはCPU版のみ
2. **早期停止なし** — 固定ラウンド数で過学習気味に最後まで実行
3. **HorseHistoryFeatures の iterrows()** — pandas最大のアンチパターン
4. **プロファイリング不在** — サブステップ単位の計測が一切ない

## 設計

### Phase 0: プロファイリング基盤

**目的**: 各フェーズの所要時間を可視化し、効果測定を可能にする。

**変更内容**:
- `_train_submodel()` 内の各ステップにタイミング計測を追加
- `FeatureEngine.build_all()` 内の各モジュールにタイミング計測を追加
- ログ形式: `"[TIMING] {step_name}: {elapsed:.1f}s"`
- `time.perf_counter()` を使用 (高精度)
- 計測結果は `logging.info()` で出力 (本番運用にも残す)

**対象箇所** (合計 ~25計測ポイント):
- `build_all()`: intra_race, odds_dynamics, market_bias, difficulty, bloodline (5)
- `_train_submodel()`: horse_history, interaction, market_model, ability_oof, place_ability, win_hit, win_return, jockey_ctx, trainer_ctx, ev_correction, place_hit, place_return, wide_pair_build, wide_hit, wide_return (15)
- `run()`: race_level_features, quality_screener, regime_detector, mlflow_logging (5)

**期待効果**: なし (計測のみ。実装・検証の基盤)

### Phase 1: LightGBM GPU学習 + 早期停止

**目的**: モデル学習フェーズを大幅に高速化。

#### 1a: GPU学習の有効化

**変更内容**:
- 全 `lgb.train()` 呼び出しの params に以下を追加:
  ```
  "device": "gpu",
  "gpu_platform_id": 0,
  "gpu_device_id": 0,
  "gpu_use_dp": false   # 単精度で高速化
  ```
- `PlaceAbilityModel` (sklearn API) には `LGBMClassifier(**params)` に同様の設定を追加
- GPU検出失敗時のフォールバック: `try/except` でCPUに自動切替

**VRAM使用量見積もり**:
- 学習データ: ~216K行 × 37特徴量 × 8bytes ≈ 64MB
- LightGBM内部バッファ: 推定 200-500MB
- 合計: 1GB未満 (12GB VRAMに対して十分余裕)

**互換性**:
- LightGBM 4.6.0 (現状バージョン) は Windows GPU版バイナリを同梱
- 追加インストール不要。`device="gpu"` を設定するだけで動作

#### 1b: 早期停止の追加

**変更内容**:
- 全 `lgb.train()` 呼び出しに以下を追加:
  - `valid_sets=[train_data, valid_data]` — バリデーションセット
  - `callbacks=[lgb.early_stopping(stopping_rounds=50)]`
- バリデーションセットの作成方法 (モデル毎):
  - OOF済みモデル (AbilityModel): OOF foldの検証セットを流用
  - TwoStageModel hit/return: 学習データの20%をランダム抽出
  - EVCorrectionModel: 同上
  - RaceQualityScreener/RegimeDetector: 時系列ベース最終20%
- `num_boost_round` は現状維持 (上限として機能)。早期停止が自動的に最適ラウンドで停止

**精度への影響**:
- **なし** (早期停止は正則化の一種。過学習を防ぐ)
- むしろ汎化性能が向上する可能性が高い

#### 1c: スレッド設定の最適化

- GPU使用時は `num_threads` を低減 (GPUがメイン計算を担当)
- `num_threads=4` に設定 (データ前処理用)

**期待効果**: モデル学習フェーズ (全体の20-35%) を 5-10倍高速化

### Phase 2: 特徴量計算のベクトル化

**目的**: 特徴量計算フェーズ (全体の45-75%) を大幅に高速化。

#### 2a: HorseHistoryFeatures のベクトル化

**現状**: `iterrows()` で各行に対して以下を処理:
- searchsorted による過去レース検索
- `tail(3)` による直近3走取得
- norm_finish_logit, z-score の計算
- 辞書検索 + DataFrame.loc

**変更戦略**:

1. **事前ソート + merge_asof**:
   - 過去レースを `(horse_id, race_date)` でソート
   - 対象行に対して `pd.merge_asof()` で「レース日以前の最新N走」を一括取得
   - `direction='backward'`, `by='horse_id'` で馬ID毎に検索

2. **groupby + cumcount による tail(3) 相当**:
   - ソート済みデータで `groupby('horse_id').cumcount(ascending=False)` を計算
   - `count < 3` で直近3走をフィルタ (ベクトル化)

3. **集約計算のベクトル化**:
   - 過去3走の着順平均、タイムz-score平均等を `groupby().agg()` で一括計算
   - norm_finish_logit は numpy 列演算で全行同時計算

**期待効果**: iterrows() の 100K+行ループ → ベクトル化で数秒〜数十秒

#### 2b: compute_odds_dynamics のベクトル化

**現状**: `groupby().apply()` で各 (race_id, umaban) グループに polyfit を適用

**変更戦略**:

1. **オッズ速度 (velocity)**:
   - 一次回帰係数 = `(n*Σxy - ΣxΣy) / (n*Σx² - (Σx)²)` を groupby + transform で一括計算
   - `groupby(['race_id', 'umaban'])` の後に `.transform()` で中間統計量を計算

2. **ボラティリティ**:
   - `groupby(['race_id', 'umaban'])['odds'].std()` でベクトル化
   - `groupby(['race_id', 'umaban'])['odds'].transform('last')` で最新オッズ

3. **mid_odds / 最終オッズ**:
   - `groupby().last()` や `groupby().nth(-1)` に置換

**期待効果**: groupby().apply() の数万回呼び出し → ベクトル化で数秒

#### 2c: WideJointPairBuilder の最適化

**現状**: `itertools.combinations` をレース毎にループ実行

**変更戦略**:
- レース毎の出走馬DataFrameに対して `pd.merge(horses, horses, on='race_id')` で自己結合
- `query('umaban_1 < umaban_2')` でペアをフィルタ
- 全レースを一括で処理 (ループなし)

**期待効果**: Pythonループ → pandas merge で大幅高速化

### Phase 3: 並列化の強化 (必要に応じて)

**判断基準**: Phase 1-2 完了後に30分目標を達成していれば不要。

#### 3a: ProcessPoolExecutor への切替

- Phase 2のベクトル化が完了すれば、特徴量計算の大部分はnumpyレベルでGILを解放
- ProcessPool化の必要性は低いと予想される
- ベクトル化後も残るPythonレベルのボトルネックがある場合のみ実施

#### 3b: モデル間並列

- Win/Place/Wide の各TwoStageModelは互いに独立 (AbilityModelの結果のみに依存)
- AbilityModel完了後、3つのパイプラインを ThreadPoolExecutor で並列実行可能
- ただしGPU使用時はGPUリソースの競合に注意 (直列の方が効率的な場合も)

**実装方針**: Phase 1-2の効果を計測してから判断

## 効果予測

| Phase | 対象 | 推定削減率 | 累積推定時間 |
|-------|------|-----------|-------------|
| 現状 | - | - | ~120分 |
| Phase 0 | プロファイリング | - (計測のみ) | ~120分 |
| Phase 1 | GPU + 早期停止 | モデル学習 5-10x | ~85-95分 |
| Phase 2 | 特徴量ベクトル化 | 特徴量計算 3-5x | ~20-35分 |
| Phase 3 | 並列化 (必要に応じて) | 追加削減 | ~15-25分 |

**最終予測**: 2時間+ → **15-35分** (30分目標を達成)

## 前提・制約

- **精度完全維持**: モデルのハイパーパラメータ、データ、特徴量定義は変更しない
- **早期停止は正則化**: 固定ラウンドとの差異は過学習防止としてのみ機能
- **GPU依存**: Phase 1はRTX 3060が前提。フォールバックとしてCPU版も維持
- **Windows環境**: GPU版LightGBMの動作確認が必要

## 検証方法

1. Phase 0 実装後、現状のタイムブレークダウンを記録 (ベースライン)
2. Phase 1 実装後、GPU + 早期停止の効果を計測
3. Phase 2 実装後、特徴量ベクトル化の効果を計測
4. 各フェーズで予測精度 (回収率) の変化がないことを確認
5. 最終的に30分以内で完了することを確認

## 影響範囲

### 変更ファイル (推定)

| ファイル | Phase | 変更内容 |
|---------|-------|---------|
| `src/models/training_pipeline.py` | 0, 1, 3 | タイミング計測、GPU params追加、並列化 |
| `src/features/feature_engine.py` | 0 | タイミング計測 |
| `src/features/horse_history_features.py` | 0, 2 | タイミング計測、ベクトル化 |
| `src/features/odds_dynamics_features.py` | 0, 2 | タイミング計測、ベクトル化 |
| `src/features/market_bias_features.py` | 0, 2 | タイミング計測、apply置換 |
| `src/models/wide_pair_builder.py` | 0, 2 | タイミング計測、merge最適化 |
| `src/models/stage1_ability_model.py` | 0, 1 | タイミング計測、GPU+早期停止 |
| `src/models/market_model.py` | 1 | GPU+早期停止 |
| `src/models/place_ability_model.py` | 1 | GPU+早期停止 |
| `src/models/win_two_stage_model.py` | 1 | GPU+早期停止 |
| `src/models/place_two_stage_model.py` | 1 | GPU+早期停止 |
| `src/models/wide_two_stage_model.py` | 1 | GPU+早期停止 |
| `src/models/ev_correction_model.py` | 1 | GPU+早期停止 |
| `src/models/race_quality_screener.py` | 1 | GPU+早期停止 |
| `src/models/regime_detector.py` | 1 | GPU+早期停止 |

### テスト方針

- 既存テストは全て mock ベース。実装変更に対して mock の期待値の更新が必要
- 高速化後の予測値が同一であることは、既存テストのパスで担保
- Phase 0 の計測値は目視確認 (ログ出力)
