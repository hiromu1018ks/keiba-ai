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
| LightGBM train() 呼び出し回数 | ~48回 (サーフェス毎23 + グローバル2) |
| 総ブーストラウンド | 11,400+ rounds (両サーフェス合計、OOF含むと更に増) |
| 学習データ行数 | ~216,000行 (2020-2023, 馬単位) |
| 特徴量モジュール数 | 10モジュール |
| サーフェス並列化 | ThreadPoolExecutor(max_workers=2) |

> **注意**: train() 呼び出し回数は内訳が AbilityModel.train_oof() の OOF fold 数に依存。
> OOF 3fold + final = 4回/サーフェス分を含むため、表面よりも多い。

### 推定タイムブレークダウン

| フェーズ | 推定割合 | ボトルネック要因 |
|----------|---------|-----------------|
| HorseHistoryFeatures.compute() | 30-50% | iterrows() による行単位処理 (外側+内側の二重ループ) |
| 共有特徴量計算 (odds_dynamics等) | 15-25% | groupby().apply() + polyfit |
| LightGBM モデル学習 (~48回) | 20-35% | CPU、early stoppingなし、num_threads競合あり |
| Wide pair構築 + その他 | 5-10% | itertools.combinations ループ |

### 主要問題

1. **早期停止なし** — 固定ラウンド数で過学習気味に最後まで実行。多くのモデルは収束後も無駄なラウンドを消費
2. **HorseHistoryFeatures の iterrows()** — 外側ループ(行単位) + 内側ループ(距離bin z-score) の二重ループ
3. **num_threads 競合** — 2サーフェス並列 × 各8スレッド = 16スレッドでハイパースレッド競合が発生
4. **プロファイリング不在** — サブステップ単位の計測が一切ない

### GPUについての判断

RTX 3060 12GB は利用可能だが、**LightGBM GPU学習は今回採用しない**:

- LightGBM の OpenCL バックエンドは、216K行程度のデータセットではホスト↔デバイス転送のオーバーヘッドが計算利点を上回る
- ベンチマーク結果: binary 3.4x遅延、lambdarank 2.4x遅延、regression_l1 7.5x遅延
- i5-12600K のCPU性能が十分に高く、CPU版の方が高速

将来的にデータサイズが大幅に増加(数百万行)した場合はGPU再評価の余地あり。

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

### Phase 1: 早期停止 + スレッド最適化

**目的**: モデル学習フェーズを効率化。

#### 1a: 早期停止の追加

**変更内容**:
- `lgb.train()` 呼び出しに `callbacks=[lgb.early_stopping(stopping_rounds=50)]` を追加
- `num_boost_round` は現状維持 (上限)。早期停止が自動的に最適ラウンドで停止
- **predict() 呼び出しの修正**: 早期停止モデルは `bst.best_iteration` で最適ラウンドが記録される。全ての `predict()` 呼び出しで `num_iteration=bst.best_iteration` を渡す必要がある。これを怠ると固定ラウンド数で予測が変わってしまう

**バリデーションセット戦略** (モデル毎に最適化):

| モデル | バリデーション戦略 | 理由 |
|-------|-------------------|------|
| MarketModel | ランダム20%抽出 | 十分なデータ量 |
| AbilityModel (final) | OOF平均損失を監視 + 最終モデルはランダム20% | OOFの検証セットは別用途 |
| AbilityModel (OOF fold) | **早期停止なし** (現状維持) | fold内データが少ない。OOFの一貫性を優先 |
| PlaceAbilityModel | sklearn内部CVを流用 | sklearn APIが管理 |
| Win/Place/Wide hit | ランダム20%抽出 | データ十分 |
| Win return | ランダム20%抽出 | ~15Kサンプルで十分 |
| Place return | ランダム20%抽出 | ~30Kサンプルで十分 |
| Wide return | ランダム20%抽出 | ペアデータは大量 |
| EVCorrection P | ランダム20%抽出 | init_score付きだが標準的なearly stoppingで対応 |
| EVCorrection E | ランダム20%抽出 | 同上 |
| RaceQualityScreener | 時系列最終20% | レース品質は時系列性あり |
| RegimeDetector | 時系列最終20% | 同上 |

**精度への影響**:
- 早期停止は予測値を変更する可能性があるが、これは**汎化性能の向上**を目的とする
- OOF fold モデルには早期停止を適用しないことで、OOF生成の一貫性を維持
- 各Phaseで予測精度の変化を検証し、悪化がある場合は当該モデルを早期停止対象外にする

#### 1b: num_threads 最適化

**現状の問題**:
- `num_threads = max(1, (os.cpu_count() or 4) // 2)` = 8
- 2サーフェスを `ThreadPoolExecutor(max_workers=2)` で並列実行
- 各8スレッド × 2 = 16スレッド → ハイパースレッド競合

**変更内容**:
- `num_threads` を並列数に応じて動的調整: `max(1, cpu_count // (parallel_workers + 1))`
- 2サーフェス並列時: `16 // 3 ≈ 5` スレッド/モデル
- 単体実行時は現状のまま

**期待効果**: モデル学習フェーズ (全体の20-35%) を **1.5-3倍高速化** (無駄ラウンド削減 + スレッド競合解消)

### Phase 2: 特徴量計算のベクトル化

**目的**: 特徴量計算フェーズ (全体の45-75%) を大幅に高速化。

#### 2a: HorseHistoryFeatures のベクトル化

**現状の問題**: 2つの `iterrows()` ループが存在:
1. **外側ループ** (行267): 各出走馬に対して searchsorted + 辞書検索 + tail(3) + loc
2. **内側ループ** (行320-337): 距離bin毎の z-score 計算 (外側ループ内のネスト)

**変更戦略**:

1. **外側ループのベクトル化 — searchsorted を一括化**:
   - `horses['kettonum']` の一意値毎に、`np.searchsorted()` をベクトル呼び出し
   - 馬ID毎に過去レースのインデックス範囲を一括取得
   - `iterrows()` → `groupby('kettonum')` ベースのベクトル処理

2. **tail(3) のベクトル化 — cumcount ベースフィルタ**:
   - 過去レースを `(kettonum, race_date)` でソート
   - `groupby('kettonum').cumcount(ascending=False)` で逆順ランク付与
   - `rank < 3` で直近3走をベクトルフィルタ

3. **内側ループのベクトル化 — 距離bin z-score**:
   - 距離bin毎の z-score を `groupby('distance_bin').transform(lambda x: (x - x.mean()) / x.std())` で一括計算
   - 行320-337のネストされた iterrows を完全に排除

4. **集約計算のベクトル化**:
   - 過去3走の着順平均、タイムz-score平均等を `groupby().agg()` で一括計算
   - norm_finish_logit は numpy 列演算で全行同時に計算

**期待効果**: iterrows() の 100K+行ループ → ベクトル化で数秒〜数十秒

#### 2b: compute_odds_dynamics のベクトル化

**現状**: 4つの `groupby().apply()` 呼び出し (_get_mid_odds, _calc_velocity, _calc_volatility, _get_mid_ninki)

**変更戦略**:

1. **オッズ速度 (velocity)**:
   - 一次回帰係数を中間統計量で計算
   - `groupby(['race_id', 'umaban']).agg(count=('tanodds', 'count'), sum_x=..., sum_xy=..., sum_x2=...)` で集約
   - 集約結果に対して係数公式をベクトル適用

2. **ボラティリティ**:
   - `groupby(['race_id', 'umaban'])['odds'].std()` でベクトル化
   - `groupby(['race_id', 'umaban'])['odds'].last()` で最新オッズ

3. **mid_odds / 最終オッズ**:
   - `groupby().last()` や `groupby().nth(-1)` に置換
   - `groupby().first()` で初期オッズ

**期待効果**: groupby().apply() の数万回呼び出し → groupby().agg() で数秒

#### 2c: WideJointPairBuilder の最適化

**現状**: `itertools.combinations` をレース毎にPythonループで実行

**変更戦略**:
- 全出走馬DataFrameの自己結合: `pd.merge(horses, horses, on='race_id', suffixes=('_1', '_2'))`
- `query('umaban_1 < umaban_2')` でペアをフィルタ
- 全レースを一括で処理 (ループなし)

**期待効果**: Pythonループ → pandas merge で大幅高速化

### Phase 3: 並列化の強化 (必要に応じて)

**判断基準**: Phase 1-2 完了後に30分目標を達成していれば不要。

#### 3a: モデル間並列

- Win/Place/Wide の各TwoStageModelは互いに独立 (AbilityModelの結果のみに依存)
- AbilityModel完了後、3つのパイプラインを ThreadPoolExecutor で並列実行可能
- この際、各モデルの `num_threads` を適切に配分 (例: 3並列なら `16 // 4 = 4`スレッド/モデル)

#### 3b: ProcessPoolExecutor への切替

- Phase 2のベクトル化が完了すれば、特徴量計算の大部分はnumpyレベルで動作
- GIL問題は小さくなると予想されるが、残存ボトルネックがある場合のみ実施

**実装方針**: Phase 1-2の効果を計測してから判断

## 効果予測

| Phase | 対象 | 推定削減率 | 累積推定時間 |
|-------|------|-----------|-------------|
| 現状 | - | - | ~120分 |
| Phase 0 | プロファイリング | - (計測のみ) | ~120分 |
| Phase 1 | 早期停止 + スレッド最適化 | モデル学習 1.5-3x | ~90-105分 |
| Phase 2 | 特徴量ベクトル化 | 特徴量計算 3-5x | ~20-35分 |
| Phase 3 | 並列化 (必要に応じて) | 追加削減 | ~15-25分 |

**最終予測**: 2時間+ → **15-35分** (30分目標を達成)

## 前提・制約

- **精度完全維持**: モデルのハイパーパラメータ、データ、特徴量定義は変更しない
- **早期停止の検証**: 予測値が変わる可能性を認識し、各Phaseで精度変化を確認
- **OOF一貫性**: OOF foldモデルには早期停止を適用せず、OOF生成の再現性を維持
- **CPU最適化**: GPU学習は採用せず、CPU性能を最大化する方針
- **predict()の修正必須**: 早期停止モデルの全predict()で `num_iteration=bst.best_iteration` を使用

## 検証方法

1. Phase 0 実装後、現状のタイムブレークダウンを記録 (ベースライン)
2. Phase 1 実装後、早期停止 + スレッド最適化の効果を計測
3. Phase 2 実装後、特徴量ベクトル化の効果を計測
4. **各フェーズで予測精度 (回収率) の変化がないことを確認**:
   - Phase 1: 早期停止前後で同一テスト期間の予測値を比較。差分が閾値(例: 回収率 ±2%)以内であることを確認
   - Phase 2: 特徴量の数値が同一であることをユニットテストで検証
5. 最終的に30分以内で完了することを確認

## 影響範囲

### 変更ファイル (推定)

| ファイル | Phase | 変更内容 |
|---------|-------|---------|
| `src/models/training_pipeline.py` | 0, 1, 3 | タイミング計測、早期停止、num_threads動的調整、並列化 |
| `src/features/feature_engine.py` | 0 | タイミング計測 |
| `src/features/horse_history_features.py` | 0, 2 | タイミング計測、外側+内側iterrowsベクトル化 |
| `src/features/odds_dynamics_features.py` | 0, 2 | タイミング計測、apply→aggベクトル化 |
| `src/features/market_bias_features.py` | 0, 2 | タイミング計測、apply置換 |
| `src/models/wide_pair_builder.py` | 0, 2 | タイミング計測、merge最適化 |
| `src/models/stage1_ability_model.py` | 0, 1 | タイミング計測、早期停止(OOF除く) |
| `src/models/market_model.py` | 1 | 早期停止 + predict修正 |
| `src/models/place_ability_model.py` | 1 | 早期停止 |
| `src/models/win_two_stage_model.py` | 1 | 早期停止 + predict修正 |
| `src/models/place_two_stage_model.py` | 1 | 早期停止 + predict修正 |
| `src/models/wide_two_stage_model.py` | 1 | 早期停止 + predict修正 |
| `src/models/ev_correction_model.py` | 1 | 早期停止 + predict修正 |
| `src/models/race_quality_screener.py` | 1 | 早期停止 + predict修正 |
| `src/models/regime_detector.py` | 1 | 早期停止 + predict修正 |

### テスト方針

- 既存テストは全て mock ベース。実装変更に対して mock の期待値の更新が必要
- 高速化後の予測値が同一であることは、既存テストのパスで担保
- Phase 0 の計測値は目視確認 (ログ出力)
- Phase 1 の早期停止: ベースラインとの予測値diffを確認するテストケースを追加
