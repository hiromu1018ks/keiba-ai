# コード構造リファレンス

本ドキュメントは各パッケージ・モジュールの責務と主要クラスを解説する。
安定度に応じて記述量を調整している。

## 安定性アノテーション

| アノテーション | 意味 |
|---|---|
| ✅ 安定 | Phase A で実装済み。本番利用前提。破壊的変更は原則行わない |
| 🔧 ほぼ安定 | Phase B-F で実装済み。バグ修正レベルの変更はあるが、設計は固定 |
| 🚧 開発中 | Phase F で実装中。インタフェースが変更される可能性がある |

---

## domain/ ✅

データクラスと型定義。システム全体の基盤となる不変の型体系。

### models.py

全データクラスを定義。frozen dataclass を基本とし、計算プロパティで導出値を提供する。

- **Race**: レース情報。複合PK `(year, month_day, jyo_cd, kaiji, nichiji, race_num)` を持ち、`race_id` (文字列化)、`surface`、`distance_band`、`is_good_track`、`is_soft_track`、`is_steeple`、`grade_name` を計算プロパティとして提供
- **Entry**: 出走馬情報。`finish_pos` から `is_winner`、`is_place`、`is_cancelled` を導出。`running_style_name` で脚質名を返却
- **Bet**: 投票情報。`stake >= 100` の `is_valid` チェックと `profit = result - stake` を持つ
- **OddsSnapshot**: 時系列オッズスナップショット。`race_id`、`happyo_time`、`umaban`、`tan_odds`、`fuku_odds`
- **DDState**: ドローダウン状態。`current_dd`、`rolling_roi`、`n_bets_eval`、`recovery_state`
- **RegimeConfig**: レジーム検知設定。`window=200`、`min_samples=100`、`fav_rate_aggressive=0.28`、`fav_rate_collapsed=0.18`、`overround_base=0.20`、`retrain_trigger=100`
- **TwoStageConfig**: 2段階モデルのハイパーパラメータ。Stage A (hit: AUC, 31 leaves, lr=0.03, 500 rounds) と Stage B (return: MAE, 15 leaves, lr=0.03, 300 rounds)
- **SubmodelSet**: サブモデルのコンテナ。`market`、`stage1`、`win`、`ev_corrector`、`place`、`wide`、`confidence` を保持。TrainingPipeline が surface ごとに生成
- **TrainedModelsV5**: 学習済みモデルのトップレベルコンテナ。`submodels: dict[str, SubmodelSet]`、`quality_screener`、`regime_detector` を保持
- **SafetyConfig**: 安全ガード設定。`min_bankroll=10000`、`max_daily_loss=10000`、`max_weekly_loss=30000`、`max_consecutive_losses=10`
- **SafetyCheckResult**: 安全チェック結果。`can_bet: bool`、`reason: str`

### types.py

Enum 定義。全て `str` 基底。

- **Surface**: `TURF`、`DIRT`
- **BetType**: `WIN`、`PLACE`、`WIDE`
- **RecoveryState**: `NORMAL`、`REDUCED`、`RECOVERING`
- **RegimeState**: `AGGRESSIVE`、`CONSERVATIVE`、`COLLAPSED`

---

## db/ ✅

PostgreSQL 接続とスキーマ定義。SQLAlchemy Core のみ使用 (ORM不使用)。

### schema.py

5スキーマのDDL定義を文字列定数として保持。`ALL_CREATE_STATEMENTS` に全DDLを格納。

- **raw**: EveryDB2 生データのローカルコピー
  - `races`: 複合PK + `race_id GENERATED ALWAYS AS` で文字列化。`surface`、`distance_band` も生成列
  - `entries`: `race_id` を FK とする出走馬データ。`finish_pos`、`win_odds`、`ba_taijyu` 等
  - `payouts`: 払戻情報。単勝・複勝の着順・払戻を格納
- **odds_history**: 時系列オッズ
  - `odds_snapshots`: 最新オッズスナップショット
  - `odds_time_series`: `happyo_time` (MMDDHHmm) ごとの時系列オッズ
  - `wide_odds`: ワイドオッズ (`kumi` = "3-7" 形式)
- **feature**: `features` テーブル。`feature_data` を JSONB で保存
- **prediction**: `predictions` テーブル。P/E/EV 全指標を列として格納
- **betting**: `bets` テーブル。投票記録。`regime_state`、`recovery_state` を保持。`race_id`、`created_at`、`bet_type` にインデックス

### connection.py

- **DatabaseConnection**: シングルトンエンジンのDB接続管理。`config/settings.yaml` から接続情報を読み取り、`PGPASSWORD` 環境変数でパスワードを上書き。データローダー (`load_races`、`load_entries_with_results`、`load_odds_snapshots`、`load_odds_time_series`) とデータセーバー (`save_predictions`、`save_bets`) を提供

---

## features/ 🔧

特徴量エンジン群。カテゴリ A-F を分担。

### feature_engine.py

- **FeatureEngine**: メインオーケストレータ。`build_all()` でバッチ学習用に3つの DataFrame をマージし全特徴量を計算。`build_features()` で推論用に `Race + list[Entry]` から単レース特徴量を計算

### intra_race_features.py

- **compute_intra_race_features()**: レース内相対値。`weight_diff_from_mean` (馬体重と平均の差)、`odds_rank` (オッズ順位) を計算

### odds_dynamics_features.py

- **compute_odds_dynamics()**: オッズ変化率・速度・ボラティリティ。`odds_drop_rate_60_10`、`odds_drop_rate_30_10`、`odds_velocity` (線形回帰の傾き)、`odds_volatility` (標準偏差)、`popularity_change_30_10`

### market_bias_features.py

- **compute_market_bias()**: 市場エントロピー・オーバーラウンド。`p_market_win_adj` (正規化市場確率)、`market_entropy` (シャノンエントロピー)、`overround` (胴元控除率)

### info_asymmetry_features.py

- **compute_hist_features()**: 履歴ベース情報非対称性。`expanding().shift(1)` で未来情報リークを完全遮断 (Rule 18)。`hist_hit_rate_topk`、`hist_roi_topk`、`hist_positive_return_ratio`、`hist_win_rate_same_condition`、`hist_market_entropy_avg`

### race_difficulty_model.py

- **compute_difficulty_score()**: レース難易度スコア。`grade_weight x field_factor x entropy_normalized` の積で算出。G1=1.0、一般=0.1 の重み付け

### leakage_validators.py

- **validate_no_future_leakage()**: expanding 系特徴量の未来情報リーク検証。hist 列と source 列を比較し、リーク箇所をリストで返却

---

## models/ 🔧

MLモデル群。10エクスポートクラスを `__init__.py` で公開。芝/ダートの2分割のみでサブモデルを管理。

### submodel_manager.py

- **SubModelManager**: surface 別モデル切り替え。`get_key(race)` で `"turf"` または `"dirt"` のキーを返却。`get_models()` で該当サブモデルを取得。`should_split_further()` で追加分割の要否を判定 (`MIN_SAMPLES=20000`)

### market_model.py

- **MarketModel**: 市場確率の予測と差分出力。`p_market_pred` は出力せず `signed_log_error`、`abs_log_error` のみを下流 (Stage2) に渡す。p_pred と p_market の両側クリップで log_error の発散を防止 (Rule 13)

### stage1_ability_model.py

- **AbilityModel**: Stage1 能力モデル。LightGBM Ranker (lambdarank) で芝/ダート別に学習。オッズ特徴量は一切使用しない (Rule 1)。出力は `p_ability_win` (softmax変換) と `p_ability_place`

### two_stage_return_model.py

- **WinTwoStageModel**: 単勝2段階モデル。Stage A = P(win) の2値分類、Stage B = E(win_odds | win) の回帰。EV = P x E でゼロ偏重問題を解決。`market_log_error` のみを入力に使用
- **PlaceTwoStageModel**: 複勝2段階モデル。同構造で複勝的中確率と払戻を予測

### ev_correction_model.py

- **EVCorrectionModel**: EV補正モデル。P補正 (binary classification, `init_score=logit(p_pred)`) と E補正 (regression, `weight=1/√p`) に分解。独立性破綻を解決。`EV_corrected = P_corrected x E_corrected`

### wide_two_stage_model.py

- **WideTwoStageModel**: ワイド予測モデル。`score = EV / (E x √P)` のシャープレシオ近似でリスク調整スコアを算出

### wide_pair_builder.py

- **WideJointPairBuilder**: レース内の全馬ペア C(n,2) を構築。各ペアに `joint_hit` ラベル、`wide_odds`、`popularity_sum`、`running_style_combo` を付与

### race_quality_screener.py

- **RaceQualityScreener**: レース品質スクリーニング。「このレースは投票する価値があるか」を判定。Stage2 出力に依存しない指標のみを使用 (Rule 16)

### regime_detector.py

- **RegimeDetector**: 3状態市場分類。`fav_rate x overround` の市場指標ベースで AGGRESSIVE / CONSERVATIVE / COLLAPSED を判定 (Rule 19)。ヒステリシス付き

### robust_confidence_estimator.py

- **RobustConfidenceEstimator**: 信頼区間推定。`min(Conformal Prediction, Rolling Quantile)` を採用し、より保守的な下限を使用 (Rule 4)

### walk_forward_cv.py

- **WalkForwardCV**: ウォークフォワード交差検証。expanding window で時系列 CV を実行。各フォールドで freeze → test のパラメータ固定を保証 (Rule 7)

---

## betting/ 🔧

ベッティング層。9モジュール構成。

### orchestrator.py

- **BettingOrchestrator**: メインオーケストレータ。`process_race()` でステップ①〜⑩を実行しベット候補を生成。`finalize_bets()` でステップ⑫の最終キャンセルチェックと投票を実行。Protocol で各コンポーネントに依存

### stake_calculator.py

- **StakeCalculator**: 賭け金計算。EV下限値に連動したケリー基準 (`kelly_fraction = (ev_lower - 1) / (odds - 1)`)。`KELLY_FRACTION_CAP=0.25` (full Kelly の1/4)。100円単位に切り捨て。1レース2%露出キャップ (Rule 6)

### drawdown_controller.py

- **DrawdownController**: DD x Rolling ROI の複合判定。EWMA ハイブリッド + ヒステリシス付き回復ロジック。3段階回復: NORMAL → REDUCED → RECOVERING → NORMAL。`ROLLING_WINDOW=150`、`MAX_ADJUSTMENT_PER_N_BETS=20` (Rule 9, Rule 17)

### late_money_filter.py

- **LastMinuteSignal**: 直前オッズ変動シグナルの Enum (`NO_ACTION`、`CANCEL`、`ADD_CANDIDATE`、`UNKNOWN`)
- **LateMoneyFilter**: t-3min で判定、t-2min はログのみ (Rule 8, Rule 14)。`check_t3()` でキャンセル要否を判定、`log_t2()` で t-2min スナップショットを記録

### meta_switcher.py

- **MetaSwitcher**: レジーム連動戦略パラメータ。RegimeDetector の出力をベッティング層で使いやすい形に変換。`get_strategy_params()` で ev_threshold、stake_multiplier 等を返却

### gate_keeper.py

- **GateKeeper**: EV下限値ベース最終足切り。`should_bet()` で `ev_lower_corrected >= ev_threshold` を判定。`filter_bets()` でリストから不合格を除外

### win_strategy.py

- **WinStrategy**: 単勝戦略。EV閾値で候補を抽出しケリー基準で賭け金を算出。`max_bets=2` がデフォルト

### place_strategy.py

- **PlaceStrategy**: 複勝戦略。`p_place x e_return_place = ev_place` の信頼区間下限値を基準に候補を抽出。単勝より分散が小さいため閾値は低め

### wide_strategy.py

- **WideStrategy**: ワイド戦略。`score = EV / (E x √P)` のシャープレシオ近似で候補を選択 (Rule 3/15)。EV閾値 + スコア閾値の複合フィルタ。`max_bets=3` がデフォルト

---

## backtest/ 🔧

バックテストパッケージ。エンジン・検証・パラメータ凍結の3モジュール。

### engine.py

- **BacktestEngine**: バックテスト実行エンジン。学習済みモデルで履歴データをシミュレーションし投資成績を評価。`BacktestResult` データクラスで total_bets、total_stake、total_return、winning_bets 等を返却

### validation_suite.py

- **BacktestValidationSuite**: 全検証テスト集約スイート。各テストは `passed (bool) + message (str)` を返却。`run_all()` で一括実行。全テスト通過が Hold-out 評価の前提条件

### parameter_freeze_protocol.py

- **ParameterFreezeProtocol**: パラメータ凍結プロトコル (Rule 7)。`freeze()` でスナップショットを取得し `verify()` で変更を検出。`frozen_period()` コンテキストマネージャで OOS 期間を定義。ハッシュベースの不変性チェック

---

## pipelines/ 🔧

### training_pipeline.py

- **TrainingPipeline**: 全学習パイプライン v5.4 (§11)。Phase C の全モデルを正しい順序で学習し `TrainedModelsV5` に格納。MLflow に実験を記録。学習順序: FeatureEngine → SubModelManager → AbilityModel → MarketModel → WinTwoStageModel → EVCorrectionModel → PlaceTwoStageModel → WideJointPairBuilder → WideTwoStageModel → RaceQualityScreener → RegimeDetector → RobustConfidenceEstimator

---

## automation/ 🚧

自動化パッケージ。JRA-IPAT 投票とタスクスケジューリング。

### pat_voter.py

- **PatVoter**: JRA-IPAT 投票インタフェース (F-1b)。`PatApiProtocol` で API を抽象化しテストで mock 注入可能。`BetSubmissionResult` で投票結果 (success, bet_id, error) を返却

### scheduler.py

- **Scheduler**: レース日タスクスケジューリング (F-2)。t-10min で `process_race()`、t-3min で `finalize_bets()`、t-2min でログ記録。SafetyGuard チェックで全体の投票可否を判定

### safety_guard.py

- **SafetyGuard**: 安全ガード (F-1a)。`SafetyConfig` の閾値に基づき投票の可否を判定。bankroll 最低ライン、日次/週次損失上限、連続敗北数、緊急停止フラグを監視。`check()` で `SafetyCheckResult` を返却

---

## monitoring/ 🚧

監視パッケージ。予測精度の劣化検知と通知。

### model_monitor.py

- **ModelMonitor**: 予測精度劣化検知 (F-3b)。Rolling ROI、的中率、EV乖離を監視。`PerformanceReport` (n_races, hit_rate, rolling_roi, ev_mean_error, regime, needs_attention) と `DriftReport` (PSI で特徴量ドリフト検知) を生成

### auto_retrain_trigger.py

- **AutoRetrainTrigger**: 再学習トリガー判定 (F-3c)。COLLAPSED 連続100レース、的中率低下、PSI 特徴量ドリフトを条件に再学習を判定。クールダウン期間で頻繁な再学習を防止。`RetrainDecision` (triggered, reason) を返却

### notifier.py

- **LoggingNotifier**: ログ出力のみの通知 (開発/テスト用)
- **CompositeNotifier**: 複数の通知バックエンドに配信。1つでも成功すれば True。`NotifierProtocol` で `send(message, level)` を定義

---

## ingestion/ 🚧

データ取得パッケージ。

### jvlink_fetcher.py

- **JVLinkFetcher**: JRA-VAN データ取得インタフェース (F-4a)。JV-Link SDK (Windows COM) または DatabaseConnection 経由でレースカード・結果・オッズを取得

### odds_collector.py

- **OddsCollector**: オッズ収集 (F-4b)。5分間隔の定期収集 + t-3min/t-2min スナップショット。`OddsFetcherProtocol` で抽象化。t-3min スナップショットは LateMoneyFilter の判定に使用

---

## テスト構造

テストは `tests/` 配下に配置。全て `unittest.mock` を使用し DB 不要で実行可能。

| 項目 | 値 |
|---|---|
| テストファイル数 | 42 ファイル |
| 総行数 | 6,163 行 |
| モック方式 | 全て `unittest.mock` |
| 実行方法 | `python -m pytest tests/ -v` |
| カバレッジ | `python -m pytest tests/ -v --cov=src --cov-report=term-missing` |

### テストファイル一覧 (モジュール別)

| モジュール | テストファイル |
|---|---|
| domain/ | `test_domain.py` |
| db/ | `test_db.py` |
| features/ | `test_feature_engine.py`, `test_intra_race_features.py`, `test_odds_dynamics_features.py`, `test_market_bias_features.py`, `test_info_asymmetry_features.py`, `test_race_difficulty.py`, `test_leakage.py` |
| models/ | `test_submodel_manager.py`, `test_market_model.py`, `test_stage1_ability.py`, `test_two_stage_return_model.py`, `test_ev_correction.py`, `test_wide_two_stage_model.py`, `test_wide_pair_builder.py`, `test_race_quality_screener.py`, `test_regime_detector.py`, `test_robust_confidence_estimator.py`, `test_walk_forward_cv.py` |
| betting/ | `test_orchestrator.py`, `test_stake_calculator.py`, `test_drawdown_controller.py`, `test_late_money_filter.py`, `test_meta_switcher.py`, `test_gate_keeper.py`, `test_win_strategy.py`, `test_place_strategy.py`, `test_wide_strategy.py` |
| backtest/ | `test_backtest_engine.py`, `test_validation_suite.py`, `test_parameter_freeze.py` |
| pipelines/ | `test_training_pipeline.py` |
| automation/ | `test_pat_voter.py`, `test_scheduler.py`, `test_safety_guard.py` |
| monitoring/ | `test_model_monitor.py`, `test_auto_retrain_trigger.py`, `test_notifier.py` |
| ingestion/ | `test_jvlink_fetcher.py`, `test_odds_collector.py` |
| 共通 | `test_settings.py` |

---

> **次のドキュメント:** [設定リファレンス](03_configuration.md) | **前のドキュメント:** [アーキテクチャ](01_architecture.md) | **ドキュメント一覧:** [README](../../README.md)
