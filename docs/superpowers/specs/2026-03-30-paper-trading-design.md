# Paper Trading システム設計書

## 1. 概要

バックテストでROI 143%を達成したMLモデルを、実際のレースでお金をかけずに3ヶ月間モニタリングするシステム。EveryDB2の速報系データをリアルタイムデータソースとし、予測→ベット記録→結果照合→ROI追跡のサイクルを自動化する。

## 2. 要件

| 項目 | 内容 |
|------|------|
| 目的 | 実戦投入前のPaper Tradingによる性能検証 |
| 期間 | 3ヶ月間 |
| データソース | EveryDB2 (PostgreSQL) 速報系データ |
| ベット戦略 | 複勝固定100円ベット（バックテスト同様） |
| モデル | 学習済みLightGBM固定（3ヶ月間再学習なし） |
| 通知 | Slack (Incoming Webhook) + HTMLレポート |
| 実行方式 | Windows Task Scheduler + バッチ型スクリプト |
| Dry-run | 過去データで本番パイプラインの動作確認 |

## 3. アーキテクチャ

### 3.1 全体フロー

```
EveryDB2 (GUI自動更新モード、レース開催日に常時起動)
  ├─ 自動でJRA-VANから取得:
  │  - 速報オッズ (0B31-0B36): 金土日随時
  │  - 速報馬体重 (0B11): 発走約1時間前
  │  - 速報レース結果 (0B12): レース確定後
  │  - 天候馬場状態 (0B14): 随時
  │  → 全て PostgreSQL (localhost:5432/everydb2) に書き込み
  └────────────────────────────────────────────────
                          ↓
run_paper_trading.py が PostgreSQL を SELECT するのみ
```

### 3.2 1日のスケジュール

```
08:30  setup     — 出走表取得・履歴特徴量生成・スケジュール生成
09:00  watch     — 各レース-5分にPostgreSQLから馬体重+オッズ取得→EV計算→Slack通知
18:30  reconcile — レース結果照合・bets蓄積・HTML更新・ModelMonitor・Slackサマリー
```

### 3.3 各レースのタイミング

```
T - 60分: EveryDB2が馬体重を自動取得 (0B11)
T - 5分:  watchスクリプトがPostgreSQLから馬体重+最新オッズをSELECT
T - 3分:  EV計算 → ベット決定 → Slack通知
T:       レース発走
T + 数分: EveryDB2がレース結果を自動取得 (0B12)
```

### 3.4 3つの実行モード

```bash
# モーニング: 出走表取得・特徴量生成
python scripts/run_paper_trading.py --mode setup --date 2026-04-05

# デイタイム: 各レースのベット通知（約8時間稼働）
python scripts/run_paper_trading.py --mode watch --date 2026-04-05

# イブニング: 結果照合・レポート
python scripts/run_paper_trading.py --mode reconcile --date 2026-04-05

# 過去データで動作確認
python scripts/run_paper_trading.py --mode dry-run --date 2024-07-13
python scripts/run_paper_trading.py --mode dry-run --start 2024-07-01 --end 2024-07-31
```

## 4. コンポーネント設計

### 4.1 新規コンポーネント

```
src/paper_trading/
├── __init__.py
├── predictor.py       # 日次予測ロジック (PaperPredictor)
├── reconciler.py      # 結果照合・ROI計算 (PaperReconciler)
├── watcher.py         # レース時刻監視・ベット通知 (RaceWatcher)
├── report.py          # Paper Trading用HTMLレポート
└── config.py          # 設定 (PaperTradingConfig)

src/db/
├── everydb2_queries.py  # NEW: EveryDB2速報系テーブルへの直接クエリ
├── model_loader.py      # NEW: MLflowからTrainedModelsV5をロード

src/backtest/
├── race_predictor.py    # REFACTOR: BacktestEngineから抽出された推論パイプライン

scripts/
└── run_paper_trading.py   # メインRunner
```

### 4.2 モデル読み込み (model_loader.py) — [I-2 修正]

Paper Tradingでは `TrainedModelsV5`（複数サブモデル + 品質スクリーナー + レジーム検出器）をMLflowから読み込む必要がある。

```python
@dataclass
class ModelInfo:
    mlflow_run_id: str
    train_start: str
    train_end: str
    loaded_at: str

class ModelLoader:
    """MLflowからTrainedModelsV5を構築してロード"""

    def load(self, run_id: str | None = None) -> tuple[TrainedModelsV5, ModelInfo]:
        """
        MLflowから学習済みモデルを読み込み、TrainedModelsV5を再構築。
        run_id未指定時は最新の成功runを使用。

        ロード対象MLflow artifacts:
        - stage1_turf / stage1_dirt — AbilityModel
        - win_hit_turf / win_ret_turf / win_hit_dirt / win_ret_dirt — WinTwoStageModel
        - ev_corrector_p_turf / ev_corrector_e_turf / ... — EVCorrectionModel
        - place_hit_turf / place_ret_turf / ... — PlaceTwoStageModel
        - race_quality — RaceQualityScreener
        - regime_detector — RegimeDetector

        model_info.json に run_id, train_period, loaded_at を保存。
        """
```

`PaperPredictor.__init__` は `model_path: str` ではなく `ModelLoader` を受け取る。

### 4.3 推論パイプラインの共通化 (race_predictor.py) — [I-3 修正]

BacktestEngineのレース別推論ループ（`engine.py` 4a-4g、約110行）を `RacePredictor` として抽出し、BacktestEngine と PaperPredictor の両方から利用する。

```python
class RacePredictor:
    """1レース分の特徴量→推論→ベット候補生成を担当する共通コンポーネント"""

    def __init__(self, models: TrainedModelsV5): ...

    def predict(
        self,
        race_df: pd.DataFrame,
        hist_features: pd.DataFrame | None = None,
        jockey_features: pd.DataFrame | None = None,
        trainer_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        1レースの推論パイプライン:
        1. サブモデル選択 (surface_key → turf/dirt)
        2. レジーム検出
        3. HorseHistoryFeatures マージ + race_transforms
        4. interaction_features 計算
        5. MarketModel → signed_log_error
        6. Stage1 → ability_probs
        7. PlaceAbilityModel
        8. WinTwoStageModel → win_ev
        9. JockeyContext + TrainerContext マージ
        10. EVCorrection
        11. PlaceTwoStageModel → place_ev
        12. RobustConfidenceEstimator → ev_lower_corrected

        Returns: 推論結果を追加したDataFrame (EV, ev_lower_corrected等の列を含む)
        """

    def select_bets(
        self, race_df: pd.DataFrame, bankroll: float,
        regime_params: dict
    ) -> list[dict]:
        """
        EV > 閾値 の馬をベット候補として抽出。
        BacktestEngine._generate_bets() と同じロジック。
        """
```

**BacktestEngineの変更**: `run()` メソッド内の推論ループを `RacePredictor.predict()` + `select_bets()` に委譲。BacktestEngine自身はバンクロール管理・バックテスト用の繰り返し処理に集中。

**PaperPredictorの変更**: `predict_race()` は `RacePredictor.predict()` + `select_bets()` を呼び出すのみ。推論ロジックの重複なし。

**BettingOrchestratorとの関係**: BettingOrchestratorは12ステップのベット決定フロー（StakeCalculator, GateKeeper, LateMoneyFilter等）を実装するが、Paper Tradingの「複勝固定100円」戦略には過剰。本番運用フェーズでBettingOrchestratorに移行する際、PaperPredictorのベット選定部分をBettingOrchestrator.process_race()に差し替える。

### 4.4 PaperPredictor (predictor.py)

```python
class PaperPredictor:
    def __init__(
        self,
        repo: DataRepository,
        race_predictor: RacePredictor,
        models: TrainedModelsV5,
    ): ...

    def setup(self, date: date) -> RaceSchedule:
        """
        当日の出走表を取得し、履歴特徴量を生成してスケジュールを返す。

        出力: RaceSchedule (各レースのrace_id, post_time, surface等)
        保存: schedule.json, 事前計算済み特徴量を predictions/YYYYMMDD_pre.parquet
        注意: 事前計算段階では weight_absolute, オッズ関連列はNaN。
               setup段階ではベット決定を行わない（EV計算が不可能なため）。
               LightGBM自体はNaNを許容するが、EVはオッズに依存するため。
        """

    def predict_race(
        self,
        race_id: str,
        pre_computed_features: pd.DataFrame,
        horse_weights: pd.DataFrame,
        odds: pd.DataFrame,
        bankroll: float,
    ) -> list[dict]:
        """
        1レース分の予測。馬体重+オッズ特徴量をマージして推論。

        1. pre_computed_features に horse_weights, odds をマージ
        2. RacePredictor.predict() で推論
        3. RacePredictor.select_bets() でベット候補を抽出
        4. 返値は BacktestEngine の bet_history と同じスキーマ
        """
```

### 4.5 RaceWatcher (watcher.py)

watchフェーズで呼び出し、各レースの発走時刻に合わせてベット通知を行う。

```python
class RaceWatcher:
    def __init__(
        self,
        predictor: PaperPredictor,
        everydb2: EveryDB2Queries,
        notifier: Notifier,
    ): ...

    def watch(self, date: date, schedule: RaceSchedule):
        """
        スケジュールに基づき、各レースの発走-5分にベット通知。

        耐障害性:
        - PostgreSQL接続断: 接続を再確立してリトライ
        - プロセスクラッシュ: watchは冪等ではないが、
          既に通知済みのレースはpredictions/YYYYMMDD.parquetで判定可能
        - ハートビート: 5分おきにログ出力（プロセス生存確認用）
        """
        bankroll = self._load_bankroll(date)  # 前日までの累積資金

        for race in schedule.races:
            # 既に処理済みのレースはスキップ（冪等性）
            if self._already_processed(race.race_id, date):
                continue

            wait_until(race.post_time - timedelta(minutes=5))

            # PostgreSQLから当日データを取得
            horse_weights = self.everydb2.get_horse_weights(race.race_id)
            odds = self.everydb2.get_latest_odds(race.race_id)

            # 取得失敗時のリトライ（最大3回、1分間隔）
            for attempt in range(3):
                if horse_weights is not None and odds is not None:
                    break
                sleep(60)
                horse_weights = self.everydb2.get_horse_weights(race.race_id)
                odds = self.everydb2.get_latest_odds(race.race_id)
            else:
                logger.warning(f"データ取得失敗: {race.race_id} をスキップ")
                continue

            bets = self.predictor.predict_race(
                race.race_id, pre_computed_features, horse_weights, odds, bankroll
            )
            self.notifier.send_prediction(bets, date)
            bankroll = self._update_bankroll(bets, bankroll)
```

### 4.6 PaperReconciler (reconciler.py)

reconcileフェーズで呼び出し、予測と結果を照合してROIを追跡する。

```python
class PaperReconciler:
    def __init__(
        self,
        repo: DataRepository,
        bet_store: ParquetStore,
        everydb2: EveryDB2Queries,
        monitor: ModelMonitor,
    ): ...

    def reconcile(self, date: date) -> DailyResult:
        """
        当日のレース結果を取得し、予測と照合。

        冪等性: 同一 race_id のレコードが既に bets.parquet に存在する場合は
        スキップ（再実行安全）。未確定のレース（result=None）のみを処理。
        """
        # 1. 当日の予測を読み込み
        # 2. EveryDB2からレース結果・払戻を取得 (0B12)
        # 3. 予測と結果をマージ
        # 4. 複勝的中判定: umaban が複勝払戻対象に含まれるか
        # 5. bets.parquet に追記（重複排除: race_id + umaban で既存チェック）
        # 6. 累積統計を計算 (ROI, max DD, bankroll)
        # 7. ModelMonitorでドリフトチェック
        # 8. DailyResultを返す
```

### 4.7 EveryDB2Queries (everydb2_queries.py)

EveryDB2のPostgreSQLテーブルへの直接クエリラッパー。

```python
class EveryDB2Queries:
    def __init__(self, connection_string: str): ...

    def get_race_schedule(self, date: date) -> list[RaceInfo] | None:
        """当日のレース時刻・出走馬を取得。
        蓄積系テーブル n_uma_race を使用（木曜以降に利用可能）。
        レースがない（非開催日等）は空リストを返す。"""

    def get_horse_weights(self, race_id: str) -> pd.DataFrame | None:
        """速報馬体重を取得。
        EveryDB2の速報系テーブル（s_ プレフィックス）から取得。
        発走約1時間前にEveryDB2自動更新で反映される。
        ※ 実装時にEveryDB2インスタンスで実際のテーブル名を確認"""

    def get_latest_odds(self, race_id: str) -> pd.DataFrame | None:
        """最新の速報オッズを取得。
        EveryDB2の速報系テーブルから取得。金土日随時更新。
        タイムスタンプ列を確認し、古すぎるデータはNoneを返す。
        ※ 実装時にEveryDB2インスタンスで実際のテーブル名を確認"""

    def get_race_results(self, date: date) -> pd.DataFrame:
        """レース結果・払戻を取得。
        蓄積系テーブル n_race + n_uma_race + n_harai を使用。
        reconcileは18:30実行のため、月曜の通常データ更新で確定データが利用可能。"""

    def get_track_condition(self, race_id: str) -> str | None:
        """天候馬場状態を取得。
        速報系テーブル（s_ プレフィックス）を使用。"""
```

### 4.8 SlackNotifier (monitoring/notifier.py 拡張) — [S-2 修正]

既存の `NotifierProtocol` を実装しつつ、Paper Trading用の構造化メソッドを追加。

```python
class SlackNotifier:
    """NotifierProtocolを実装 + Paper Trading用の構造化メソッド"""

    def __init__(self, webhook_url: str): ...

    # NotifierProtocol準拠（ModelMonitor等からのアラート用）
    def send(self, message: str, level: str = "info") -> bool:
        """汎用通知。CompositeNotifierに登録して使用。"""

    # Paper Trading専用メソッド（Slack Blocks/Attachmentsでリッチ通知）
    def send_prediction(self, bets: list[dict], date: date):
        """当日のベット推薦を通知。
        内部でsend()を呼び出すか、Slack APIを直接使用してBlocks形式で送信。"""

    def send_daily_result(self, summary: dict, date: date):
        """日次サマリーを通知（当日ROI + 累積ROI + max DD）"""

    # CompositeNotifierへの登録
    # SlackNotifier を CompositeNotifier に追加することで、
    # ModelMonitor のアラートも自動的にSlackに通知される。
```

### 4.9 PaperTradingReport (paper_trading/report.py)

既存のBacktestReportGeneratorを拡張し、Paper Trading用レポートを生成。

- バックテストレポートと同じ構造（KPIカード、資金推移チャート、月次ダッシュボード）
- 追加: 日次パフォーマンステーブル
- 追加: バックテストROIとの比較ライン

### 4.10 PaperTradingConfig (config.py) — [S-7 修正]

```python
@dataclass
class PaperTradingConfig:
    # Slack
    slack_webhook_url: str           # 環境変数 SLACK_WEBHOOK_URL

    # モデル
    mlflow_run_id: str | None = None # None時は最新runを使用
    mlflow_tracking_uri: str = "file:///mlruns"

    # ベット
    ev_threshold: float = 1.0        # EV閾値（バックテスト同様）
    initial_bankroll: float = 100000.0
    stake: float = 100.0             # 固定100円

    # タイミング
    watch_lead_minutes: int = 5      # 発走何分前にベット通知
    retry_count: int = 3             # データ取得リトライ回数
    retry_interval_seconds: int = 60

    # EveryDB2
    everydb2_connection_string: str  # config/settings.yaml から取得
    query_timeout_seconds: int = 30

    # パス
    paper_trading_dir: Path = Path("data/paper_trading")
```

## 5. データスキーマ

### 5.1 bets.parquet — [I-4 修正]

BacktestEngineの `bet_history` スキーマの**スーパーセット**として定義。既存フィールドは名前・型を完全に一致させる。

```python
# === BacktestEngine bet_history と共通フィールド（変更不可）===
{
    "race_id": str,          # レースID (16文字)
    "bet_type": str,         # "place" (BetType.value)
    "umaban": int,           # 馬番
    "stake": float,          # 100.0
    "odds": float,           # 複勝オッズ（予測時点）
    "result": float,         # 払戻額（0 = 不的中）
    "surface": str,          # "turf" / "dirt"
    "distance": int,         # 距離
    "ev": float,             # ev_lower_corrected (float)
    "popularity": int,       # 人気順位
    "bankroll_after": float, # ベット後の資金
}

# === Paper Trading追加フィールド ===
{
    "race_date": datetime64, # race_id[:8] から導出（BacktestReportGenerator._derive_fieldsと同じ）
    "horse_name": str,       # 馬名（Slack通知用、レポート表示用）
    "is_paper": bool,        # True (Paper Trading識別用)
}
```

**格納先**: `data/paper_trading/bets.parquet`（バックテストの `data/bets/bets.parquet` とは別ファイル）。
**理由**: Paper Tradingデータは `is_paper=True` で識別可能だが、格納場所を分けることでバックテストデータとの偶発的な混入を防止。将来の統合レポートでは両ファイルを結合して使用。

**PostgreSQL betting.bets との関係**: Paper Tradingでは `data/paper_trading/bets.parquet` にのみ保存し、PostgreSQLの `betting.bets` テーブルには書き込まない。理由はPaper Trading中はスキーマの結合を避け、実戦運用開始時にPostgreSQLへ移行するため。

### 5.2 schedule.json

```python
{
    "date": "2026-04-05",
    "races": [
        {
            "race_id": "2026040510010101",
            "venue": "中山",
            "race_num": 1,
            "post_time": "10:05",
            "surface": "turf",
            "distance": 1200,
            "horses": ["馬名1", "馬名2", ...]
        },
        ...
    ]
}
```

## 6. EveryDB2 テーブル参照 — [I-1 修正]

### 既存の蓄積系テーブル（n_ プレフィックス）

プロジェクトのETL（`run_etl.py`）でParquetに取り込んでいるテーブル。これらは確定データで、月曜14:00頃に更新される。

| テーブル | 用途 | 確定タイミング |
|---------|------|-------------|
| `n_race` | レース条件（場名、距離、コース、天候、クラス） | 月曜 14:00 |
| `n_uma_race` | 馬毎レース結果（着順、オッズ、人気、馬体重） | 月曜 14:00 |
| `n_uma` | 競走馬マスタ（血統、能力） | 差分更新 |
| `n_harai` | 払戻情報 | 月曜 14:00 |
| `n_odds_tanpuku` | 単複オッズ（確定） | 月曜 14:00 |
| `n_jodds_tanpuku` | 時系列単複オッズ（Late Money用） | 月曜 14:00 |
| `n_kisyu_seiseki` | 騎手成績統計 | 差分更新 |

### 速報系テーブル（s_ プレフィックス）— 実装時に確認必須

EveryDB2の自動更新でリアルタイムに反映されるテーブル。**実装開始前にEveryDB2インスタンスで実際のテーブル名を確認すること。**

| 用途 | 推定テーブル名 | データ種別 | 取得タイミング | リテンション |
|------|-------------|-----------|-------------|------------|
| 出走表 | `n_uma_race` | 0B15 | 木曜〜 | 蓄積系（恒久） |
| 馬体重 | `s_bataijyu` (要確認) | 0B11 | 発走約1時間前 | 1週間 |
| オッズ(単複) | `s_odds_tanpuku` (要確認) | 0B31 | 金土日随時 | 1週間 |
| オッズ(ワイド) | `s_odds_wide` (要確認) | 0B33 | 金土日随時 | 1週間 |
| レース結果 | `s_race` (要確認) | 0B12 | レース確定後 | 1週間 |
| 払戻 | `s_harai` (要確認) | 0B12 | レース確定後 | 1週間 |
| 天候馬場 | `s_tenko_baba` (要確認) | 0B14 | 随時 | 1週間 |

**確認方法**: EveryDB2を起動し、「テーブル状況参照」（10章）またはSQL実行（8章）で `s_` プレフィックスのテーブル一覧と列構造を確認。

**複合主キー**: 全レース系テーブルの共通キーは `(Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum)`。プロジェクトでは `race_id = YYYYMMDD + JyoCD + Kaiji + Nichiji + RaceNum` (16文字) に変換して使用。

## 7. 特徴量の2段階生成 — [I-5 修正]

| 段階 | タイミング | 特徴量 | 取得元 | 備考 |
|------|----------|--------|--------|------|
| 事前計算 | setup (08:30) | 血統、過去走成績、騎手/調教師統計、坂路調教等 | Parquet (蓄積系) | weight_absolute, オッズ関連はNaN |
| 当日取得 | watch (各レース-5分) | 馬体重(weight_absolute)、オッズ関連、人気順位 | EveryDB2 PostgreSQL (速報系) | NaN列を上書き |

**注意点**:
- setup段階ではNaN列があるためEV計算が不可能 → setupではベット決定を行わない
- LightGBMはNaNをネイティブに処理可能だが、EV = P(hit) × E(odds|hit) のうち E(odds|hit) はオッズに依存するため、オッズなしでは意味のあるEVが計算できない
- watch段階で当日データをマージした後にのみ、RacePredictor.predict() → select_bets() のフルパイプラインを実行

## 8. Dry-runモード — [S-4 修正]

本番パイプラインの動作確認用。過去データで日次シミュレーションを実行。

```bash
# 1日分
python scripts/run_paper_trading.py --mode dry-run --date 2024-07-13

# 期間指定
python scripts/run_paper_trading.py --mode dry-run --start 2024-07-01 --end 2024-07-31
```

### データフロー

```
dry-run のデータフロー（本番との違いはデータソースのみ）:

本番:  EveryDB2 PostgreSQL (速報系) → 特徴量生成 → 推論
dry-run: Parquet (蓄積系、過去データ) → 特徴量生成 → 推論

dry-runでは2段階生成は不要:
- Parquetには全特徴量（馬体重、オッズ含む）が既に存在
- 1日分のDataFrameを抽出し、RacePredictor.predict() → select_bets() を実行
- 結果は n_uma_race の確定データから取得
```

### 出力

- `data/paper_trading/dry_run/YYYYMMDD.json` に日次結果を保存
- `data/paper_trading/dry_run/` に**のみ**出力（bets.parquet は更新しない）
- 最後にdry-run期間の集計ROIを表示

### バックテストとの比較

| | バックテスト | Dry-run |
|---|---|---|
| 期間 | 数年一括 | 1日〜1ヶ月単位 |
| 予測方式 | テスト期間全体を一括推論 | 1日ずつ推論（本番と同じパイプライン） |
| 目的 | モデル評価 | 本番パイプラインの動作確認 |
| データソース | Parquet | Parquet（同じ） |

## 9. エラーハンドリング — [S-1, S-5 修正]

| シナリオ | 対応 |
|---------|------|
| EveryDB2未起動 | setup開始時にPostgreSQL接続チェック。失敗→Slack警告+終了 |
| 馬体重未取得 | 発走-5分時点でNULL→1分間リトライ(最大3回)。不可ならスキップ |
| オッズが古い | タイムスタンプ確認。発走-10分以前なら警告付きで使用 |
| レース結果未反映 | reconcileで未確定レースをスキップ、次回に委ねる |
| 出走取消/騎手変更 | 0B14確認。取消馬をベット候補から除外 |
| ネットワーク障害 | 接続タイムアウト→Slack通知+次回cronに委ねる |
| watch中にクラッシュ | 再起動後、既に通知済みのrace_idはpredictionsで判定してスキップ |
| reconcile重複実行 | race_id + umaban で既存レコードをチェック。重複はスキップ |
| PostgreSQL接続断 | watch中の接続断は自動再接続。再接続失敗はSlack通知 |

## 10. 運用スケジュール

```
レース開催日（土日+祝日）:
  08:30  setup     — 出走表取得・特徴量生成
  09:00  watch     — 各レース-5分にベット通知
  18:30  reconcile — 結果照合・レポート更新

非開催日:
  何もしない（setupが開催情報を確認し、レースがなければ終了）
```

## 11. データディレクトリ

```
data/paper_trading/
├── predictions/
│   ├── YYYYMMDD_pre.parquet  # setup: 事前計算済み特徴量
│   └── YYYYMMDD.parquet      # watch: 最終予測（当日データマージ済み）
├── bets.parquet               # 全ベット履歴（累積、reconcileで追記）
├── daily_summary/
│   └── YYYYMMDD.json         # 日次サマリー
├── schedule.json              # 当日のレーススケジュール
├── report.html                # 累積HTMLレポート
├── dry_run/                   # dry-run結果（bets.parquetとは別管理）
│   └── YYYYMMDD.json
└── model/
    └── model_info.json        # 使用モデルのメタ情報 (mlflow_run_id等)
```

## 12. 前提条件

1. EveryDB2がインストールされ、PostgreSQL (localhost:5432/everydb2) に接続済み
2. EveryDB2の自動更新が有効化され、速報系データ種別（0B11, 0B31, 0B12, 0B14等）が選択済み
3. レース開催日にEveryDB2がGUI自動更新モードで起動している
4. 学習済みモデルがMLflowに保存済み（run_train.pyで生成）
5. Slack Incoming Webhook URLが環境変数 `SLACK_WEBHOOK_URL` に設定済み
6. Windows Task Schedulerで3つのタスクが登録済み
7. 実装開始前にEveryDB2インスタンスで `s_` プレフィックスのテーブル名と列構造を確認済み
