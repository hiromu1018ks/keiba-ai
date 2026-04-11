# レース当日自動化設計

Date: 2026-04-11
Status: Draft

## Context

現在のレース当日の手動フロー:
1. EveryDB2 GUI を開いてデータを手動更新
2. `python scripts/run_paper_trading.py --mode predict --date YYYYMMDD --ensemble` を手動実行

これを1コマンドに統合し、将来的なフル自動化への基盤を構築する。

## 実装方針

Phase 1 (今回): 単発実行スクリプト — EveryDB2更新→予測を1コマンド化
Phase 2 (次回): 1日スケジューラ — 朝準備→レースループ→リコンサイル

## Phase 1 設計

### コマンド

```bash
# 予測実行 (EveryDB2更新 + Setup + Predict)
python scripts/run_race_day.py --mode full --date 2026-04-11 --ensemble

# リコンサイル (レース後、結果照合)
python scripts/run_race_day.py --mode reconcile --date 2026-04-11
```

### フロー

```
run_race_day.py --mode full
  │
  ├─ 1. EveryDB2Updater.run()
  │     subprocess.run(["EveryDB2.exe", "CMDLINE"])
  │     → PostgreSQL (everydb2) の s_/n_ テーブルを更新
  │     → タイムアウト10分、リトライ2回
  │
  ├─ 2. load_models()
  │     MLflow から学習済みモデルをロード
  │
  ├─ 3. run_setup(config, models, store, target_date)
  │     レーススケジュール構築 → schedule.json
  │
  └─ 4. run_predict(config, models, store, target_date, use_ensemble)
        特徴量生成 → 推論 → ベット保存 → Slack通知

run_race_day.py --mode reconcile
  └─ run_reconcile(config, store, target_date)
      結果照合 → ROI計算 → レポート
```

### 新規ファイル

#### `src/automation/everydb2_updater.py`

EveryDB2.exe CMDLINE の subprocess ラッパー。

```python
@dataclass
class UpdateResult:
    """EveryDB2更新結果"""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    elapsed_seconds: float

class EveryDB2Updater:
    def __init__(self, exe_path: Path, timeout_minutes: int = 10, retry_count: int = 2): ...
    def run(self) -> UpdateResult:
        """EveryDB2 CMDLINE を実行してデータを更新。
        タイムアウト・リトライ付き。結果は UpdateResult で返す。"""
    def verify_update(self, db: EveryDB2Queries, target_date: date) -> bool:
        """PostgreSQL の target_date データが存在するか確認"""
```

#### `src/paper_trading/runner.py` (新規: run_paper_trading.py から抽出)

`run_paper_trading.py` の各モード関数を、`argparse.Namespace` に依存しない
型付き関数として再定義する。`run_paper_trading.py` は薄いCLIラッパーになる。

```python
def load_config(
    *,
    slack_webhook_url: str,
    everydb2_connection_string: str,
    mlflow_run_id: str | None = None,
    mlflow_tracking_uri: str = "file:///mlruns",
    use_ensemble: bool = False,
) -> PaperTradingConfig:
    """PaperTradingConfig を構築"""

def load_models(config: PaperTradingConfig, *, use_ensemble: bool = False) -> tuple[TrainedModelsV5, ModelInfo]:
    """MLflowから学習済みモデルをロード"""

def run_setup(config: PaperTradingConfig, models: TrainedModelsV5, store: ParquetStore, target_date: date) -> list[dict]:
    """レーススケジュールを構築して schedule.json に保存。スケジュールを返す。"""

def run_predict(config: PaperTradingConfig, models: TrainedModelsV5, store: ParquetStore, target_date: date) -> None:
    """特徴量生成 → 推論 → ベット保存 → Slack通知"""

def run_reconcile(config: PaperTradingConfig, store: ParquetStore, target_date: date) -> None:
    """結果照合 → ROI計算 → レポート"""

def send_slack(config: PaperTradingConfig, message: str) -> None:
    """Slack通知 (エラーはログ出力のみ)"""
```

#### `src/automation/race_day_orchestrator.py`

パイプライン統括クラス。`runner.py` の関数を呼び出す。

```python
class RaceDayOrchestrator:
    def __init__(self, config: PaperTradingConfig, updater: EveryDB2Updater, store: ParquetStore): ...

    def run_full(self, target_date: date, *, use_ensemble: bool = False) -> None:
        """Phase 1: EveryDB2更新 → モデルロード → Setup → Predict"""
        # 1. result = self.updater.run()
        #    if not result.success: send_slack(error); return
        # 2. models, info = load_models(config, use_ensemble=use_ensemble)
        # 3. schedule = run_setup(config, models, store, target_date)
        # 4. run_predict(config, models, store, target_date)

    def run_reconcile(self, target_date: date) -> None:
        """結果照合 → ROI計算 → レポート"""
        # run_reconcile(config, store, target_date)

    # run_auto() は Phase 2 で追加
```

#### `scripts/run_race_day.py`

CLI エントリポイント。

```python
# --mode full:      EveryDB2更新 → Setup → Predict
# --mode reconcile: 結果照合 → レポート
# --mode auto:      Phase 2（次回実装）
```

#### `scripts/run_paper_trading.py` (修正)

各モードの実装を `src/paper_trading/runner.py` に委譲する薄いCLIラッパーに変更。

```python
def _run_setup(args, config, models, store):
    from paper_trading.runner import run_setup
    run_setup(config, models, store, date.fromisoformat(args.date))

def _run_predict(args, config, models, store):
    from paper_trading.runner import run_predict
    run_predict(config, models, store, date.fromisoformat(args.date))

# ... 同様に _run_reconcile, _load_models, load_config も委譲
```

#### `config/automation.yaml`

```yaml
everydb2:
  exe_path: "C:/Program Files/EveryDB2/EveryDB2.exe"
  timeout_minutes: 10
  retry_count: 2
  retry_interval_seconds: 30

pipeline:
  watch_lead_minutes: 5
```

### 再利用する既存コード

| 関数 | 移行先 | 役割 |
|------|--------|------|
| `_run_setup()` | `runner.run_setup()` | レーススケジュール構築 |
| `_run_predict()` | `runner.run_predict()` | 特徴量生成→推論→通知 |
| `_run_reconcile()` | `runner.run_reconcile()` | 結果照合→ROI計算 |
| `_load_models()` | `runner.load_models()` | MLflowモデルロード |
| `load_config()` | `runner.load_config()` | 設定読み込み |
| `_send_slack()` | `runner.send_slack()` | Slack通知 |

**移行方針:** `scripts/run_paper_trading.py` のロジックを `src/paper_trading/runner.py` に抽出する。
`run_paper_trading.py` は薄いCLIラッパー（引数解析→runner関数呼び出し）になる。
これにより `argparse.Namespace` への依存を排除し、オーケストレータから型安全に呼び出せる。

### 依存する既存クラス

| クラス | ファイル | 役割 |
|--------|---------|------|
| `EveryDB2Queries` | `src/db/everydb2_queries.py` | PostgreSQL直読みクエリ |
| `PaperTradingConfig` | `src/paper_trading/config.py` | Paper Trading設定 |
| `ParquetStore` | `src/db/parquet_store.py` | Parquet読み書き |
| `FeatureEngine` | `src/features/feature_engine.py` | 特徴量生成 |
| `RacePredictor` | `src/backtest/race_predictor.py` | レース予測 |

### 前提条件

1. EveryDB2 GUI で更新設定が事前構成済み（蓄積系 + 今週データ + 時系列）
2. `EveryDB2.exe` のパスを `config/automation.yaml` に設定
3. 環境変数: `PGPASSWORD`, `SLACK_WEBHOOK_URL`
4. Parquet データが既に存在（ETL は別途実行済み）

### エラー処理

- **EveryDB2更新失敗**: リトライ(2回) → Slack通知 → プロセス終了
- **データ取得失敗**: ログ出力 → Slack通知 → 処理続行（空データとして扱う）
- **モデルロード失敗**: エラーログ → Slack通知 → プロセス終了
- **Slack通知失敗**: ログ出力のみ（処理は続行）

### テスト方針

- `EveryDB2Updater`: subprocessをモックしてテスト
- `RaceDayOrchestrator`: runner関数をモックしてテスト
- `runner.py`: 既存のテストパターンに従い、DB不要のmockベース
- `run_paper_trading.py`: CLIラッパー化後も既存テストが通ることを確認

## Phase 2 設計（概要のみ）

```bash
python scripts/run_race_day.py --mode auto --date 2026-04-11 --ensemble
```

```
[08:00] 起動
  → EveryDB2更新 → Setup → ベース特徴量計算
  ↓
[各レース発走-5分] レースループ
  → EveryDB2差分更新（最新オッズ）
  → オッズ特徴量再計算
  → 推論 → Slack通知
  ↓
[最終レース後+30分] リコンサイル
  → EveryDB2更新（結果データ）
  → 結果照合 → ROI計算 → レポート
```

Phase 2 では `RaceWatcher` の待機・リトライロジックを参考にしつつ、
`RaceDayOrchestrator.run_auto()` を追加実装する。

## 検証方法

1. **EveryDB2 CMDLINE 単体テスト**: 任意のタイミングでCMDLINEが正常終了することを確認
2. **Phase 1 手動テスト**: `run_race_day.py --mode full --date YYYYMMDD` が正常実行されること
3. **結果確認**: `data/paper_trading/predictions/YYYYMMDD.parquet` が生成されること
4. **Slack通知**: 通知が正常に送信されること
5. **既存テスト**: `run_paper_trading.py` のリファクタリング後も `pytest` が通ること
