# Paper Trading System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** バックテストでROI 143%のMLモデルをPaper Tradingで3ヶ月間モニタリングするシステムを構築する

**Architecture:** EveryDB2のPostgreSQLをデータソースとし、setup→watch→reconcileの3フェーズで日次運用。RacePredictorをBacktestEngineから抽出して共通化し、ModelLoaderでMLflowからTrainedModelsV5を再構築。Slack通知 + HTMLレポートでROI追跡。

**Tech Stack:** Python 3.11, LightGBM, MLflow, PostgreSQL (EveryDB2), Parquet, Slack Incoming Webhook, Jinja2

---

## 重要な設計上の注意点

### MLflow ロギングの拡張が必要

現在の `_log_to_mlflow()` は以下のモデルを**保存していない**。Paper Trading用に拡張が必要：

| モデル | 保存方法 | 理由 |
|--------|----------|------|
| `MarketModel.model` | `mlflow.lightgbm.log_model()` | 推論パイプライン必須 |
| `PlaceAbilityModel._calibrated` | `joblib.dump()` + `mlflow.log_artifact()` | sklearn CalibratedClassifierCV (非LightGBM) |
| `WideTwoStageModel.hit/return_model` | `mlflow.lightgbm.log_model()` | ワイドベット用 |
| `RobustConfidenceEstimator` | `mlflow.log_dict()` (JSON) | キャリブレーション値4つ (float) |
| `RaceQualityScreener.threshold` | `mlflow.log_param()` | 閾値 (float) |

拡張後、`run_train.py`で再学習が必要（既存runには新artifactがないため）。

### EveryDB2 テーブル名の未確定

`s_` プレフィックスの速報系テーブル名は設計書で「要確認」となっている。実装前にEveryDB2インスタンスで実際のテーブル名を確認すること。`EveryDB2Queries` はインタフェース先行で実装し、テーブル名はconfigで外部化する。

---

## Phase 1: Foundation

### Task 1: PaperTradingConfig

**Files:**
- Create: `src/paper_trading/__init__.py`
- Create: `src/paper_trading/config.py`
- Test: `tests/test_paper_trading_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paper_trading_config.py
"""PaperTradingConfig のテスト"""

import os
from pathlib import Path

import pytest


class TestPaperTradingConfig:
    def test_default_values(self) -> None:
        from paper_trading.config import PaperTradingConfig

        cfg = PaperTradingConfig(
            slack_webhook_url="https://hooks.slack.com/test",
            everydb2_connection_string="postgresql://localhost/everydb2",
        )
        assert cfg.ev_threshold == 1.0
        assert cfg.initial_bankroll == 100000.0
        assert cfg.stake == 100.0
        assert cfg.watch_lead_minutes == 5
        assert cfg.retry_count == 3
        assert cfg.mlflow_run_id is None

    def test_paper_trading_dir_default(self) -> None:
        from paper_trading.config import PaperTradingConfig

        cfg = PaperTradingConfig(
            slack_webhook_url="https://hooks.slack.com/test",
            everydb2_connection_string="postgresql://localhost/everydb2",
        )
        assert cfg.paper_trading_dir == Path("data/paper_trading")

    def test_custom_values(self) -> None:
        from paper_trading.config import PaperTradingConfig

        cfg = PaperTradingConfig(
            slack_webhook_url="https://hooks.slack.com/test",
            everydb2_connection_string="postgresql://localhost/everydb2",
            ev_threshold=1.3,
            initial_bankroll=200000.0,
            stake=200.0,
            mlflow_run_id="abc123",
        )
        assert cfg.ev_threshold == 1.3
        assert cfg.initial_bankroll == 200000.0
        assert cfg.stake == 200.0
        assert cfg.mlflow_run_id == "abc123"

    def test_data_dir_structure(self, tmp_path: Path) -> None:
        from paper_trading.config import PaperTradingConfig

        cfg = PaperTradingConfig(
            slack_webhook_url="https://hooks.slack.com/test",
            everydb2_connection_string="postgresql://localhost/everydb2",
            paper_trading_dir=tmp_path / "pt",
        )
        dirs = cfg.ensure_dirs()
        assert (dirs["predictions"]).exists()
        assert (dirs["bets"]).parent == cfg.paper_trading_dir
        assert (dirs["daily_summary"]).exists()
        assert (dirs["dry_run"]).exists()
        assert (dirs["model"]).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_paper_trading_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'paper_trading'`

- [ ] **Step 3: Write implementation**

```python
# src/paper_trading/__init__.py
```

```python
# src/paper_trading/config.py
"""Paper Trading 設定"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PaperTradingConfig:
    """Paper Trading の全設定"""

    # Slack
    slack_webhook_url: str

    # EveryDB2
    everydb2_connection_string: str

    # モデル
    mlflow_run_id: str | None = None
    mlflow_tracking_uri: str = "file:///mlruns"

    # ベット
    ev_threshold: float = 1.0
    initial_bankroll: float = 100000.0
    stake: float = 100.0

    # タイミング
    watch_lead_minutes: int = 5
    retry_count: int = 3
    retry_interval_seconds: int = 60

    # EveryDB2 クエリ
    query_timeout_seconds: int = 30

    # パス
    paper_trading_dir: Path = Path("data/paper_trading")

    def ensure_dirs(self) -> dict[str, Path]:
        """必要なディレクトリを作成してパスを返す"""
        dirs = {
            "predictions": self.paper_trading_dir / "predictions",
            "daily_summary": self.paper_trading_dir / "daily_summary",
            "dry_run": self.paper_trading_dir / "dry_run",
            "model": self.paper_trading_dir / "model",
            "bets": self.paper_trading_dir,
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_paper_trading_config.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/paper_trading/__init__.py src/paper_trading/config.py tests/test_paper_trading_config.py
git commit -m "feat: PaperTradingConfig データクラスを追加"
```

---

### Task 2: SlackNotifier

**Files:**
- Modify: `src/monitoring/notifier.py`
- Test: `tests/test_slack_notifier.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_slack_notifier.py
"""SlackNotifier のテスト"""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestSlackNotifier:
    def test_send_calls_webhook(self) -> None:
        from monitoring.notifier import SlackNotifier

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        with patch.object(notifier, "_post") as mock_post:
            mock_post.return_value = True
            result = notifier.send("test message", level="info")
            assert result is True
            mock_post.assert_called_once()

    def test_send_prediction_formats_bets(self) -> None:
        from monitoring.notifier import SlackNotifier

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        with patch.object(notifier, "_post") as mock_post:
            mock_post.return_value = True
            bets = [
                {"race_id": "2026040510010101", "umaban": 3, "horse_name": "テスト馬",
                 "odds": 2.4, "ev": 1.5, "stake": 100.0},
            ]
            notifier.send_prediction(bets=bets, date="2026-04-05")
            call_args = mock_post.call_args
            assert "テスト馬" in call_args[0][0]

    def test_send_daily_result(self) -> None:
        from monitoring.notifier import SlackNotifier

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        with patch.object(notifier, "_post") as mock_post:
            mock_post.return_value = True
            summary = {
                "date": "2026-04-05",
                "n_bets": 5,
                "n_wins": 2,
                "daily_roi": 1.20,
                "cumulative_roi": 1.10,
                "max_dd": 0.03,
                "bankroll": 101500.0,
            }
            notifier.send_daily_result(summary=summary)
            mock_post.assert_called_once()

    def test_send_returns_false_on_error(self) -> None:
        from monitoring.notifier import SlackNotifier

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        with patch.object(notifier, "_post", side_effect=Exception("network error")):
            result = notifier.send("test", level="warning")
            assert result is False

    def test_notifier_protocol_compliance(self) -> None:
        from monitoring.notifier import NotifierProtocol, SlackNotifier

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        assert isinstance(notifier, NotifierProtocol)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_slack_notifier.py -v`
Expected: FAIL — `ImportError: cannot import name 'SlackNotifier'`

- [ ] **Step 3: Write implementation**

`src/monitoring/notifier.py` に以下を追加（既存クラスは変更しない）:

```python
# --- 追加 imports ---
import json
import urllib.request
import urllib.error
from datetime import date as date_type
from typing import Any


# --- 追加クラス（ファイル末尾） ---

class SlackNotifier:
    """Slack Incoming Webhook 通知 (NotifierProtocol 準拠 + Paper Trading専用メソッド)"""

    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    def send(self, message: str, level: str = "info") -> bool:
        """NotifierProtocol 準拠: 汎用メッセージ送信"""
        payload = {"text": f"[{level.upper()}] {message}"}
        return self._post(payload)

    def send_prediction(self, bets: list[dict[str, Any]], date: str) -> bool:
        """ベット推薦を Slack に通知"""
        if not bets:
            return True
        lines = [f"*Paper Trading 予測 — {date}*\n"]
        for b in bets[:10]:  # 最大10件
            lines.append(
                f"  #{b['umaban']} {b.get('horse_name', '?')} "
                f"オッズ={b['odds']:.1f} EV={b['ev']:.2f}"
            )
        if len(bets) > 10:
            lines.append(f"  ...他 {len(bets) - 10} 件")
        return self._post({"text": "\n".join(lines)})

    def send_daily_result(self, summary: dict[str, Any]) -> bool:
        """日次サマリーを通知"""
        lines = [
            f"*Paper Trading サマリー — {summary['date']}*",
            f"  ベット数: {summary['n_bets']} / 的中: {summary['n_wins']}",
            f"  日次ROI: {summary['daily_roi']:.1%}",
            f"  累積ROI: {summary['cumulative_roi']:.1%}",
            f"  Max DD: {summary['max_dd']:.1%}",
            f"  資金: ¥{summary['bankroll']:,.0f}",
        ]
        return self._post({"text": "\n".join(lines)})

    def _post(self, payload: dict[str, Any]) -> bool:
        """Slack Webhook に POST"""
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception:
            logger.exception("Slack通知失敗")
            return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_slack_notifier.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run existing notifier tests to ensure no regression**

Run: `python -m pytest tests/test_notifier.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/monitoring/notifier.py tests/test_slack_notifier.py
git commit -m "feat: SlackNotifier を追加 (NotifierProtocol準拠)"
```

---

### Task 3: Extend MLflow Logging

現在の `_log_to_mlflow()` は MarketModel, PlaceAbilityModel, WideTwoStageModel, RobustConfidenceEstimator を保存していない。Paper Trading の ModelLoader のために全モデルを保存するよう拡張する。

**Files:**
- Modify: `src/pipelines/training_pipeline.py`
- Test: `tests/test_mlflow_logging.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mlflow_logging.py
"""MLflow ロギング拡張のテスト"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestExtendedMLflowLogging:
    """_log_to_mlflow が全モデルを保存することを確認"""

    def test_market_model_logged_per_surface(self) -> None:
        """MarketModel が各surfaceごとにログされる"""
        from pipelines.training_pipeline import TrainingPipelineV5

        # _log_to_mlflow が market_turf, market_dirt を呼び出すことを確認
        # (統合テストではなく、mlflow.lightgbm.log_model の呼び出しを検証)
        with patch("pipelines.training_pipeline.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value.__enter__ = MagicMock()
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            mock_sub = MagicMock()
            mock_sub.market.model = MagicMock()
            mock_sub.stage1.models = {"turf": MagicMock()}
            mock_sub.win.hit_model = MagicMock()
            mock_sub.win.return_model = MagicMock()
            mock_sub.ev_corrector.p_correction_model = MagicMock()
            mock_sub.ev_corrector.e_correction_model = MagicMock()
            mock_sub.place.hit_model = MagicMock()
            mock_sub.place.return_model = MagicMock()
            mock_sub.place_ability._model = MagicMock()
            mock_sub.place_ability._calibrated = MagicMock()
            mock_sub.wide.hit_model = MagicMock()
            mock_sub.wide.return_model = MagicMock()

            mock_quality = MagicMock()
            mock_quality.model = MagicMock()
            mock_quality.threshold = 0.42
            mock_regime = MagicMock()
            mock_regime.model = MagicMock()

            mock_confidence = MagicMock()
            mock_confidence.alpha = 0.1
            mock_confidence.rolling_window = 200
            mock_confidence._calibrated = True
            mock_confidence._win_cp_quantile = 0.05
            mock_confidence._place_cp_quantile = 0.08
            mock_confidence._win_rolling_quantile = 0.06
            mock_confidence._place_rolling_quantile = 0.09

            pipeline = TrainingPipelineV5.__new__(TrainingPipelineV5)
            pipeline._log_to_mlflow(
                models={"turf": mock_sub, "dirt": mock_sub},
                quality_screen=mock_quality,
                regime_det=mock_regime,
                train_end="2024-12-31",
            )

            # market_turf が呼ばれる
            log_model_calls = [
                c[0][1] for c in mock_mlflow.lightgbm.log_model.call_args_list
            ]
            assert "market_turf" in log_model_calls
            assert "market_dirt" in log_model_calls
            # wide もログされる
            assert "wide_hit_turf" in log_model_calls
            assert "wide_ret_turf" in log_model_calls
            # confidence params がログされる
            mock_mlflow.log_dict.assert_called()
            # quality threshold がログされる
            log_param_calls = [c[0][0] for c in mock_mlflow.log_param.call_args_list]
            assert "quality_threshold" in log_param_calls

    def test_place_ability_saved_as_artifact(self) -> None:
        """PlaceAbilityModel が joblib artifact として保存される"""
        with patch("pipelines.training_pipeline.mlflow") as mock_mlflow, \
             patch("pipelines.training_pipeline.joblib") as mock_joblib, \
             patch("pipelines.training_pipeline.tempfile") as mock_tempfile:

            mock_mlflow.start_run.return_value.__enter__ = MagicMock()
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            mock_tmp = MagicMock()
            mock_tmp.name = "/tmp/place_ability_turf.joblib"
            mock_tempfile.NamedTemporaryFile.return_value = mock_tmp

            mock_sub = MagicMock()
            mock_sub.market.model = MagicMock()
            mock_sub.stage1.models = {"turf": MagicMock()}
            mock_sub.win.hit_model = MagicMock()
            mock_sub.win.return_model = MagicMock()
            mock_sub.ev_corrector.p_correction_model = MagicMock()
            mock_sub.ev_corrector.e_correction_model = MagicMock()
            mock_sub.place.hit_model = MagicMock()
            mock_sub.place.return_model = MagicMock()
            mock_sub.place_ability._model = MagicMock()
            mock_sub.place_ability._calibrated = MagicMock()
            mock_sub.wide.hit_model = MagicMock()
            mock_sub.wide.return_model = MagicMock()

            mock_quality = MagicMock()
            mock_quality.model = MagicMock()
            mock_quality.threshold = 0.42
            mock_regime = MagicMock()
            mock_regime.model = MagicMock()

            pipeline = TrainingPipelineV5.__new__(TrainingPipelineV5)
            pipeline._log_to_mlflow(
                models={"turf": mock_sub},
                quality_screen=mock_quality,
                regime_det=mock_regime,
                train_end="2024-12-31",
            )

            mock_joblib.dump.assert_called()
            mock_mlflow.log_artifact.assert_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mlflow_logging.py -v`
Expected: FAIL — `market_turf` not in log_model_calls

- [ ] **Step 3: Modify `_log_to_mlflow` in `src/pipelines/training_pipeline.py`**

既存の `_log_to_mlflow` メソッド (lines 417-443) を以下で置き換える。`joblib` と `tempfile` をファイル先頭の imports に追加:

```python
# ファイル先頭に追加
import joblib
import tempfile
```

```python
    def _log_to_mlflow(
        self,
        models: dict[str, SubmodelSet],
        quality_screen: RaceQualityScreener,
        regime_det: RegimeDetector,
        train_end: str,
    ) -> None:
        """MLflow に全モデルとメトリクスを記録 (Paper Trading対応)"""
        with mlflow.start_run(run_name=f"v5.5_{train_end}"):
            for surface, sub in models.items():
                # Stage1 (AbilityModel per surface)
                stage1_model = sub.stage1.models.get(surface)
                if stage1_model is not None:
                    mlflow.lightgbm.log_model(stage1_model, f"stage1_{surface}")

                # MarketModel
                mlflow.lightgbm.log_model(sub.market.model, f"market_{surface}")

                # WinTwoStageModel
                mlflow.lightgbm.log_model(sub.win.hit_model, f"win_hit_{surface}")
                mlflow.lightgbm.log_model(sub.win.return_model, f"win_ret_{surface}")

                # EVCorrectionModel
                mlflow.lightgbm.log_model(
                    sub.ev_corrector.p_correction_model, f"ev_corrector_p_{surface}"
                )
                mlflow.lightgbm.log_model(
                    sub.ev_corrector.e_correction_model, f"ev_corrector_e_{surface}"
                )

                # PlaceTwoStageModel
                mlflow.lightgbm.log_model(sub.place.hit_model, f"place_hit_{surface}")
                mlflow.lightgbm.log_model(sub.place.return_model, f"place_ret_{surface}")

                # PlaceAbilityModel (sklearn CalibratedClassifierCV → joblib)
                calibrated = sub.place_ability._calibrated or sub.place_ability._model
                if calibrated is not None:
                    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                        joblib.dump(calibrated, f.name)
                        mlflow.log_artifact(f.name, f"place_ability_{surface}")

                # WideTwoStageModel
                mlflow.lightgbm.log_model(sub.wide.hit_model, f"wide_hit_{surface}")
                mlflow.lightgbm.log_model(sub.wide.return_model, f"wide_ret_{surface}")

            # RaceQualityScreener
            mlflow.lightgbm.log_model(quality_screen.model, "race_quality")
            mlflow.log_param("quality_threshold", quality_screen.threshold)

            # RegimeDetector
            mlflow.lightgbm.log_model(regime_det.model, "regime_detector")

            # RobustConfidenceEstimator キャリブレーション値 (JSON)
            # SubmodelSet.confidence に含まれる最初の surface の値を保存
            first_sub = next(iter(models.values()))
            conf = first_sub.confidence
            if hasattr(conf, "_calibrated") and conf._calibrated:
                conf_params = {
                    "alpha": conf.alpha,
                    "rolling_window": conf.rolling_window,
                    "win_cp_quantile": conf._win_cp_quantile,
                    "place_cp_quantile": conf._place_cp_quantile,
                    "win_rolling_quantile": conf._win_rolling_quantile,
                    "place_rolling_quantile": conf._place_rolling_quantile,
                }
                import io
                buf = io.StringIO()
                json.dump(conf_params, buf)
                mlflow.log_dict(conf_params, "confidence_params.json")

            mlflow.log_param("train_end", train_end)
            mlflow.log_param("n_surfaces", str(len(models)))
            mlflow.log_param("pipeline_version", "v5.5")
```

**注意**: `_log_to_mlflow` のシグネチャに `train_start: str` パラメータを追加し、`mlflow.log_param("train_start", train_start)` も記録すること。`TrainingPipelineV5.run()` の呼び出し箇所で `train_start` を渡すよう変更が必要。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mlflow_logging.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run existing tests to ensure no regression**

Run: `python -m pytest tests/ -v -k "not slow"`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/pipelines/training_pipeline.py tests/test_mlflow_logging.py
git commit -m "feat: MLflow ロギングを拡張 (MarketModel, PlaceAbility, Wide, QualityThreshold)"
```

---

### Task 4: ModelLoader

MLflow から全モデルをロードして `TrainedModelsV5` を再構築する。

**Files:**
- Create: `src/db/model_loader.py`
- Test: `tests/test_model_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_loader.py
"""ModelLoader のテスト"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domain.models import SubmodelSet, TrainedModelsV5


class TestModelInfo:
    def test_model_info_fields(self) -> None:
        from db.model_loader import ModelInfo

        info = ModelInfo(
            mlflow_run_id="abc123",
            train_start="2020-01-01",
            train_end="2023-12-31",
            loaded_at="2026-04-01 00:00:00",
        )
        assert info.mlflow_run_id == "abc123"


class TestModelLoader:
    def _mock_mlflow_run(self) -> MagicMock:
        """mlflow.load_model をモックして全モデルを返す"""
        mock_booster = MagicMock()
        with patch("db.model_loader.mlflow") as mock_mlflow:
            # 全ての mlflow.lightgbm.load_model が mock_booster を返す
            mock_mlflow.lightgbm.load_model.return_value = mock_booster
            # mlflow.artifacts.download_artifacts でディレクトリを返す
            mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/artifacts"
            # run の情報
            mock_run = MagicMock()
            mock_run.data.params = {
                "train_end": "2023-12-31",
                "quality_threshold": "0.42",
            }
            mock_mlflow.get_run.return_value = mock_run
            # mlflow.search_runs で最新runを返す
            mock_df = MagicMock()
            mock_df.iloc = MagicMock()
            mock_df.iloc.__getitem__ = MagicMock(return_value="latest_run_id")
            mock_mlflow.search_runs.return_value = mock_df
        return mock_mlflow

    @patch("db.model_loader.joblib")
    @patch("db.model_loader.mlflow")
    def test_load_returns_trained_models(self, mock_mlflow: MagicMock, mock_joblib: MagicMock) -> None:
        from db.model_loader import ModelLoader

        mock_booster = MagicMock()
        mock_mlflow.lightgbm.load_model.return_value = mock_booster
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/artifacts"
        mock_run = MagicMock()
        mock_run.data.params = {"train_end": "2023-12-31", "quality_threshold": "0.42"}
        mock_mlflow.get_run.return_value = mock_run
        mock_mlflow.search_runs.return_value = MagicMock()

        mock_joblib.load.return_value = MagicMock()

        loader = ModelLoader(tracking_uri="file:///mlruns")
        models, info = loader.load(run_id="test_run")

        assert isinstance(models, TrainedModelsV5)
        assert "turf" in models.submodels
        assert "dirt" in models.submodels
        assert info.mlflow_run_id == "test_run"

    @patch("db.model_loader.mlflow")
    def test_load_uses_latest_run_when_no_run_id(self, mock_mlflow: MagicMock) -> None:
        from db.model_loader import ModelLoader

        mock_booster = MagicMock()
        mock_mlflow.lightgbm.load_model.return_value = mock_booster
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/artifacts"
        mock_run = MagicMock()
        mock_run.data.params = {"train_end": "2023-12-31", "quality_threshold": "0.42"}
        mock_mlflow.get_run.return_value = mock_run

        mock_df = MagicMock()
        mock_df.sort_values.return_value = mock_df
        mock_df.iloc = MagicMock()
        mock_df.iloc.__getitem__ = MagicMock(return_value="latest_run_id")
        mock_mlflow.search_runs.return_value = mock_df

        with patch("db.model_loader.joblib", MagicMock()):
            loader = ModelLoader(tracking_uri="file:///mlruns")
            models, info = loader.load()

            mock_mlflow.search_runs.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'db.model_loader'`

- [ ] **Step 3: Write implementation**

```python
# src/db/model_loader.py
"""MLflow から TrainedModelsV5 をロードする"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import joblib
import mlflow

if TYPE_CHECKING:
    from domain.models import TrainedModelsV5

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """ロードしたモデルのメタ情報"""

    mlflow_run_id: str
    train_start: str
    train_end: str
    loaded_at: str


class ModelLoader:
    """MLflow から TrainedModelsV5 を構築してロード"""

    def __init__(self, tracking_uri: str = "file:///mlruns") -> None:
        mlflow.set_tracking_uri(tracking_uri)

    def load(self, run_id: str | None = None) -> tuple[TrainedModelsV5, ModelInfo]:
        """MLflow から学習済みモデルを読み込み、TrainedModelsV5 を再構築。

        run_id 未指定時は最新の成功 run を使用。
        """
        if run_id is None:
            run_id = self._find_latest_run()

        run = mlflow.get_run(run_id)
        params = run.data.params
        train_end = params.get("train_end", "unknown")
        train_start = params.get("train_start", "2020-01-01")
        quality_threshold = float(params.get("quality_threshold", "0.0"))

        surfaces = ["turf", "dirt"]
        artifact_uri = mlflow.get_artifact_uri(run_id)

        from domain.models import SubmodelSet, TrainedModelsV5
        from models.ev_correction_model import EVCorrectionModel
        from models.market_model import MarketModel
        from models.place_ability_model import PlaceAbilityModel
        from models.race_quality_screener import RaceQualityScreener
        from models.regime_detector import RegimeDetector
        from models.robust_confidence_estimator import RobustConfidenceEstimator
        from models.stage1_ability_model import AbilityModel
        from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel
        from models.wide_two_stage_model import WideTwoStageModel

        submodels: dict[str, SubmodelSet] = {}
        for surface in surfaces:
            # MarketModel
            market = MarketModel()
            market.model = mlflow.lightgbm.load_model(f"{artifact_uri}/market_{surface}")

            # AbilityModel (per-surface booster)
            ability = AbilityModel()
            ability.models = {
                surface: mlflow.lightgbm.load_model(f"{artifact_uri}/stage1_{surface}")
            }

            # WinTwoStageModel
            win = WinTwoStageModel()
            win.hit_model = mlflow.lightgbm.load_model(f"{artifact_uri}/win_hit_{surface}")
            win.return_model = mlflow.lightgbm.load_model(f"{artifact_uri}/win_ret_{surface}")

            # EVCorrectionModel
            ev_corr = EVCorrectionModel()
            ev_corr.p_correction_model = mlflow.lightgbm.load_model(
                f"{artifact_uri}/ev_corrector_p_{surface}"
            )
            ev_corr.e_correction_model = mlflow.lightgbm.load_model(
                f"{artifact_uri}/ev_corrector_e_{surface}"
            )

            # PlaceTwoStageModel
            place = PlaceTwoStageModel()
            place.hit_model = mlflow.lightgbm.load_model(f"{artifact_uri}/place_hit_{surface}")
            place.return_model = mlflow.lightgbm.load_model(f"{artifact_uri}/place_ret_{surface}")

            # PlaceAbilityModel (joblib artifact)
            pa = PlaceAbilityModel()
            pa_dir = mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/place_ability_{surface}"
            )
            pa_files = list(Path(pa_dir).glob("*.joblib"))
            if pa_files:
                pa._calibrated = joblib.load(pa_files[0])
            else:
                logger.warning("PlaceAbilityModel artifact not found for %s", surface)

            # WideTwoStageModel
            wide = WideTwoStageModel()
            wide.hit_model = mlflow.lightgbm.load_model(f"{artifact_uri}/wide_hit_{surface}")
            wide.return_model = mlflow.lightgbm.load_model(f"{artifact_uri}/wide_ret_{surface}")

            # RobustConfidenceEstimator (JSON params)
            confidence = RobustConfidenceEstimator()
            try:
                conf_path = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/confidence_params.json"
                )
                with open(conf_path) as f:
                    conf_data = json.load(f)
                confidence.alpha = conf_data["alpha"]
                confidence.rolling_window = conf_data["rolling_window"]
                confidence._win_cp_quantile = conf_data["win_cp_quantile"]
                confidence._place_cp_quantile = conf_data["place_cp_quantile"]
                confidence._win_rolling_quantile = conf_data["win_rolling_quantile"]
                confidence._place_rolling_quantile = conf_data["place_rolling_quantile"]
                confidence._calibrated = True
            except Exception:
                logger.warning("RobustConfidenceEstimator params not found, using defaults")

            submodels[surface] = SubmodelSet(
                market=market,
                stage1=ability,
                place_ability=pa,
                win=win,
                ev_corrector=ev_corr,
                place=place,
                wide=wide,
                confidence=confidence,
            )

        # RaceQualityScreener
        quality = RaceQualityScreener()
        quality.model = mlflow.lightgbm.load_model(f"{artifact_uri}/race_quality")
        quality.threshold = quality_threshold

        # RegimeDetector
        regime = RegimeDetector()
        regime.model = mlflow.lightgbm.load_model(f"{artifact_uri}/regime_detector")

        models = TrainedModelsV5(
            submodels=submodels,
            quality_screener=quality,
            regime_detector=regime,
            train_period=("2020-01-01", train_end),
        )

        info = ModelInfo(
            mlflow_run_id=run_id,
            train_start=train_start,
            train_end=train_end,
            loaded_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        return models, info

    def _find_latest_run(self) -> str:
        """最新の成功 run ID を取得"""
        from mlflow.entities import ViewType

        df = mlflow.search_runs(
            order_by=["start_time DESC"],
            max_results=1,
            filter_string="status = 'FINISHED'",
            view_type=ViewType.ACTIVE_ONLY,
        )
        if df.empty:
            raise ValueError("No successful MLflow runs found")
        return str(df.iloc[0]["run_id"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/model_loader.py tests/test_model_loader.py
git commit -m "feat: ModelLoader — MLflowからTrainedModelsV5を再構築"
```

---

## Phase 2: RacePredictor Extraction

### Task 5: RacePredictor

BacktestEngine のレース別推論ループ (~110行) を共通コンポーネントとして抽出する。

**Files:**
- Create: `src/backtest/race_predictor.py`
- Test: `tests/test_race_predictor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_race_predictor.py
"""RacePredictor のテスト"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from domain.models import SubmodelSet, TrainedModelsV5
from domain.types import RegimeState


@pytest.fixture
def mock_models() -> MagicMock:
    models = MagicMock(spec=TrainedModelsV5)
    models.submodels = {"turf": MagicMock(spec=SubmodelSet)}
    models.quality_screener = MagicMock()
    models.quality_screener.should_bet.return_value = True
    models.regime_detector = MagicMock()
    models.regime_detector.current_regime = RegimeState.CONSERVATIVE
    models.regime_detector.get_strategy_params.return_value = {
        "ev_threshold": 1.20,
        "max_bets_per_race": 3,
    }
    return models


class TestRacePredictor:
    def test_predict_returns_dataframe_with_ev_columns(
        self, mock_models: MagicMock
    ) -> None:
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)

        race_df = pd.DataFrame({
            "race_id": ["20240101010101"],
            "umaban": [1],
            "surface_key": ["turf"],
            "surface": ["turf"],
            "distance": [1200],
            "distance_bin": ["sprint"],
            "popularity_rank": [3],
            "ninki": [3],
            "place_odds_actual": [2.4],
            "finish_pos": [2],
            "ketto_num": [1234],
            "win_odds": [5.0],
            "ba_taijyu": [480],
            "field_size": [10],
            "track_condition_code": [2],
            "grade_code": ["C"],
        })

        submodel = mock_models.submodels["turf"]
        submodel.market.predict_and_calc_error.return_value = race_df.copy()
        submodel.stage1.add_ability_probs.return_value = race_df.copy()
        submodel.place_ability.predict.return_value = race_df.copy()
        submodel.win.predict_ev.return_value = race_df.copy()
        submodel.ev_corrector.correct_ev.return_value = race_df.copy()
        submodel.place.predict_ev.return_value = race_df.copy()
        submodel.confidence.predict_lower_bound.return_value = (
            race_df.copy(),
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )

        result = predictor.predict(race_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_predict_skips_unknown_surface(self, mock_models: MagicMock) -> None:
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)

        race_df = pd.DataFrame({
            "race_id": ["20240101010101"],
            "umaban": [1],
            "surface_key": ["unknown"],
            "surface": ["unknown"],
            "distance": [1200],
            "distance_bin": ["sprint"],
            "popularity_rank": [3],
            "ninki": [3],
            "place_odds_actual": [2.4],
            "finish_pos": [2],
            "ketto_num": [1234],
            "win_odds": [5.0],
            "ba_taijyu": [480],
        })

        result = predictor.predict(race_df)
        assert result.empty

    def test_select_bets_returns_list(self, mock_models: MagicMock) -> None:
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)

        race_df = pd.DataFrame({
            "race_id": ["20240101010101"] * 3,
            "umaban": [1, 2, 3],
            "surface_key": ["turf"] * 3,
            "surface": ["turf"] * 3,
            "distance": [1200] * 3,
            "distance_bin": ["sprint"] * 3,
            "popularity_rank": [3, 5, 7],
            "ninki": [3, 5, 7],
            "place_odds_actual": [2.4, 1.5, 5.0],
            "ev_place": [1.5, 0.8, 1.8],
            "finish_pos": [2, 1, 3],
            "ketto_num": [1234, 5678, 9012],
            "win_odds": [5.0, 2.0, 10.0],
            "ba_taijyu": [480, 470, 490],
        })

        bets = predictor.select_bets(race_df, bankroll=100000.0)
        assert isinstance(bets, list)
        assert len(bets) >= 1
        assert all(b.stake == 100.0 for b in bets)

    def test_build_race_features(self, mock_models: MagicMock) -> None:
        from backtest.race_predictor import RacePredictor

        race_df = pd.DataFrame({
            "race_id": ["20240101010101"] * 2,
            "umaban": [1, 2],
            "surface": ["turf"] * 2,
            "distance_bin": ["sprint"] * 2,
            "track_condition_code": [2] * 2,
            "grade_code": ["C"] * 2,
            "field_size": [10] * 2,
            "difficulty_score": [0.5] * 2,
            "signed_log_error_win": [0.1, -0.2],
            "abs_log_error_win": [0.1, 0.2],
            "market_entropy": [2.0] * 2,
            "overround": [0.2] * 2,
        })

        features = RacePredictor.build_race_features(race_df)
        assert isinstance(features, dict)
        assert features["surface"] == "turf"
        assert features["field_size"] == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_race_predictor.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/backtest/race_predictor.py
"""1レース分の推論パイプライン (BacktestEngine と PaperPredictor の共通コンポーネント)

BacktestEngine.run() のレース別ループ (4a-4g) を抽出。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from domain.models import Bet, BetType

if TYPE_CHECKING:
    from domain.models import TrainedModelsV5

logger = logging.getLogger(__name__)


class RacePredictor:
    """1レース分の特徴量→推論→ベット候補生成を担当する共通コンポーネント"""

    def __init__(self, models: TrainedModelsV5) -> None:
        self.models = models

    def predict(
        self,
        race_df: pd.DataFrame,
        hist_features: pd.DataFrame | None = None,
        jockey_features: pd.DataFrame | None = None,
        trainer_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """1レースの推論パイプラインを実行。

        Returns:
            推論結果列 (EV, ev_lower_corrected等) を追加した DataFrame。
            サーフェスが不明な場合は空 DataFrame を返す。
        """
        from features.horse_history_features import HorseHistoryFeatures
        from features.interaction_features import compute_interaction_features

        if race_df.empty:
            return race_df

        # 1. サブモデル選択
        surface_key = race_df["surface_key"].iloc[0]
        if surface_key not in self.models.submodels:
            logger.debug("Unknown surface: %s, skipping", surface_key)
            return pd.DataFrame()
        submodel = self.models.submodels[surface_key]

        df = race_df.copy()

        # 2. HorseHistoryFeatures マージ + race_transforms
        if hist_features is not None and not hist_features.empty:
            df = df.merge(hist_features, on=["race_id", "umaban"], how="left")
        df = HorseHistoryFeatures.add_race_transforms(df)

        # 3. interaction_features (kyakusitu_cd が必要なため HorseHistoryFeatures 後)
        df = compute_interaction_features(df)

        # 4. 推論チェーン
        try:
            df = submodel.market.predict_and_calc_error(df)
        except Exception as e:
            logger.debug("Market prediction failed: %s", e)
            return pd.DataFrame()
        df = submodel.stage1.add_ability_probs(df)
        df = submodel.place_ability.predict(df)
        df = submodel.win.predict_ev(df)

        # 5. 騎手/調教師コンテキスト マージ
        if jockey_features is not None and not jockey_features.empty:
            jockey_race = jockey_features[
                jockey_features["race_id"] == race_df["race_id"].iloc[0]
            ]
            df = df.merge(jockey_race, on=["race_id", "umaban"], how="left")
        if trainer_features is not None and not trainer_features.empty:
            trainer_race = trainer_features[
                trainer_features["race_id"] == race_df["race_id"].iloc[0]
            ]
            df = df.merge(trainer_race, on=["race_id", "umaban"], how="left")

        # 6. EV補正 + Place推論
        df = submodel.ev_corrector.correct_ev(df)
        df = submodel.place.predict_ev(df)

        if "ev_place_corrected" not in df.columns:
            df["ev_place_corrected"] = df.get("ev_place", 0.0)

        # 7. 信頼区間
        win_df, place_df = submodel.confidence.predict_lower_bound(df, df)
        df = win_df
        if "EV_lower_place" in place_df.columns:
            df["EV_lower_place"] = place_df["EV_lower_place"].values

        return df

    def should_bet(self, race_df: pd.DataFrame) -> bool:
        """RaceQualityScreener でベット対象か判定"""
        features = self.build_race_features(race_df)
        return self.models.quality_screener.should_bet(features)

    def select_bets(
        self,
        race_df: pd.DataFrame,
        bankroll: float,
    ) -> list[Bet]:
        """EV > 閾値 の馬をベット候補として抽出。

        BacktestEngine._generate_bets() と同じロジック。
        """
        regime = self.models.regime_detector.current_regime
        regime_params = self.models.regime_detector.get_strategy_params(regime)

        bets: list[Bet] = []
        ev_threshold = regime_params.get("ev_threshold", 1.20)
        max_bets = regime_params.get("max_bets_per_race", 3)

        if "ev_place" not in race_df.columns or "place_odds_actual" not in race_df.columns:
            return bets

        candidates = race_df[race_df["ev_place"].fillna(0) >= ev_threshold].copy()
        candidates = candidates.nlargest(max_bets, "ev_place")

        for _, row in candidates.iterrows():
            stake = 100.0
            if bankroll >= stake:
                bets.append(
                    Bet(
                        race_id=row["race_id"],
                        umaban=int(row["umaban"]),
                        bet_type=BetType.PLACE,
                        odds=float(row["place_odds_actual"]),
                        ev_lower_corrected=float(row.get("ev_place", 0)),
                        stake=stake,
                    )
                )

        return bets

    @staticmethod
    def build_race_features(race_df: pd.DataFrame) -> dict[str, Any]:
        """レースレベル特徴量を dict に変換 (QualityScreener 用)。

        BacktestEngine._build_race_features() から移行。
        """
        row = race_df.iloc[0]
        signed_error = (
            race_df["signed_log_error_win"]
            if "signed_log_error_win" in race_df.columns
            else pd.Series([0.0])
        )
        abs_error = (
            race_df["abs_log_error_win"]
            if "abs_log_error_win" in race_df.columns
            else pd.Series([0.0])
        )
        return {
            "surface": row.get("surface", "turf"),
            "distance_bin": row.get("distance_bin", "mile"),
            "track_condition_code": row.get("track_condition_code", 2),
            "grade_code": row.get("grade_code", "C"),
            "field_size": row.get("field_size", 10),
            "difficulty_score": row.get("difficulty_score", 0.5),
            "market_log_error_mean": float(signed_error.mean()),
            "market_log_error_std": float(signed_error.std()) if len(signed_error) > 1 else 0.0,
            "market_log_error_abs_mean": float(abs_error.mean()),
            "market_log_error_max_abs": float(abs_error.max()) if len(abs_error) > 0 else 0.0,
            "market_log_error_top_q75": float(abs_error.quantile(0.75))
            if len(abs_error) > 1
            else 0.0,
            "n_positive_errors": int((signed_error > 0).sum()),
            "top_k_error_sum": float(signed_error.nlargest(3).sum())
            if len(signed_error) >= 3
            else 0.0,
            "positive_error_ratio": float((signed_error > 0).sum()) / max(len(signed_error), 1),
            "market_entropy": row.get("market_entropy", 2.0),
            "overround": row.get("overround", 0.20),
            "overround_deviation": 0.0,
            "hist_hit_rate_topk": row.get("hist_hit_rate_topk", 0.3),
            "hist_roi_topk": row.get("hist_roi_topk", 1.0),
            "hist_positive_return_ratio": row.get("hist_positive_return_ratio", 0.3),
            "hist_win_rate_same_condition": row.get("hist_hit_rate_topk", 0.3),
            "hist_market_entropy_avg": row.get("market_entropy", 2.0),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_race_predictor.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/backtest/race_predictor.py tests/test_race_predictor.py
git commit -m "feat: RacePredictor を抽出 (BacktestEngine と PaperPredictor の共通化)"
```

---

### Task 6: BacktestEngine Refactor

BacktestEngine の推論ループを RacePredictor に委譲。既存テストが全て PASS することを確認。

**Files:**
- Modify: `src/backtest/engine.py`
- Test: `tests/test_backtest_engine.py` (既存、変更なしで PASS)

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `python -m pytest tests/test_backtest_engine.py -v`
Expected: PASS (全テスト)

- [ ] **Step 2: Refactor BacktestEngine.run() to use RacePredictor**

`engine.py` の `run()` メソッド内で、レース別ループ (lines 132-251) を RacePredictor に委譲:

**変更点:**
1. `__init__` に `self._race_predictor = RacePredictor(models)` を追加
2. `run()` のステップ 4 (レースごとシミュレーション) を RacePredictor に委譲
3. `_build_race_features()` と `_generate_bets()` を RacePredictor に移行済み (delegated)
4. `_settle_bet()` は BacktestEngine に残す (バックテスト固有)

```python
# engine.py の変更

# import 追加
from backtest.race_predictor import RacePredictor

class BacktestEngine:
    def __init__(self, models: TrainedModelsV5, initial_bankroll: float = 100_000,
                 repo: DataRepository | None = None) -> None:
        self.models = models
        self.initial_bankroll = initial_bankroll
        self.repo = repo or DataRepository(ParquetStore())
        self._race_predictor = RacePredictor(models)  # NEW

    def run(self, test_start: str, test_end: str) -> BacktestResult:
        # ... ステップ 1-3 は変更なし ...

        # 4. レースごとにシミュレーション
        bankroll = self.initial_bankroll
        peak_bankroll = bankroll
        max_dd = 0.0
        bet_history: list[dict[str, Any]] = []
        monthly_returns: dict[str, float] = {}

        for race_id in race_ids:
            race_df_single = feat_df[feat_df["race_id"] == race_id].copy()
            if race_df_single.empty:
                continue

            # 事前計算済み特徴量をマージ
            hist_df_race = hist_df_all[hist_df_all["race_id"] == race_id]
            jockey_df_race = jockey_df_all[jockey_df_all["race_id"] == race_id]
            trainer_df_race = trainer_df_all[trainer_df_all["race_id"] == race_id]

            # RacePredictor に委譲
            result_df = self._race_predictor.predict(
                race_df_single,
                hist_features=hist_df_race,
                jockey_features=jockey_df_race,
                trainer_features=trainer_df_race,
            )
            if result_df.empty:
                continue

            # Quality screening
            if not self._race_predictor.should_bet(result_df):
                continue

            # Bet generation
            surface_key = result_df["surface_key"].iloc[0]
            bets = self._race_predictor.select_bets(result_df, bankroll)

            # Settlement (BacktestEngine 固有)
            for bet in bets:
                bet_result = self._settle_bet(bet, result_df)
                bankroll -= bet.stake
                if bet_result > 0:
                    bankroll += bet_result

                horse_rows = result_df[result_df["umaban"] == bet.umaban]
                pop_val = (
                    horse_rows["popularity_rank"].iloc[0]
                    if not horse_rows.empty and "popularity_rank" in horse_rows.columns
                    else 0
                )

                bet_history.append({
                    "race_id": race_id,
                    "bet_type": bet.bet_type.value,
                    "umaban": bet.umaban,
                    "stake": bet.stake,
                    "odds": bet.odds,
                    "result": bet_result,
                    "surface": surface_key,
                    "distance": int(result_df["distance"].iloc[0])
                    if "distance" in result_df.columns else 0,
                    "ev": float(bet.ev_lower_corrected),
                    "popularity": int(pop_val),
                    "bankroll_after": round(bankroll, 2),
                })

                peak_bankroll = max(peak_bankroll, bankroll)
                dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                max_dd = max(max_dd, dd)

        # ... ステップ 5 は変更なし ...
```

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `python -m pytest tests/test_backtest_engine.py -v`
Expected: PASS — 全ての既存テストが通ること。特に `test_engine_populates_enriched_fields` が RacePredictor の委譲後も正しく動作することを確認。

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/engine.py
git commit -m "refactor: BacktestEngine の推論ループを RacePredictor に委譲"
```

---

## Phase 3: Paper Trading Core

### Task 7: EveryDB2Queries

EveryDB2のPostgreSQLテーブルへのクエリラッパー。インタフェース先行で実装し、テーブル名はconfig化。

**Files:**
- Create: `src/db/everydb2_queries.py`
- Test: `tests/test_everydb2_queries.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_everydb2_queries.py
"""EveryDB2Queries のテスト"""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestEveryDB2Queries:
    @patch("db.everydb2_queries.psycopg2")
    def test_get_race_schedule_returns_list(self, mock_psycopg2: MagicMock) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.description = [("race_id",), ("venue",), ("race_num",),
                                   ("post_time",), ("surface",), ("distance",)]
        mock_cursor.fetchall.return_value = [
            ("2026040510010101", "中山", 1, "10:05", "turf", 1200),
        ]

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        schedule = queries.get_race_schedule(date(2026, 4, 5))

        assert len(schedule) == 1
        assert schedule[0]["race_id"] == "2026040510010101"
        assert schedule[0]["venue"] == "中山"

    @patch("db.everydb2_queries.psycopg2")
    def test_get_race_schedule_returns_empty_on_non_racing_day(
        self, mock_psycopg2: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        schedule = queries.get_race_schedule(date(2026, 4, 7))  # 月曜
        assert schedule == []

    @patch("db.everydb2_queries.psycopg2")
    def test_get_race_results(self, mock_psycopg2: MagicMock) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.description = [
            ("race_id",), ("umaban",), ("finish_pos",), ("place_pay",),
            ("place_odds",), ("horse_name",),
        ]
        mock_cursor.fetchall.return_value = [
            ("2026040510010101", 3, 1, 240.0, 2.4, "テスト馬"),
        ]

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        results = queries.get_race_results(date(2026, 4, 5))

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1
        assert results.iloc[0]["umaban"] == 3

    @patch("db.everydb2_queries.psycopg2")
    def test_connection_error_raises(self, mock_psycopg2: MagicMock) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_psycopg2.connect.side_effect = Exception("Connection refused")

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        with pytest.raises(Exception, match="Connection refused"):
            queries.get_race_schedule(date(2026, 4, 5))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_everydb2_queries.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# src/db/everydb2_queries.py
"""EveryDB2 速報系テーブルへの直接クエリラッパー

実装開始前に EveryDB2 インスタンスで s_ プレフィックスのテーブル名と列構造を確認すること。
テーブル名はハードコードせず、将来の変更に対応できるようにする。
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
import psycopg2

logger = logging.getLogger(__name__)


class EveryDB2Queries:
    """EveryDB2 PostgreSQL テーブルへのクエリラッパー"""

    def __init__(self, connection_string: str, timeout_seconds: int = 30) -> None:
        self._connection_string = connection_string
        self._timeout = timeout_seconds

    def _connect(self) -> Any:
        """PostgreSQL に接続"""
        return psycopg2.connect(self._connection_string, connect_timeout=self._timeout)

    def _query(self, sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
        """SQL を実行して DataFrame を返す"""
        with self._connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
            return df

    def get_race_schedule(self, target_date: date) -> list[dict[str, Any]]:
        """当日のレーススケジュールを取得。

        蓄積系テーブル n_uma_race を使用（木曜以降に利用可能）。
        非開催日は空リストを返す。

        注意: 実際のテーブル名・列名は EveryDB2 インスタンスで確認が必要。
        """
        # TODO: 実際のテーブル名・列名を EveryDB2 で確認後に修正
        ymd = target_date.strftime("%Y%m%d")
        sql = """
            SELECT
                CAST(Year || MonthDay || JyoCD || Kaiji || Nichiji || RaceNum AS VARCHAR) as race_id,
                ' venue_name ' as venue,
                RaceNum as race_num,
                'HH:MM' as post_time,
                CASE WHEN TrackCD BETWEEN 10 AND 22 THEN 'turf' ELSE 'dirt' END as surface,
                Distance as distance
            FROM n_uma_race
            WHERE Year || MonthDay = %s
              AND TrackCD < 51
            ORDER BY JyoCD, RaceNum
        """
        try:
            df = self._query(sql, (ymd,))
        except Exception:
            logger.exception("Failed to get race schedule for %s", target_date)
            return []

        if df.empty:
            return []

        return df.to_dict("records")

    def get_horse_weights(self, race_id: str) -> pd.DataFrame | None:
        """速報馬体重を取得。

        発走約1時間前に EveryDB2 自動更新で反映される。
        テーブル名は要確認 (s_bataijyu 推定)。
        """
        # TODO: 実際のテーブル名を確認
        year = race_id[:4]
        month_day = race_id[4:8]
        sql = """
            SELECT umaban, bataijyu as weight
            FROM s_bataijyu
            WHERE Year || MonthDay = %s
        """
        try:
            df = self._query(sql, (year + month_day,))
            return df if not df.empty else None
        except Exception:
            logger.exception("Failed to get horse weights for %s", race_id)
            return None

    def get_latest_odds(self, race_id: str) -> pd.DataFrame | None:
        """最新の速報オッズを取得。

        テーブル名は要確認 (s_odds_tanpuku 推定)。
        """
        # TODO: 実際のテーブル名を確認
        year = race_id[:4]
        month_day = race_id[4:8]
        sql = """
            SELECT umaban, tan_odds, fuku_odds
            FROM s_odds_tanpuku
            WHERE Year || MonthDay = %s
        """
        try:
            df = self._query(sql, (year + month_day,))
            return df if not df.empty else None
        except Exception:
            logger.exception("Failed to get odds for %s", race_id)
            return None

    def get_race_results(self, target_date: date) -> pd.DataFrame:
        """レース結果・払戻を取得。

        蓄積系テーブル n_uma_race + n_harai を使用。
        reconcile は 18:30 実行のため確定データが利用可能。
        """
        # TODO: 実際のテーブル名・列名を確認
        ymd = target_date.strftime("%Y%m%d")
        sql = """
            SELECT
                CAST(Year || MonthDay || JyoCD || Kaiji || Nichiji || RaceNum AS VARCHAR) as race_id,
                Umaban as umaban,
                KakuteiJyunni as finish_pos,
                0.0 as place_pay,
                0.0 as place_odds,
                '' as horse_name
            FROM n_uma_race
            WHERE Year || MonthDay = %s
              AND TrackCD < 51
        """
        try:
            return self._query(sql, (ymd,))
        except Exception:
            logger.exception("Failed to get race results for %s", target_date)
            return pd.DataFrame()

    def get_track_condition(self, race_id: str) -> str | None:
        """天候馬場状態を取得。"""
        # TODO: 実際のテーブル名を確認
        year = race_id[:4]
        month_day = race_id[4:8]
        sql = """
            SELECT BabaCD as baba_cd, TenkoCD as tenko_cd
            FROM n_race
            WHERE Year || MonthDay = %s
        """
        try:
            df = self._query(sql, (year + month_day,))
            if df.empty:
                return None
            return str(df.iloc[0].get("baba_cd", ""))
        except Exception:
            logger.exception("Failed to get track condition for %s", race_id)
            return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_everydb2_queries.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/db/everydb2_queries.py tests/test_everydb2_queries.py
git commit -m "feat: EveryDB2Queries — PostgreSQL速報系テーブルのクエリラッパー"
```

---

### Task 8: PaperPredictor

Paper Trading の中核。setup (事前計算) と predict_race (当日推論) を担当。

**Files:**
- Create: `src/paper_trading/predictor.py`
- Test: `tests/test_paper_predictor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paper_predictor.py
"""PaperPredictor のテスト"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from domain.models import SubmodelSet, TrainedModelsV5
from domain.types import RegimeState


@pytest.fixture
def mock_models() -> MagicMock:
    models = MagicMock(spec=TrainedModelsV5)
    models.submodels = {"turf": MagicMock(spec=SubmodelSet)}
    models.quality_screener = MagicMock()
    models.quality_screener.should_bet.return_value = True
    models.regime_detector = MagicMock()
    models.regime_detector.current_regime = RegimeState.CONSERVATIVE
    models.regime_detector.get_strategy_params.return_value = {
        "ev_threshold": 1.20,
        "max_bets_per_race": 3,
    }
    return models


class TestPaperPredictor:
    @patch("paper_trading.predictor.TrainerContextFeatures")
    @patch("paper_trading.predictor.JockeyContextFeatures")
    @patch("paper_trading.predictor.HorseHistoryFeatures")
    @patch("paper_trading.predictor.SubModelManager")
    @patch("paper_trading.predictor.FeatureEngine")
    def test_setup_returns_race_schedule(
        self,
        mock_feat_cls: MagicMock,
        mock_submgr_cls: MagicMock,
        mock_hist_cls: MagicMock,
        mock_jockey_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_models: MagicMock,
        tmp_path: Path,
    ) -> None:
        from paper_trading.predictor import PaperPredictor

        mock_repo = MagicMock()
        mock_repo.load_races.return_value = pd.DataFrame({
            "race_id": ["2026040510010101"],
            "race_date": pd.to_datetime("2026-04-05"),
        })
        mock_repo.load_entries.return_value = pd.DataFrame({
            "race_id": ["2026040510010101"],
            "umaban": [1],
            "ketto_num": [1234],
        })
        mock_repo.load_odds_snapshots.return_value = pd.DataFrame()

        mock_feat = MagicMock()
        mock_feat_cls.return_value = mock_feat
        mock_feat.build_all.return_value = pd.DataFrame({
            "race_id": ["2026040510010101"],
            "umaban": [1],
            "surface_key": ["turf"],
            "distance": [1200],
            "finish_pos": [0],
            "win_odds": [0.0],
            "popularity_rank": [0],
            "ba_taijyu": [0],
            "ketto_num": [1234],
        })
        mock_submgr = MagicMock()
        mock_submgr_cls.return_value = mock_submgr
        mock_submgr.add_distance_band_features.return_value = mock_feat.build_all.return_value

        mock_hist = MagicMock()
        mock_hist_cls.return_value = mock_hist
        mock_hist.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        mock_jockey = MagicMock()
        mock_jockey_cls.return_value = mock_jockey
        mock_jockey.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        mock_everydb2 = MagicMock()
        mock_everydb2.get_race_schedule.return_value = [
            {"race_id": "2026040510010101", "venue": "中山", "race_num": 1,
             "post_time": "10:05", "surface": "turf", "distance": 1200,
             "horses": ["馬1"]},
        ]

        predictor = PaperPredictor(
            repo=mock_repo,
            race_predictor=MagicMock(),
            models=mock_models,
        )
        schedule = predictor.setup(date(2026, 4, 5), everydb2=mock_everydb2)

        assert schedule is not None
        assert len(schedule) == 1

    def test_predict_race_returns_bets(self, mock_models: MagicMock) -> None:
        from paper_trading.predictor import PaperPredictor

        mock_repo = MagicMock()
        mock_race_predictor = MagicMock()

        pre_computed = pd.DataFrame({
            "race_id": ["2026040510010101"] * 2,
            "umaban": [1, 2],
            "surface_key": ["turf"] * 2,
            "ev_place": [1.5, 0.8],
            "place_odds_actual": [2.4, 1.5],
        })
        horse_weights = pd.DataFrame({"umaban": [1, 2], "weight": [480, 470]})
        odds = pd.DataFrame({"umaban": [1, 2], "tan_odds": [5.0, 2.0],
                             "fuku_odds": [2.4, 1.5]})

        mock_race_predictor.predict.return_value = pre_computed
        mock_race_predictor.should_bet.return_value = True
        mock_race_predictor.select_bets.return_value = [
            MagicMock(race_id="2026040510010101", umaban=1, bet_type="place",
                      odds=2.4, ev_lower_corrected=1.5, stake=100.0),
        ]

        predictor = PaperPredictor(
            repo=mock_repo,
            race_predictor=mock_race_predictor,
            models=mock_models,
        )
        bets = predictor.predict_race(
            race_id="2026040510010101",
            pre_computed_features=pre_computed,
            horse_weights=horse_weights,
            odds=odds,
            bankroll=100000.0,
        )

        assert len(bets) == 1
        mock_race_predictor.predict.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_paper_predictor.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# src/paper_trading/predictor.py
"""Paper Trading 日次予測ロジック"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from backtest.race_predictor import RacePredictor
    from db.everydb2_queries import EveryDB2Queries
    from db.repository import DataRepository
    from domain.models import TrainedModelsV5

logger = logging.getLogger(__name__)


class PaperPredictor:
    """Paper Trading の予測コア。

    setup() で事前特徴量を生成し、predict_race() で当日データをマージして推論。
    """

    def __init__(
        self,
        repo: DataRepository,
        race_predictor: RacePredictor,
        models: TrainedModelsV5,
        output_dir: Path = Path("data/paper_trading"),
    ) -> None:
        self.repo = repo
        self.race_predictor = race_predictor
        self.models = models
        self.output_dir = output_dir

    def setup(
        self,
        target_date: date,
        everydb2: EveryDB2Queries,
    ) -> list[dict[str, Any]]:
        """当日の出走表を取得し、履歴特徴量を生成。

        Returns:
            レーススケジュール (race_id, venue, race_num, post_time, surface, distance)。
            事前計算済み特徴量を predictions/YYYYMMDD_pre.parquet に保存。

        注意: setup 段階では weight_absolute, オッズ関連列は NaN のまま。
              EV計算が不可能なため、ベット決定は行わない。
        """
        from features.feature_engine import FeatureEngine
        from features.horse_history_features import HorseHistoryFeatures
        from features.jockey_context_features import JockeyContextFeatures
        from features.trainer_context_features import TrainerContextFeatures
        from models.submodel_manager import SubModelManager

        # 1. スケジュール取得
        schedule = everydb2.get_race_schedule(target_date)
        if not schedule:
            logger.info("No races on %s", target_date)
            return []

        # 2. Parquet から特徴量を生成
        ymd = target_date.strftime("%Y%m%d")
        dash_date = target_date.isoformat()

        race_df = self.repo.load_races(ymd, ymd)
        entry_df = self.repo.load_entries(ymd, ymd)
        odds_df = self.repo.load_odds_snapshots(ymd, ymd)

        if race_df.empty:
            logger.warning("No race data in Parquet for %s", target_date)
            return []

        feat_engine = FeatureEngine()
        submodel_mgr = SubModelManager()
        feat_df = feat_engine.build_all(
            race_df, entry_df, odds_df, odds_ts_df=None, repo=self.repo
        )
        feat_df = submodel_mgr.add_distance_band_features(feat_df)

        # 3. 事前特徴量の計算
        race_ids = feat_df["race_id"].unique()
        hist_all = HorseHistoryFeatures(repo=self.repo)
        hist_df = hist_all.compute(race_df, entry_df, race_ids)

        jockey_ctx = JockeyContextFeatures(self.repo)
        jockey_df = jockey_ctx.compute(entry_df)

        trainer_ctx = TrainerContextFeatures(self.repo)
        trainer_df = trainer_ctx.compute(entry_df)

        # マージして保存
        for col_df in [hist_df, jockey_df, trainer_df]:
            if not col_df.empty:
                common_cols = [c for c in col_df.columns if c in ["race_id", "umaban"]]
                merge_cols = [c for c in col_df.columns if c not in feat_df.columns or c in common_cols]
                feat_df = feat_df.merge(
                    col_df[merge_cols], on=["race_id", "umaban"], how="left"
                )

        # 事前計算済み特徴量を保存
        pred_dir = self.output_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pre_path = pred_dir / f"{ymd}_pre.parquet"
        feat_df.to_parquet(pre_path, index=False)
        logger.info("Pre-computed features saved: %s (%d races)", pre_path, len(race_ids))

        return schedule

    def predict_race(
        self,
        race_id: str,
        pre_computed_features: pd.DataFrame,
        horse_weights: pd.DataFrame,
        odds: pd.DataFrame,
        bankroll: float,
    ) -> list[dict[str, Any]]:
        """1レース分の予測。馬体重+オッズ特徴量をマージして推論。

        Returns:
            bet_history と同じスキーマの dict リスト。
        """
        race_df = pre_computed_features[
            pre_computed_features["race_id"] == race_id
        ].copy()

        if race_df.empty:
            logger.warning("No pre-computed features for %s", race_id)
            return []

        # 馬体重マージ
        if horse_weights is not None and not horse_weights.empty:
            weight_map = dict(zip(horse_weights["umaban"], horse_weights["weight"]))
            race_df["ba_taijyu"] = race_df["umaban"].map(weight_map)
            if "weight_absolute" in race_df.columns:
                race_df["weight_absolute"] = race_df["umaban"].map(weight_map)

        # オッズマージ
        if odds is not None and not odds.empty:
            odds_map = dict(zip(odds["umaban"], odds["fuku_odds"]))
            race_df["place_odds_actual"] = race_df["umaban"].map(odds_map)
            tan_map = dict(zip(odds["umaban"], odds["tan_odds"]))
            race_df["win_odds"] = race_df["umaban"].map(tan_map)

        # 推論
        result_df = self.race_predictor.predict(race_df)
        if result_df.empty:
            return []

        # Quality screening
        if not self.race_predictor.should_bet(result_df):
            logger.info("Race %s skipped by quality screener", race_id)
            return []

        # ベット選定
        bets = self.race_predictor.select_bets(result_df, bankroll)

        # dict リストに変換 (bet_history スキーマ)
        bet_dicts = []
        surface_key = result_df["surface_key"].iloc[0]
        race_date = pd.Timestamp(f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}")
        for bet in bets:
            bet_dicts.append({
                "race_id": race_id,
                "bet_type": bet.bet_type.value,
                "umaban": bet.umaban,
                "stake": bet.stake,
                "odds": bet.odds,
                "result": 0.0,  # 未確定
                "surface": surface_key,
                "distance": int(result_df["distance"].iloc[0])
                if "distance" in result_df.columns else 0,
                "ev": float(bet.ev_lower_corrected),
                "popularity": 0,
                "bankroll_after": bankroll - bet.stake,
                "race_date": race_date,  # datetime64 (spec Section 5.1)
                "horse_name": "",
                "is_paper": True,
            })

        return bet_dicts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_paper_predictor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/paper_trading/predictor.py tests/test_paper_predictor.py
git commit -m "feat: PaperPredictor — setup/predict_race の実装"
```

---

### Task 9: PaperReconciler

予測と結果を照合し、ROI を追跡する。

**Files:**
- Create: `src/paper_trading/reconciler.py`
- Test: `tests/test_paper_reconciler.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paper_reconciler.py
"""PaperReconciler のテスト"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestPaperReconciler:
    def test_reconcile_settles_winning_bets(self, tmp_path: Path) -> None:
        from paper_trading.reconciler import PaperReconciler

        mock_repo = MagicMock()
        mock_everydb2 = MagicMock()
        mock_everydb2.get_race_results.return_value = pd.DataFrame({
            "race_id": ["2026040510010101"],
            "umaban": [3],
            "finish_pos": [2],
            "place_pay": [240.0],
            "place_odds": [2.4],
            "horse_name": ["テスト馬"],
        })

        reconciler = PaperReconciler(
            repo=mock_repo,
            bets_path=tmp_path / "bets.parquet",
            everydb2=mock_everydb2,
        )

        # 既存のベット履歴 (未確定)
        existing_bets = pd.DataFrame([{
            "race_id": "2026040510010101",
            "bet_type": "place",
            "umaban": 3,
            "stake": 100.0,
            "odds": 2.4,
            "result": 0.0,
            "surface": "turf",
            "distance": 1200,
            "ev": 1.5,
            "popularity": 3,
            "bankroll_after": 99900.0,
            "race_date": pd.Timestamp("2026-04-05"),
            "horse_name": "テスト馬",
            "is_paper": True,
        }])
        existing_bets.to_parquet(tmp_path / "bets.parquet", index=False)

        result = reconciler.reconcile(date(2026, 4, 5))

        assert result["n_settled"] == 1
        assert result["n_wins"] == 1

    def test_reconcile_idempotent(self, tmp_path: Path) -> None:
        """重複実行時は既存レコードをスキップ (race_id + umaban で判定)"""
        from paper_trading.reconciler import PaperReconciler

        mock_everydb2 = MagicMock()
        mock_everydb2.get_race_results.return_value = pd.DataFrame()

        reconciler = PaperReconciler(
            repo=MagicMock(),
            bets_path=tmp_path / "bets.parquet",
            everydb2=mock_everydb2,
        )

        # 既に確定済みのベット (result > 0 の勝ちケース)
        existing_bets = pd.DataFrame([{
            "race_id": "2026040510010101",
            "bet_type": "place",
            "umaban": 3,
            "stake": 100.0,
            "odds": 2.4,
            "result": 240.0,  # 既に確定
            "surface": "turf",
            "distance": 1200,
            "ev": 1.5,
            "popularity": 3,
            "bankroll_after": 100140.0,
            "race_date": pd.Timestamp("2026-04-05"),
            "horse_name": "テスト馬",
            "is_paper": True,
        }])
        existing_bets.to_parquet(tmp_path / "bets.parquet", index=False)

        result = reconciler.reconcile(date(2026, 4, 5))
        assert result["n_settled"] == 0  # スキップされる
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_paper_reconciler.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# src/paper_trading/reconciler.py
"""Paper Trading 結果照合・ROI計算"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from db.everydb2_queries import EveryDB2Queries
    from db.repository import DataRepository
    from monitoring.model_monitor import ModelMonitor

logger = logging.getLogger(__name__)


class PaperReconciler:
    """reconcile フェーズ: 予測と結果を照合して ROI を追跡。

    冪等性: 同一 race_id + umaban のレコードが既に存在する場合はスキップ。
    """

    def __init__(
        self,
        repo: DataRepository,
        bets_path: Path,
        everydb2: EveryDB2Queries,
        monitor: ModelMonitor | None = None,
    ) -> None:
        self.repo = repo
        self.bets_path = bets_path
        self.everydb2 = everydb2
        self.monitor = monitor

    def reconcile(self, target_date: date) -> dict[str, Any]:
        """当日のレース結果を取得し、予測と照合。

        Returns:
            日次結果サマリー (n_settled, n_wins, daily_roi, cumulative_roi, etc.)
        """
        # 1. 既存ベットを読み込み
        if self.bets_path.exists():
            bets_df = pd.read_parquet(self.bets_path)
        else:
            bets_df = pd.DataFrame()

        if bets_df.empty:
            logger.info("No bets to reconcile for %s", target_date)
            return self._empty_result(target_date)

        # 2. 当日の未確定ベットを抽出
        target_ts = pd.Timestamp(target_date)
        pending = bets_df[
            (bets_df["race_date"] == target_ts)
            & (bets_df["result"] == 0.0)
        ]

        if pending.empty:
            logger.info("No pending bets for %s", target_date)
            return self._compute_summary(bets_df, target_date)

        # 3. レース結果を取得
        results_df = self.everydb2.get_race_results(target_date)
        if results_df.empty:
            logger.warning("No race results available for %s", target_date)
            return self._compute_summary(bets_df, target_date)

        # 4. 照合: race_id + umaban でマージ
        n_settled = 0
        n_wins = 0

        for _, bet_row in pending.iterrows():
            race_id = bet_row["race_id"]
            umaban = bet_row["umaban"]

            # 既に処理済みかチェック (冪等性: race_id + umaban で重複排除)
            # result > 0 だけでなく result == 0 (不的中) もスキップ
            existing = bets_df[
                (bets_df["race_id"] == race_id)
                & (bets_df["umaban"] == umaban)
            ]
            if not existing.empty:
                continue

            # 結果検索
            result_row = results_df[
                (results_df["race_id"] == race_id)
                & (results_df["umaban"] == umaban)
            ]
            if result_row.empty:
                continue

            finish_pos = int(result_row.iloc[0]["finish_pos"])
            bet_type = bet_row["bet_type"]

            # 複勝的中判定
            payout = 0.0
            if bet_type == "place" and 1 <= finish_pos <= 3:
                payout = bet_row["stake"] * bet_row["odds"]
                n_wins += 1

            # 払戻を更新
            mask = (bets_df["race_id"] == race_id) & (bets_df["umaban"] == umaban)
            bets_df.loc[mask, "result"] = payout
            n_settled += 1

        # 5. 保存
        bets_df.to_parquet(self.bets_path, index=False)

        # 6. ModelMonitor でドリフトチェック (オプション)
        if self.monitor is not None and n_settled > 0:
            settled_bets = bets_df[
                (bets_df["race_date"] == target_ts)
                & (bets_df["result"] > 0)
            ]
            if not settled_bets.empty:
                try:
                    drift_report = self.monitor.detect_drift(
                        settled_bets, settled_bets
                    )
                    if drift_report.needs_retrain:
                        logger.warning(
                            "Model drift detected on %s: %s",
                            target_date, drift_report.drifted_features,
                        )
                except Exception:
                    logger.debug("Drift check failed (insufficient data)")

        return self._compute_summary(bets_df, target_date, n_settled, n_wins)

    def _compute_summary(
        self,
        bets_df: pd.DataFrame,
        target_date: date,
        n_settled: int = 0,
        n_wins: int = 0,
    ) -> dict[str, Any]:
        """累積統計を計算"""
        total_bets = len(bets_df)
        total_stake = bets_df["stake"].sum()
        total_return = bets_df[bets_df["result"] > 0]["result"].sum()
        total_wins = (bets_df["result"] > 0).sum()

        cumulative_roi = total_return / total_stake if total_stake > 0 else 0.0

        # Max drawdown
        bankroll_series = bets_df["bankroll_after"]
        peak = bankroll_series.cummax()
        dd = (peak - bankroll_series) / peak
        max_dd = dd.max() if not dd.empty else 0.0

        return {
            "date": target_date.isoformat(),
            "n_bets": total_bets,
            "n_wins": int(total_wins),
            "total_stake": float(total_stake),
            "total_return": float(total_return),
            "cumulative_roi": float(cumulative_roi),
            "max_dd": float(max_dd),
            "bankroll": float(bankroll_series.iloc[-1]) if not bankroll_series.empty else 100000.0,
            "n_settled": n_settled,
            "n_new_wins": n_wins,
        }

    def _empty_result(self, target_date: date) -> dict[str, Any]:
        return {
            "date": target_date.isoformat(),
            "n_bets": 0, "n_wins": 0,
            "total_stake": 0.0, "total_return": 0.0,
            "cumulative_roi": 0.0, "max_dd": 0.0,
            "bankroll": 100000.0,
            "n_settled": 0, "n_new_wins": 0,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_paper_reconciler.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/paper_trading/reconciler.py tests/test_paper_reconciler.py
git commit -m "feat: PaperReconciler — 結果照合・ROI追跡 (冪等性保証)"
```

---

## Phase 4: Orchestration & Reporting

### Task 10: RaceWatcher

watch フェーズ。各レースの発走時刻に合わせてベット通知を行う。

**Files:**
- Create: `src/paper_trading/watcher.py`
- Test: `tests/test_race_watcher.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_race_watcher.py
"""RaceWatcher のテスト"""

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestRaceWatcher:
    def test_watch_processes_scheduled_races(self, tmp_path: Path) -> None:
        from paper_trading.watcher import RaceWatcher

        mock_predictor = MagicMock()
        mock_everydb2 = MagicMock()
        mock_notifier = MagicMock()

        schedule = [
            {"race_id": "2026040510010101", "venue": "中山", "race_num": 1,
             "post_time": "10:05", "surface": "turf", "distance": 1200,
             "horses": ["馬1", "馬2"]},
        ]

        mock_predictor.predict_race.return_value = [
            {"race_id": "2026040510010101", "umaban": 1, "stake": 100.0,
             "odds": 2.4, "ev": 1.5},
        ]
        mock_everydb2.get_horse_weights.return_value = MagicMock()
        mock_everydb2.get_latest_odds.return_value = MagicMock()

        watcher = RaceWatcher(
            predictor=mock_predictor,
            everydb2=mock_everydb2,
            notifier=mock_notifier,
            predictions_dir=tmp_path / "predictions",
        )

        # wait_until をモック (待機しない)
        with patch("paper_trading.watcher.wait_until"):
            watcher.watch(date(2026, 4, 5), schedule, bankroll=100000.0)

        mock_predictor.predict_race.assert_called_once()
        mock_notifier.send_prediction.assert_called_once()

    def test_watch_skips_processed_races(self, tmp_path: Path) -> None:
        """既に処理済みのレースはスキップ (冪等性)"""
        from paper_trading.watcher import RaceWatcher

        mock_predictor = MagicMock()
        mock_everydb2 = MagicMock()
        mock_notifier = MagicMock()

        schedule = [
            {"race_id": "2026040510010101", "venue": "中山", "race_num": 1,
             "post_time": "10:05", "surface": "turf", "distance": 1200,
             "horses": ["馬1"]},
        ]

        watcher = RaceWatcher(
            predictor=mock_predictor,
            everydb2=mock_everydb2,
            notifier=mock_notifier,
            predictions_dir=tmp_path / "predictions",
        )

        # 既に最終予測ファイルが存在
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"race_id": ["2026040510010101"]}).to_parquet(
            pred_dir / "20260405.parquet", index=False
        )

        with patch("paper_trading.watcher.wait_until"):
            watcher.watch(date(2026, 4, 5), schedule, bankroll=100000.0)

        mock_predictor.predict_race.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_race_watcher.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# src/paper_trading/watcher.py
"""Paper Trading watch フェーズ — レース時刻監視・ベット通知"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from db.everydb2_queries import EveryDB2Queries
    from monitoring.notifier import NotifierProtocol
    from paper_trading.predictor import PaperPredictor

logger = logging.getLogger(__name__)


def wait_until(target_time: datetime) -> None:
    """target_time まで待機 (テスト用にモック可能)"""
    now = datetime.now()
    if target_time > now:
        import time
        time.sleep((target_time - now).total_seconds())


class RaceWatcher:
    """watch フェーズ: スケジュールに基づき、各レースの発走-5分にベット通知。

    耐障害性:
    - PostgreSQL接続断: 接続を再確立してリトライ
    - プロセスクラッシュ: 既通知済みレースは predictions/YYYYMMDD.parquet で判定
    - ハートビート: 5分おきにログ出力
    """

    def __init__(
        self,
        predictor: PaperPredictor,
        everydb2: EveryDB2Queries,
        notifier: NotifierProtocol,
        predictions_dir: Path,
        retry_count: int = 3,
        retry_interval_seconds: int = 60,
        watch_lead_minutes: int = 5,
    ) -> None:
        self.predictor = predictor
        self.everydb2 = everydb2
        self.notifier = notifier
        self.predictions_dir = predictions_dir
        self.retry_count = retry_count
        self.retry_interval_seconds = retry_interval_seconds
        self.watch_lead_minutes = watch_lead_minutes

    def watch(
        self,
        target_date: date,
        schedule: list[dict[str, Any]],
        bankroll: float,
    ) -> list[dict[str, Any]]:
        """スケジュールに基づき、各レースの発走-5分にベット通知。

        Returns:
            当日の全ベット記録
        """
        import time

        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        ymd = target_date.strftime("%Y%m%d")
        final_pred_path = self.predictions_dir / f"{ymd}.parquet"
        pre_pred_path = self.predictions_dir / f"{ymd}_pre.parquet"

        all_bets: list[dict[str, Any]] = []

        if not pre_pred_path.exists():
            logger.error("Pre-computed features not found: %s", pre_pred_path)
            return all_bets

        pre_computed = pd.read_parquet(pre_pred_path)

        for race in schedule:
            race_id = race["race_id"]

            # 既に処理済みならスキップ
            if self._already_processed(race_id, final_pred_path):
                logger.info("Skipping already processed race: %s", race_id)
                continue

            # 発走-5分まで待機
            post_time = self._parse_post_time(target_date, race["post_time"])
            wait_until(post_time - timedelta(minutes=self.watch_lead_minutes))

            # PostgreSQLから当日データを取得
            horse_weights = None
            odds = None
            for attempt in range(self.retry_count):
                horse_weights = self.everydb2.get_horse_weights(race_id)
                odds = self.everydb2.get_latest_odds(race_id)
                if horse_weights is not None and odds is not None:
                    break
                logger.warning(
                    "Data fetch attempt %d/%d failed for %s",
                    attempt + 1, self.retry_count, race_id,
                )
                time.sleep(self.retry_interval_seconds)
            else:
                logger.warning("All data fetch attempts failed for %s, skipping", race_id)
                continue

            # 推論
            bets = self.predictor.predict_race(
                race_id, pre_computed, horse_weights, odds, bankroll
            )

            if bets:
                self.notifier.send_prediction(bets=bets, date=target_date.isoformat())
                all_bets.extend(bets)
                bankroll = bets[-1]["bankroll_after"]

        # 最終予測を保存
        if all_bets:
            pd.DataFrame(all_bets).to_parquet(final_pred_path, index=False)

        return all_bets

    def _already_processed(self, race_id: str, final_pred_path: Path) -> bool:
        """既に処理済みのレースかチェック"""
        if not final_pred_path.exists():
            return False
        try:
            df = pd.read_parquet(final_pred_path)
            return race_id in df["race_id"].values
        except Exception:
            return False

    @staticmethod
    def _parse_post_time(target_date: date, post_time_str: str) -> datetime:
        """'HH:MM' 形式の発走時刻を datetime に変換"""
        h, m = map(int, post_time_str.split(":"))
        return datetime.combine(target_date, __import__("datetime").time(h, m))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_race_watcher.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/paper_trading/watcher.py tests/test_race_watcher.py
git commit -m "feat: RaceWatcher — レース時刻監視・ベット通知"
```

---

### Task 11: PaperTradingReport

Paper Trading 用の HTML レポート生成。BacktestReportGenerator を拡張。

**Files:**
- Create: `src/paper_trading/report.py`
- Test: `tests/test_paper_trading_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paper_trading_report.py
"""PaperTradingReport のテスト"""

from pathlib import Path

import pytest


class TestPaperTradingReport:
    def test_generate_creates_html(self, tmp_path: Path) -> None:
        from paper_trading.report import PaperTradingReport

        report = PaperTradingReport(output_dir=tmp_path)
        bets = [
            {
                "race_id": "2026040510010101", "bet_type": "place", "umaban": 3,
                "stake": 100.0, "odds": 2.4, "result": 240.0,
                "surface": "turf", "distance": 1200, "ev": 1.5,
                "popularity": 3, "bankroll_after": 100140.0,
                "race_date": pd.Timestamp("2026-04-05"), "horse_name": "テスト馬", "is_paper": True,
            },
        ]
        summary = {
            "n_bets": 1, "n_wins": 1, "cumulative_roi": 1.40,
            "max_dd": 0.0, "bankroll": 100140.0,
        }

        report_path = report.generate(bets, summary)
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "Paper Trading" in content
        assert "テスト馬" in content

    def test_generate_with_empty_bets(self, tmp_path: Path) -> None:
        from paper_trading.report import PaperTradingReport

        report = PaperTradingReport(output_dir=tmp_path)
        summary = {
            "n_bets": 0, "n_wins": 0, "cumulative_roi": 0.0,
            "max_dd": 0.0, "bankroll": 100000.0,
        }

        report_path = report.generate([], summary)
        assert report_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_paper_trading_report.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# src/paper_trading/report.py
"""Paper Trading 用 HTML レポート生成器"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import BaseLoader, Environment


class PaperTradingReport:
    """Paper Trading 累積レポートを生成。

    BacktestReportGenerator と同じ構造だが、
    Paper Trading 固有の情報 (horse_name, daily breakdown) を追加。
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        bets: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> Path:
        """HTML レポートを生成"""
        enriched = self._derive_fields(bets)
        monthly = self._compute_monthly_stats(enriched)
        bankroll_series = self._compute_bankroll_series(enriched)

        try:
            commit_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit_hash = "unknown"

        html = self._render_html(enriched, monthly, bankroll_series, summary, commit_hash)

        outpath = self.output_dir / "report.html"
        outpath.write_text(html, encoding="utf-8")
        return outpath

    @staticmethod
    def _derive_fields(bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not bets:
            return []
        return [
            {
                **b,
                "profit": b["result"] - b["stake"],
                "is_win": b["result"] > 0,
            }
            for b in bets
        ]

    @staticmethod
    def _compute_monthly_stats(bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not bets:
            return []
        from collections import defaultdict

        monthly: dict[str, dict[str, float]] = defaultdict(
            lambda: {"bets": 0, "wins": 0, "stake": 0.0, "total_return": 0.0}
        )
        for b in bets:
            month = b["race_date"][:7]
            monthly[month]["bets"] += 1
            monthly[month]["stake"] += b["stake"]
            if b["result"] > 0:
                monthly[month]["wins"] += 1
                monthly[month]["total_return"] += b["result"]

        return [
            {
                "month": m,
                "bets": s["bets"],
                "wins": int(s["wins"]),
                "roi": s["total_return"] / s["stake"] if s["stake"] > 0 else 0.0,
            }
            for m, s in sorted(monthly.items())
        ]

    @staticmethod
    def _compute_bankroll_series(bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not bets:
            return []
        peak = 0.0
        series = []
        for b in bets:
            bal = b["bankroll_after"]
            peak = max(peak, bal)
            dd = (peak - bal) / peak if peak > 0 else 0.0
            series.append({"date": b["race_date"], "bankroll": bal, "drawdown": dd})
        return series

    @staticmethod
    def _render_html(
        bets: list[dict[str, Any]],
        monthly: list[dict[str, Any]],
        bankroll_series: list[dict[str, Any]],
        summary: dict[str, Any],
        commit_hash: str,
    ) -> str:
        """シンプルな HTML レポートを生成 (Jinja2 テンプレート)"""
        env = Environment(loader=BaseLoader(), autoescape=True)
        env.filters["pct"] = lambda x: f"{x:.1%}"
        env.filters["yen"] = lambda x: f"¥{x:,.0f}"

        template_str = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Paper Trading Report</title>
<style>
body{font-family:sans-serif;max-width:1200px;margin:0 auto;padding:20px}
h1{color:#333}.kpi{display:flex;gap:20px;margin:20px 0}
.kpi-card{background:#f5f5f5;padding:15px;border-radius:8px;flex:1;text-align:center}
.kpi-card .value{font-size:24px;font-weight:bold}
.kpi-card .label{color:#666;font-size:14px}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ddd;padding:8px;text-align:right}
th{background:#f0f0f0}
.win{color:green;font-weight:bold}.lose{color:red}
</style></head><body>
<h1>Paper Trading Report</h1>
<div class="kpi">
<div class="kpi-card"><div class="value">{{ summary.cumulative_roi|pct }}</div><div class="label">Cumulative ROI</div></div>
<div class="kpi-card"><div class="value">{{ summary.n_bets }}</div><div class="label">Total Bets</div></div>
<div class="kpi-card"><div class="value">{{ summary.max_dd|pct }}</div><div class="label">Max Drawdown</div></div>
<div class="kpi-card"><div class="value">{{ summary.bankroll|yen }}</div><div class="label">Bankroll</div></div>
</div>
<h2>Bet History</h2>
<table><tr><th>Date</th><th>Race</th><th>Horse</th><th>Uma</th><th>Odds</th><th>EV</th><th>Result</th><th>P/L</th></tr>
{% for b in bets %}
<tr><td>{{ b.race_date }}</td><td>{{ b.race_id }}</td><td>{{ b.horse_name }}</td>
<td>{{ b.umaban }}</td><td>{{ b.odds }}</td><td>{{ b.ev }}</td>
<td>{{ b.result }}</td><td class="{{ 'win' if b.is_win else 'lose' }}">{{ b.profit }}</td></tr>
{% endfor %}
</table>
<p style="color:#999;font-size:12px">commit: {{ commit_hash }} | generated: {{ generated_at }}</p>
</body></html>"""

        template = env.from_string(template_str)
        return template.render(
            bets=bets,
            monthly=monthly,
            bankroll_series=bankroll_series,
            summary=summary,
            commit_hash=commit_hash,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_paper_trading_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/paper_trading/report.py tests/test_paper_trading_report.py
git commit -m "feat: PaperTradingReport — HTMLレポート生成"
```

---

### Task 12: run_paper_trading.py (CLI Script)

全フェーズを統合するメインスクリプト。

**Files:**
- Create: `scripts/run_paper_trading.py`
- Test: `tests/test_run_paper_trading.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_paper_trading.py
"""run_paper_trading.py CLI のテスト"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestRunPaperTradingCLI:
    def test_parse_args_setup_mode(self) -> None:
        sys.argv = ["run_paper_trading.py", "--mode", "setup", "--date", "2026-04-05"]
        from scripts.run_paper_trading import parse_args
        args = parse_args()
        assert args.mode == "setup"
        assert args.date == "2026-04-05"

    def test_parse_args_watch_mode(self) -> None:
        sys.argv = ["run_paper_trading.py", "--mode", "watch", "--date", "2026-04-05"]
        from scripts.run_paper_trading import parse_args
        args = parse_args()
        assert args.mode == "watch"

    def test_parse_args_reconcile_mode(self) -> None:
        sys.argv = ["run_paper_trading.py", "--mode", "reconcile", "--date", "2026-04-05"]
        from scripts.run_paper_trading import parse_args
        args = parse_args()
        assert args.mode == "reconcile"

    def test_parse_args_dry_run_mode(self) -> None:
        sys.argv = ["run_paper_trading.py", "--mode", "dry-run", "--date", "2024-07-13"]
        from scripts.run_paper_trading import parse_args
        args = parse_args()
        assert args.mode == "dry-run"
        assert args.date == "2024-07-13"

    def test_parse_args_dry_run_range(self) -> None:
        sys.argv = [
            "run_paper_trading.py", "--mode", "dry-run",
            "--start", "2024-07-01", "--end", "2024-07-31",
        ]
        from scripts.run_paper_trading import parse_args
        args = parse_args()
        assert args.mode == "dry-run"
        assert args.start == "2024-07-01"
        assert args.end == "2024-07-31"

    @patch("scripts.run_paper_trading.PaperTradingConfig")
    def test_main_setup_mode(self, mock_config_cls: MagicMock) -> None:
        """setup モードが正しいコンポーネントを呼び出すことを確認"""
        from scripts.run_paper_trading import main

        mock_config = MagicMock()
        mock_config_cls.return_value = mock_config
        mock_config.ensure_dirs.return_value = {
            "predictions": Path("/tmp/pred"),
            "daily_summary": Path("/tmp/summary"),
            "dry_run": Path("/tmp/dry"),
            "model": Path("/tmp/model"),
            "bets": Path("/tmp/bets"),
        }

        with patch("scripts.run_paper_trading.DataRepository"), \
             patch("scripts.run_paper_trading.ModelLoader") as mock_loader_cls, \
             patch("scripts.run_paper_trading.RacePredictor"), \
             patch("scripts.run_paper_trading.PaperPredictor") as mock_pred_cls, \
             patch("scripts.run_paper_trading.EveryDB2Queries") as mock_edb_cls, \
             patch("scripts.run_paper_trading.SlackNotifier"), \
             patch("scripts.run_paper_trading.PaperReconciler"):

            mock_models = MagicMock()
            mock_info = MagicMock()
            mock_loader = MagicMock()
            mock_loader_cls.return_value = mock_loader
            mock_loader.load.return_value = (mock_models, mock_info)

            mock_pred = MagicMock()
            mock_pred_cls.return_value = mock_pred
            mock_pred.setup.return_value = []

            mock_edb = MagicMock()
            mock_edb_cls.return_value = mock_edb

            sys.argv = ["run_paper_trading.py", "--mode", "setup", "--date", "2026-04-05"]
            main()

            mock_pred.setup.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_paper_trading.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/run_paper_trading.py
"""Paper Trading メインスクリプト

使い方:
  python scripts/run_paper_trading.py --mode setup --date 2026-04-05
  python scripts/run_paper_trading.py --mode watch --date 2026-04-05
  python scripts/run_paper_trading.py --mode reconcile --date 2026-04-05
  python scripts/run_paper_trading.py --mode dry-run --date 2024-07-13
  python scripts/run_paper_trading.py --mode dry-run --start 2024-07-01 --end 2024-07-31
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

# プロジェクトルートをパスに追加
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper Trading")
    parser.add_argument("--mode", required=True,
                        choices=["setup", "watch", "reconcile", "dry-run"],
                        help="実行モード")
    parser.add_argument("--date", help="対象日 (YYYY-MM-DD)")
    parser.add_argument("--start", help="期間開始 (YYYYMMDD, dry-run用)")
    parser.add_argument("--end", help="期間終了 (YYYYMMDD, dry-run用)")
    parser.add_argument("--run-id", help="MLflow run ID (省略時は最新)")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> "PaperTradingConfig":
    from paper_trading.config import PaperTradingConfig

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set, notifications disabled")

    db_password = os.environ.get("PGPASSWORD", "")
    conn_str = f"postgresql://postgres:{db_password}@localhost:5432/everydb2"

    config = PaperTradingConfig(
        slack_webhook_url=webhook_url,
        everydb2_connection_string=conn_str,
        mlflow_run_id=args.run_id,
    )
    config.ensure_dirs()
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args)

    # --- モデルロード (全モードで ModelMonitor 用に必要) ---
    models = None
    if args.mode in ("setup", "watch", "reconcile", "dry-run"):
        from backtest.race_predictor import RacePredictor
        from db.model_loader import ModelLoader

        t0 = time.time()
        loader = ModelLoader(tracking_uri=config.mlflow_tracking_uri)
        models, model_info = loader.load(run_id=config.mlflow_run_id)
        logger.info("Model loaded: %s (train: %s ~ %s) in %.1fs",
                    model_info.mlflow_run_id, model_info.train_start,
                    model_info.train_end, time.time() - t0)

        # model_info.json を保存
        info_path = config.paper_trading_dir / "model" / "model_info.json"
        info_path.write_text(json.dumps({
            "mlflow_run_id": model_info.mlflow_run_id,
            "train_start": model_info.train_start,
            "train_end": model_info.train_end,
            "loaded_at": model_info.loaded_at,
        }, indent=2), encoding="utf-8")

    # --- リポジトリ ---
    from db.parquet_store import ParquetStore
    from db.repository import DataRepository

    repo = DataRepository(ParquetStore())

    if args.mode == "setup":
        _run_setup(args, config, models, repo)

    elif args.mode == "watch":
        _run_watch(args, config, models, repo)

    elif args.mode == "reconcile":
        _run_reconcile(args, config, repo, models)

    elif args.mode == "dry-run":
        _run_dry_run(args, config, models, repo)


def _run_setup(
    args: argparse.Namespace,
    config: "PaperTradingConfig",
    models: "TrainedModelsV5",
    repo: "DataRepository",
) -> None:
    from backtest.race_predictor import RacePredictor
    from db.everydb2_queries import EveryDB2Queries
    from paper_trading.predictor import PaperPredictor

    target_date = date.fromisoformat(args.date)
    race_predictor = RacePredictor(models)
    predictor = PaperPredictor(
        repo=repo,
        race_predictor=race_predictor,
        models=models,
        output_dir=config.paper_trading_dir,
    )
    everydb2 = EveryDB2Queries(connection_string=config.everydb2_connection_string)

    t0 = time.time()
    schedule = predictor.setup(target_date, everydb2)
    logger.info("Setup complete: %d races (%.1fs)", len(schedule), time.time() - t0)

    # schedule.json を保存
    schedule_path = config.paper_trading_dir / "schedule.json"
    schedule_path.write_text(
        json.dumps({"date": args.date, "races": schedule}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Schedule saved: %s", schedule_path)

    # Slack 通知
    if config.slack_webhook_url:
        from monitoring.notifier import SlackNotifier

        notifier = SlackNotifier(webhook_url=config.slack_webhook_url)
        notifier.send(f"Setup complete: {len(schedule)} races scheduled for {args.date}")


def _run_watch(
    args: argparse.Namespace,
    config: "PaperTradingConfig",
    models: "TrainedModelsV5",
    repo: "DataRepository",
) -> None:
    from backtest.race_predictor import RacePredictor
    from db.everydb2_queries import EveryDB2Queries
    from monitoring.notifier import LoggingNotifier, SlackNotifier
    from monitoring.notifier import CompositeNotifier
    from paper_trading.predictor import PaperPredictor
    from paper_trading.watcher import RaceWatcher

    target_date = date.fromisoformat(args.date)
    race_predictor = RacePredictor(models)
    predictor = PaperPredictor(
        repo=repo,
        race_predictor=race_predictor,
        models=models,
        output_dir=config.paper_trading_dir,
    )
    everydb2 = EveryDB2Queries(connection_string=config.everydb2_connection_string)

    # 通知設定
    notifiers = [LoggingNotifier()]
    if config.slack_webhook_url:
        notifiers.append(SlackNotifier(webhook_url=config.slack_webhook_url))
    notifier = CompositeNotifier(notifiers)

    # スケジュール読み込み
    schedule_path = config.paper_trading_dir / "schedule.json"
    if not schedule_path.exists():
        logger.error("schedule.json not found. Run --mode setup first.")
        sys.exit(1)
    schedule_data = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule = schedule_data["races"]

    watcher = RaceWatcher(
        predictor=predictor,
        everydb2=everydb2,
        notifier=notifier,
        predictions_dir=config.paper_trading_dir / "predictions",
        retry_count=config.retry_count,
        retry_interval_seconds=config.retry_interval_seconds,
        watch_lead_minutes=config.watch_lead_minutes,
    )

    logger.info("Watch mode started for %s (%d races)", args.date, len(schedule))
    bets = watcher.watch(target_date, schedule, bankroll=config.initial_bankroll)
    logger.info("Watch complete: %d bets placed", len(bets))


def _run_reconcile(
    args: argparse.Namespace,
    config: "PaperTradingConfig",
    repo: "DataRepository",
    models: "TrainedModelsV5 | None" = None,
) -> None:
    from db.everydb2_queries import EveryDB2Queries
    from monitoring.model_monitor import ModelMonitor
    from monitoring.notifier import LoggingNotifier, SlackNotifier
    from monitoring.notifier import CompositeNotifier
    from paper_trading.reconciler import PaperReconciler
    from paper_trading.report import PaperTradingReport

    target_date = date.fromisoformat(args.date)
    everydb2 = EveryDB2Queries(connection_string=config.everydb2_connection_string)

    # ModelMonitor (ドリフト検知用、オプション)
    monitor = None
    if models is not None and hasattr(models, "regime_detector"):
        try:
            monitor = ModelMonitor(
                regime_detector=models.regime_detector,
            )
        except Exception:
            logger.debug("ModelMonitor not available for reconcile")

    reconciler = PaperReconciler(
        repo=repo,
        bets_path=config.paper_trading_dir / "bets.parquet",
        everydb2=everydb2,
        monitor=monitor,
    )

    t0 = time.time()
    result = reconciler.reconcile(target_date)
    logger.info("Reconcile complete (%.1fs): %s", time.time() - t0, result)

    # 日次サマリー保存
    summary_dir = config.paper_trading_dir / "daily_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{target_date.strftime('%Y%m%d')}.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # HTML レポート更新
    if config.paper_trading_dir.joinpath("bets.parquet").exists():
        import pandas as pd
        bets_df = pd.read_parquet(config.paper_trading_dir / "bets.parquet")
        report = PaperTradingReport(output_dir=config.paper_trading_dir)
        report.generate(bets_df.to_dict("records"), result)
        logger.info("Report updated")

    # Slack 通知
    if config.slack_webhook_url:
        notifier = SlackNotifier(webhook_url=config.slack_webhook_url)
        notifier.send_daily_result(result)


def _run_dry_run(
    args: argparse.Namespace,
    config: "PaperTradingConfig",
    models: "TrainedModelsV5",
    repo: "DataRepository",
) -> None:
    """過去データで本番パイプラインの動作確認"""
    import pandas as pd
    from backtest.race_predictor import RacePredictor
    from features.feature_engine import FeatureEngine
    from features.horse_history_features import HorseHistoryFeatures
    from features.interaction_features import compute_interaction_features
    from features.jockey_context_features import JockeyContextFeatures
    from features.trainer_context_features import TrainerContextFeatures
    from models.submodel_manager import SubModelManager

    # 期間決定
    if args.date:
        dates = [date.fromisoformat(args.date)]
    elif args.start and args.end:
        start = date(int(args.start[:4]), int(args.start[4:6]), int(args.start[6:8]))
        end = date(int(args.end[:4]), int(args.end[4:6]), int(args.end[6:8]))
        dates = []
        d = start
        while d <= end:
            dates.append(d)
            d += __import__("datetime").timedelta(days=1)
    else:
        logger.error("--date or --start/--end required for dry-run")
        sys.exit(1)

    race_predictor = RacePredictor(models)

    # 特徴量を一括生成
    all_start = dates[0].strftime("%Y%m%d")
    all_end = dates[-1].strftime("%Y%m%d")

    logger.info("Loading data: %s ~ %s", all_start, all_end)
    race_df = repo.load_races(all_start, all_end)
    entry_df = repo.load_entries(all_start, all_end)
    odds_df = repo.load_odds_snapshots(all_start, all_end)

    if race_df.empty:
        logger.error("No race data found")
        sys.exit(1)

    feat_engine = FeatureEngine()
    submodel_mgr = SubModelManager()
    feat_df = feat_engine.build_all(race_df, entry_df, odds_df, repo=repo)
    feat_df = submodel_mgr.add_distance_band_features(feat_df)

    race_ids = feat_df["race_id"].unique()
    hist_all = HorseHistoryFeatures(repo=repo).compute(race_df, entry_df, race_ids)
    jockey_all = JockeyContextFeatures(repo).compute(entry_df)
    trainer_all = TrainerContextFeatures(repo).compute(entry_df)

    # 日次シミュレーション
    total_bets = 0
    total_stake = 0.0
    total_return = 0.0
    dry_run_dir = config.paper_trading_dir / "dry_run"
    dry_run_dir.mkdir(parents=True, exist_ok=True)

    bankroll = config.initial_bankroll

    for target_date in dates:
        ymd = target_date.strftime("%Y%m%d")
        day_races = [rid for rid in race_ids if rid[:8] == ymd]

        if not day_races:
            continue

        day_bets = []
        for race_id in day_races:
            race_df_single = feat_df[feat_df["race_id"] == race_id].copy()
            hist_race = hist_all[hist_all["race_id"] == race_id]
            jockey_race = jockey_all[jockey_all["race_id"] == race_id]
            trainer_race = trainer_all[trainer_all["race_id"] == race_id]

            result_df = race_predictor.predict(
                race_df_single, hist_race, jockey_race, trainer_race
            )
            if result_df.empty:
                continue

            if not race_predictor.should_bet(result_df):
                continue

            bets = race_predictor.select_bets(result_df, bankroll)
            for bet in bets:
                # 結果判定 (BacktestEngine と同じ)
                horse = result_df[result_df["umaban"] == bet.umaban]
                if not horse.empty:
                    finish_pos = int(horse.iloc[0]["finish_pos"])
                    payout = 0.0
                    if bet.bet_type.value == "place" and 1 <= finish_pos <= 3:
                        payout = bet.stake * bet.odds
                    bankroll -= bet.stake
                    if payout > 0:
                        bankroll += payout
                    total_stake += bet.stake
                    total_return += payout
                    total_bets += 1
                    day_bets.append({
                        "race_id": race_id, "umaban": bet.umaban,
                        "odds": bet.odds, "ev": bet.ev_lower_corrected,
                        "stake": bet.stake, "payout": payout,
                        "bankroll": bankroll,
                    })

        # 日次結果保存
        day_result = {
            "date": ymd,
            "n_bets": len(day_bets),
            "bankroll": bankroll,
        }
        (dry_run_dir / f"{ymd}.json").write_text(
            json.dumps(day_result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    roi = total_return / total_stake if total_stake > 0 else 0.0
    print(f"\nDry-run Results ({all_start} ~ {all_end}):")
    print(f"  Bets:    {total_bets}")
    print(f"  Stake:   ¥{total_stake:,.0f}")
    print(f"  Return:  ¥{total_return:,.0f}")
    print(f"  ROI:     {roi:.1%}")
    print(f"  Bankroll:¥{bankroll:,.0f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_paper_trading.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_paper_trading.py tests/test_run_paper_trading.py
git commit -m "feat: run_paper_trading.py — setup/watch/reconcile/dry-run CLI"
```

---

## Phase 5: Dry-run

### Task 13: Dry-run Integration Test

dry-run モードが実際の Parquet データで動作することを確認する統合テスト。

**Files:**
- Test: `tests/test_dry_run.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/test_dry_run.py
"""Dry-run 統合テスト (Parquet データ使用)"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestDryRunIntegration:
    """dry-run パイプラインの統合テスト。

    Parquet データが存在する環境でのみ実行。
    """

    @pytest.fixture
    def mock_env(self, tmp_path: Path) -> None:
        """最小限のモック環境を構築"""
        # 必要なディレクトリを作成
        (tmp_path / "data" / "paper_trading" / "dry_run").mkdir(parents=True)
        (tmp_path / "data" / "paper_trading" / "model").mkdir(parents=True)
        (tmp_path / "data" / "paper_trading" / "predictions").mkdir(parents=True)

    @patch("scripts.run_paper_trading.DataRepository")
    @patch("scripts.run_paper_trading.ModelLoader")
    def test_dry_run_single_day(
        self,
        mock_loader_cls: MagicMock,
        mock_repo_cls: MagicMock,
        tmp_path: Path,
        mock_env: None,
    ) -> None:
        """1日分の dry-run が正常終了することを確認"""
        from domain.models import SubmodelSet, TrainedModelsV5
        from domain.types import RegimeState

        # モックモデル
        models = MagicMock(spec=TrainedModelsV5)
        models.submodels = {"turf": MagicMock(spec=SubmodelSet)}
        models.quality_screener = MagicMock()
        models.quality_screener.should_bet.return_value = True
        models.regime_detector = MagicMock()
        models.regime_detector.current_regime = RegimeState.CONSERVATIVE
        models.regime_detector.get_strategy_params.return_value = {
            "ev_threshold": 1.20, "max_bets_per_race": 3,
        }

        mock_info = MagicMock()
        mock_info.mlflow_run_id = "test"
        mock_info.train_start = "2020-01-01"
        mock_info.train_end = "2023-12-31"
        mock_info.loaded_at = "now"

        mock_loader = MagicMock()
        mock_loader_cls.return_value = mock_loader
        mock_loader.load.return_value = (models, mock_info)

        # モックリポジトリ (空データ → 0 bets)
        mock_repo = MagicMock()
        mock_repo_cls.return_value = mock_repo
        mock_repo.load_races.return_value = pd.DataFrame()
        mock_repo.load_entries.return_value = pd.DataFrame()
        mock_repo.load_odds_snapshots.return_value = pd.DataFrame()

        import sys
        old_argv = sys.argv
        sys.argv = [
            "run_paper_trading.py", "--mode", "dry-run", "--date", "2024-07-13",
        ]

        try:
            from scripts.run_paper_trading import main
            main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

        # 空データなのでエラーで終了するはず (logger.error → sys.exit(1))
        mock_loader.load.assert_called_once()
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_dry_run.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_dry_run.py
git commit -m "test: dry-run 統合テストを追加"
```

---

## 実装後の確認事項

1. **EveryDB2 テーブル名確認**: `EveryDB2Queries` の SQL を実際の EveryDB2 インスタンスで確認・修正
2. **再学習**: 拡張した MLflow ロギングでモデルを再学習 (`run_train.py`)
3. **dry-run 動作確認**: 過去データで `--mode dry-run` が正常動作することを確認
4. **Windows Task Scheduler 登録**: setup/watch/reconcile の3タスクをスケジュール
5. **SLACK_WEBHOOK_URL**: 環境変数の設定を確認
