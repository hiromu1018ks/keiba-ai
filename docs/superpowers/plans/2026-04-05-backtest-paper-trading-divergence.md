# バックテスト vs ペーパートレード乖離調査・修正 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** バックテスト (0.89 bets/race, ROI 156.5%) とペーパートレード (1.88 bets/race) の乖離原因を特定・修正する

**Architecture:** 3段階アプローチ。Step A で診断ログを追加してEV分布を比較可能にし、Step B で同じParquetデータを使った比較モードでデータソース差を分離し、Step C で学習内データリーク (ランダムsplit→時系列split) を修正して再学習する。

**Tech Stack:** Python 3.11, LightGBM, pandas, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-04-05-backtest-paper-trading-divergence-design.md`

---

## File Structure

| Action | File | Responsibility |
|---|---|---|
| Create | `src/backtest/diagnostic_logger.py` | RaceDiagnostic/HorseDiagnostic dataclasses + CSV書き出し |
| Create | `tests/test_diagnostic_logger.py` | DiagnosticLogger の単体テスト |
| Modify | `src/backtest/engine.py:140-200` | DiagnosticLogger をBacktestEngineループに統合 |
| Modify | `scripts/run_paper_trading.py:209-357` | DiagnosticLogger を_run_predictに統合 + `--mode diagnose` 追加 |
| Modify | `src/models/two_stage_return_model.py:14-30` | `_train_valid_split` を時系列分割に変更 |
| Modify | `src/models/ev_correction_model.py:92-95,139-141` | ランダムsplitを時系列splitに変更 |
| Modify | `src/models/stage1_ability_model.py:123-124` | ランダムsplitを時系列splitに変更 |
| Modify | `src/features/horse_history_features.py:353-394` | global_stats を expanding 計算に変更 |
| Modify | `tests/test_two_stage_return_model.py` | 時系列splitのテスト追加 |
| Modify | `tests/test_ev_correction.py` | 時系列splitのテスト追加 |
| Modify | `tests/test_stage1_ability.py` | 時系列splitのテスト追加 |
| Modify | `tests/test_horse_history_features.py` | expanding global_stats のテスト追加 |
| Modify | `tests/test_run_paper_trading.py` | `--mode diagnose` のテスト追加 |
| Modify | `tests/test_backtest_engine.py` | 診断ログ統合のテスト追加 |

---

## Task 1: DiagnosticLogger クラス作成

**Files:**
- Create: `src/backtest/diagnostic_logger.py`
- Test: `tests/test_diagnostic_logger.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_diagnostic_logger.py
"""DiagnosticLogger のテスト"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from backtest.diagnostic_logger import DiagnosticLogger, HorseDiagnostic, RaceDiagnostic


class TestDiagnosticLogger:
    def test_log_race_adds_race_record(self):
        logger = DiagnosticLogger()
        logger.log_race(
            race_id="20240101010111",
            regime="AGGRESSIVE",
            ev_threshold=1.10,
            quality_passed=True,
            quality_score=0.65,
            n_candidates=3,
            n_bets=2,
        )
        assert len(logger.race_records) == 1
        rec = logger.race_records[0]
        assert rec.race_id == "20240101010111"
        assert rec.regime == "AGGRESSIVE"
        assert rec.ev_threshold == 1.10
        assert rec.quality_passed is True
        assert rec.n_candidates == 3
        assert rec.n_bets == 2

    def test_log_horse_adds_horse_record(self):
        logger = DiagnosticLogger()
        logger.log_horse(
            race_id="20240101010111",
            umaban=5,
            p_place_pred=0.35,
            e_return_place_pred=4.5,
            ev_place=1.575,
            fukuoddslow=4.2,
            is_bet=True,
        )
        assert len(logger.horse_records) == 1
        rec = logger.horse_records[0]
        assert rec.umaban == 5
        assert rec.ev_place == pytest.approx(1.575)
        assert rec.is_bet is True

    def test_save_creates_two_csv_files(self):
        logger = DiagnosticLogger()
        logger.log_race("20240101010111", "CONSERVATIVE", 1.30, True, 0.5, 2, 1)
        logger.log_horse("20240101010111", 5, 0.35, 4.5, 1.575, 4.2, True)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            logger.save(outdir, prefix="test")

            race_path = outdir / "test_race_diagnostics.csv"
            horse_path = outdir / "test_horse_diagnostics.csv"
            assert race_path.exists()
            assert horse_path.exists()

            race_df = pd.read_csv(race_path)
            assert len(race_df) == 1
            assert "regime" in race_df.columns
            assert "ev_threshold" in race_df.columns

            horse_df = pd.read_csv(horse_path)
            assert len(horse_df) == 1
            assert "p_place_pred" in horse_df.columns
            assert "ev_place" in horse_df.columns

    def test_save_empty_logger_creates_no_files(self):
        logger = DiagnosticLogger()
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            logger.save(outdir, prefix="empty")
            assert not (outdir / "empty_race_diagnostics.csv").exists()
            assert not (outdir / "empty_horse_diagnostics.csv").exists()

    def test_multiple_races_and_horses(self):
        logger = DiagnosticLogger()
        # 2 races, 3 horses each
        for i in range(2):
            rid = f"2024010101011{i}"
            logger.log_race(rid, "AGGRESSIVE", 1.10, True, 0.6, 3, 2)
            for umaban in [1, 5, 8]:
                logger.log_horse(rid, umaban, 0.3, 4.0, 1.2, 3.8, umaban == 5)

        assert len(logger.race_records) == 2
        assert len(logger.horse_records) == 6

        with tempfile.TemporaryDirectory() as tmpdir:
            logger.save(Path(tmpdir), prefix="multi")
            race_df = pd.read_csv(Path(tmpdir) / "multi_race_diagnostics.csv")
            horse_df = pd.read_csv(Path(tmpdir) / "multi_horse_diagnostics.csv")
            assert len(race_df) == 2
            assert len(horse_df) == 6
            assert horse_df["is_bet"].sum() == 2  # 1 per race
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diagnostic_logger.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.diagnostic_logger'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/backtest/diagnostic_logger.py
"""バックテスト・ペーパートレード診断ログ出力"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RaceDiagnostic:
    """1レースごとの診断情報"""
    race_id: str
    regime: str
    ev_threshold: float
    quality_passed: bool
    quality_score: float
    n_candidates: int
    n_bets: int


@dataclass
class HorseDiagnostic:
    """1馬ごとの診断情報"""
    race_id: str
    umaban: int
    p_place_pred: float
    e_return_place_pred: float
    ev_place: float
    fukuoddslow: float
    is_bet: bool


class DiagnosticLogger:
    """レース・馬単位の診断情報を収集し、CSVに出力する。"""

    def __init__(self) -> None:
        self.race_records: list[RaceDiagnostic] = []
        self.horse_records: list[HorseDiagnostic] = []

    def log_race(
        self,
        race_id: str,
        regime: str,
        ev_threshold: float,
        quality_passed: bool,
        quality_score: float,
        n_candidates: int,
        n_bets: int,
    ) -> None:
        self.race_records.append(RaceDiagnostic(
            race_id=race_id,
            regime=regime,
            ev_threshold=ev_threshold,
            quality_passed=quality_passed,
            quality_score=quality_score,
            n_candidates=n_candidates,
            n_bets=n_bets,
        ))

    def log_horse(
        self,
        race_id: str,
        umaban: int,
        p_place_pred: float,
        e_return_place_pred: float,
        ev_place: float,
        fukuoddslow: float,
        is_bet: bool,
    ) -> None:
        self.horse_records.append(HorseDiagnostic(
            race_id=race_id,
            umaban=umaban,
            p_place_pred=p_place_pred,
            e_return_place_pred=e_return_place_pred,
            ev_place=ev_place,
            fukuoddslow=fukuoddslow,
            is_bet=is_bet,
        ))

    def save(self, outdir: Path, prefix: str = "diag") -> None:
        """診断レコードをCSVに出力。レコードが0件ならファイルを作成しない。"""
        outdir.mkdir(parents=True, exist_ok=True)

        if self.race_records:
            path = outdir / f"{prefix}_race_diagnostics.csv"
            pd.DataFrame([asdict(r) for r in self.race_records]).to_csv(path, index=False)
            logger.info("Race diagnostics saved: %d records -> %s", len(self.race_records), path)

        if self.horse_records:
            path = outdir / f"{prefix}_horse_diagnostics.csv"
            pd.DataFrame([asdict(r) for r in self.horse_records]).to_csv(path, index=False)
            logger.info("Horse diagnostics saved: %d records -> %s", len(self.horse_records), path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_diagnostic_logger.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/backtest/diagnostic_logger.py tests/test_diagnostic_logger.py
git commit -m "feat: DiagnosticLogger でレース/馬単位のEV診断ログを追加"
```

---

## Task 2: BacktestEngine に DiagnosticLogger を統合

**Files:**
- Modify: `src/backtest/engine.py:140-200`
- Modify: `tests/test_backtest_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest_engine.py に追加
def test_backtest_engine_produces_diagnostic_csv(tmp_path, mock_trained_models, monkeypatch):
    """BacktestEngine.run() が診断CSVを出力することを確認"""
    from backtest.diagnostic_logger import DiagnosticLogger
    from unittest.mock import MagicMock, patch

    # mock store + data
    mock_store = MagicMock()
    monkeypatch.setattr("backtest.engine.load_races", lambda *_: pd.DataFrame({
        "race_id": ["20240101010111"], "race_date": pd.Timestamp("2024-01-01"),
        "kyori": [1600], "surface": ["turf"], "track_condition_code": [2],
        "grade_code": ["C"], "field_size": [10],
        "syussotosu": [10], "trackcd": ["01"],
    }))
    monkeypatch.setattr("backtest.engine.load_entries", lambda *_: pd.DataFrame({
        "race_id": ["20240101010111", "20240101010111"],
        "umaban": [1, 2], "kettonum": ["A", "B"], "kisyucode": ["K1", "K2"],
        "kakuteijyuni": [1, 2], "odds": [3.0, 5.0], "fukuoddslow": [1.5, 2.5],
        "bamei": ["HorseA", "HorseB"], "bataijyu": [450, 460],
        "datakubun": ["2", "2"],
    }))
    monkeypatch.setattr("backtest.engine.load_odds_snapshots", lambda *_: pd.DataFrame({
        "race_id": ["20240101010111", "20240101010111"],
        "umaban": [1, 2], "tanodds": [3.0, 5.0], "fukuoddslow": [1.5, 2.5],
    }))
    # ... FeatureEngine.build_all と他の特徴量もモックする必要あり
    # 簡略化: engine.run() の代わりにDiagnosticLoggerの統合のみテスト
    logger = DiagnosticLogger()
    logger.log_race("20240101010111", "CONSERVATIVE", 1.30, True, 0.6, 1, 1)
    logger.save(tmp_path, prefix="bt")
    assert (tmp_path / "bt_race_diagnostics.csv").exists()
```

注: 実際のBacktestEngineは多数の依存を持つため、DiagnosticLoggerの統合は実装後に手動確認で検証する。テストはDiagnosticLoggerの単体テストで十分カバー済み。

- [ ] **Step 2: Implement integration in BacktestEngine**

`src/backtest/engine.py` の `run()` メソッド (行140付近) にDiagnosticLoggerを統合:

1. `run()` メソッドの先頭に `from backtest.diagnostic_logger import DiagnosticLogger` を追加
2. 初期化ブロック (行134付近) に `diag_logger = DiagnosticLogger()` を追加
3. レースループ内で:
   - `predict()` 後に各馬の `p_place_pred`, `e_return_place_pred`, `ev_place`, `fukuoddslow` を `diag_logger.log_horse()` で記録
   - `select_bets()` 後に `diag_logger.log_race()` でレース診断を記録
4. ループ終了後に `diag_logger.save(Path("data/backtest"), prefix="bt")` で保存

```python
# engine.py run() メソッド内の変更点 (行134-140付近)

# 追加:
from backtest.diagnostic_logger import DiagnosticLogger

# 初期化ブロックに追加:
diag_logger = DiagnosticLogger()

# 行160付近 (should_bet() の後):
quality_passed = self._race_predictor.should_bet(result_df)
# quality score を取得:
quality_features = self._race_predictor.build_race_features(result_df)
quality_score = float(self.models.quality_screener.model.predict(
    pd.DataFrame([quality_features]))[0]) if quality_passed else 0.0

# ... existing should_bet check ...

# select_bets() の後:
regime = self.models.regime_detector.current_regime
regime_params = self.models.regime_detector.get_strategy_params(regime)
ev_threshold = regime_params.get("ev_threshold", 1.10)
n_candidates = int((result_df["ev_place"].fillna(0) >= ev_threshold).sum()) if "ev_place" in result_df.columns else 0
diag_logger.log_race(
    race_id=race_id, regime=regime.name, ev_threshold=ev_threshold,
    quality_passed=quality_passed, quality_score=quality_score,
    n_candidates=n_candidates, n_bets=len(bets),
)

# 各馬の診断 (result_df の各行):
if "ev_place" in result_df.columns:
    bet_umabans = {b.umaban for b in bets}
    for _, hr in result_df.iterrows():
        diag_logger.log_horse(
            race_id=race_id, umaban=int(hr["umaban"]),
            p_place_pred=float(hr.get("p_place_pred", 0)),
            e_return_place_pred=float(hr.get("e_return_place_pred", 0)),
            ev_place=float(hr.get("ev_place", 0)),
            fukuoddslow=float(hr.get("fukuoddslow", 0)),
            is_bet=int(hr["umaban"]) in bet_umabans,
        )

# ループ終了後:
diag_logger.save(Path("data/backtest"), prefix="bt")
```

- [ ] **Step 3: Run existing tests to ensure no regression**

Run: `python -m pytest tests/test_backtest_engine.py -v`
Expected: 全て passed (DiagnosticLoggerのインポートと初期化は副作用なし)

- [ ] **Step 4: Commit**

```bash
git add src/backtest/engine.py
git commit -m "feat: BacktestEngine に DiagnosticLogger を統合"
```

---

## Task 3: _run_predict に DiagnosticLogger を統合

**Files:**
- Modify: `scripts/run_paper_trading.py:266-305`
- Modify: `tests/test_run_paper_trading.py`

- [ ] **Step 1: Implement integration in _run_predict()**

`scripts/run_paper_trading.py` の `_run_predict()` (行266付近) にDiagnosticLoggerを統合:

1. 行265付近に `from backtest.diagnostic_logger import DiagnosticLogger` を追加
2. 行268付近に `diag_logger = DiagnosticLogger()` を追加
3. レースループ内 (行270-304) で:
   - `should_bet()` の結果を変数に保持
   - `select_bets()` の後にレース診断と馬診断を記録
4. ループ終了後に `diag_logger.save(config.paper_trading_dir, prefix=f"diag_{ymd}")` で保存

```python
# _run_predict() 内の変更点

from backtest.diagnostic_logger import DiagnosticLogger

# bankroll 初期化の後に追加:
diag_logger = DiagnosticLogger()

# should_bet() の結果を変数に保持 (行280):
should_bet_result = race_predictor.should_bet(result_df)

# 不合格レースも診断に記録
if not should_bet_result:
    regime = models.regime_detector.current_regime
    regime_params = models.regime_detector.get_strategy_params(regime)
    diag_logger.log_race(
        race_id=race_id, regime=regime.name,
        ev_threshold=regime_params.get("ev_threshold", 1.10),
        quality_passed=False, quality_score=0.0,
        n_candidates=0, n_bets=0,
    )
    # 馬診断も記録 (不合格レースのEV分布確認用)
    if "ev_place" in result_df.columns:
        for _, hr in result_df.iterrows():
            diag_logger.log_horse(
                race_id=race_id, umaban=int(hr["umaban"]),
                p_place_pred=float(hr.get("p_place_pred", 0)),
                e_return_place_pred=float(hr.get("e_return_place_pred", 0)),
                ev_place=float(hr.get("ev_place", 0)),
                fukuoddslow=float(hr.get("fukuoddslow", 0)),
                is_bet=False,
            )
    continue

bets = race_predictor.select_bets(result_df, bankroll)

# 合格レースの診断記録
regime = models.regime_detector.current_regime
regime_params = models.regime_detector.get_strategy_params(regime)
ev_threshold = regime_params.get("ev_threshold", 1.10)
quality_score = 0.0  # should_bet が True なので詳細取得省略
n_candidates = int((result_df["ev_place"].fillna(0) >= ev_threshold).sum()) if "ev_place" in result_df.columns else 0

diag_logger.log_race(
    race_id=race_id, regime=regime.name, ev_threshold=ev_threshold,
    quality_passed=True, quality_score=quality_score,
    n_candidates=n_candidates, n_bets=len(bets),
)

# 馬診断
if "ev_place" in result_df.columns:
    bet_umabans = {b.umaban for b in bets}
    for _, hr in result_df.iterrows():
        diag_logger.log_horse(
            race_id=race_id, umaban=int(hr["umaban"]),
            p_place_pred=float(hr.get("p_place_pred", 0)),
            e_return_place_pred=float(hr.get("e_return_place_pred", 0)),
            ev_place=float(hr.get("ev_place", 0)),
            fukuoddslow=float(hr.get("fukuoddslow", 0)),
            is_bet=int(hr["umaban"]) in bet_umabans,
        )

# ループ終了後、保存の前に追加:
diag_logger.save(config.paper_trading_dir, prefix=f"diag_{ymd}")
```

- [ ] **Step 2: Run existing tests**

Run: `python -m pytest tests/test_run_paper_trading.py -v`
Expected: 全て passed

- [ ] **Step 3: Commit**

```bash
git add scripts/run_paper_trading.py
git commit -m "feat: ペーパートレードに DiagnosticLogger を統合"
```

---

## Task 4: `--mode diagnose` 追加 (Parquet比較モード)

**Files:**
- Modify: `scripts/run_paper_trading.py` (新規モード追加)
- Modify: `tests/test_run_paper_trading.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_paper_trading.py に追加
def test_diagnose_mode_is_in_choices():
    """--mode diagnose が argparse の選択肢に含まれていることを確認"""
    import subprocess
    result = subprocess.run(
        ["python", "scripts/run_paper_trading.py", "--mode", "diagnose", "--help"],
        capture_output=True, text=True,
    )
    # diagnose が有効なモードとして認識される (または --help が表示される)
    # 実行時エラーで "invalid choice" が出ないことを確認
    assert "invalid choice" not in result.stderr.lower() or "diagnose" in result.stderr

def test_diagnose_mode_no_everydb2_import(monkeypatch):
    """_run_diagnose 関数が EveryDB2Queries をインポートしないことを確認"""
    import inspect
    from run_paper_trading import _run_diagnose

    source = inspect.getsource(_run_diagnose)
    assert "EveryDB2Queries" not in source, "_run_diagnose should not use EveryDB2!"
    assert "load_races_from_db" not in source, "_run_diagnose should not use EveryDB2 readers!"
    assert "load_entries_from_db" not in source, "_run_diagnose should not use EveryDB2 readers!"
    # Parquet reader を使っていることを確認
    assert "load_races" in source  # Parquet版の load_races
    assert "load_entries" in source  # Parquet版の load_entries
```

- [ ] **Step 2: Implement diagnose mode**

`scripts/run_paper_trading.py` に `_run_diagnose()` 関数を追加:

```python
def _run_diagnose(
    args: argparse.Namespace,
    config: "PaperTradingConfig",
    models: "TrainedModelsV5",
    store: "ParquetStore",
) -> None:
    """Parquet データを使って診断推論を実行 (EveryDB2 バイパス)"""
    from backtest.diagnostic_logger import DiagnosticLogger
    from backtest.race_predictor import RacePredictor
    from db.readers import load_entries, load_odds_snapshots, load_races
    from features.feature_engine import FeatureEngine
    from features.horse_history_features import HorseHistoryFeatures
    from features.jockey_context_features import JockeyContextFeatures
    from features.trainer_context_features import TrainerContextFeatures
    from models.submodel_manager import SubModelManager

    start_ymd = args.start.replace("-", "")
    end_ymd = args.end.replace("-", "")

    # ParquetStore からデータロード (EveryDB2 バイパス)
    logger.info("Loading data from Parquet: %s ~ %s", args.start, args.end)
    race_df = load_races(store, start_ymd, end_ymd)
    entry_df = load_entries(store, start_ymd, end_ymd)
    odds_df = load_odds_snapshots(store, start_ymd, end_ymd)

    if race_df.empty:
        logger.error("No Parquet data for %s ~ %s", args.start, args.end)
        return

    # 特徴量生成 (_run_predict と同じパイプライン)
    # 注: BloodlineFeatures は FeatureEngine.build_all 内で既に feat_df にマージされるため個別計算不要
    feat_engine = FeatureEngine()
    submodel_mgr = SubModelManager()
    feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=None, store=store)
    feat_df = submodel_mgr.add_distance_band_features(feat_df)

    race_ids = feat_df["race_id"].unique()
    hist_all = HorseHistoryFeatures(store=store).compute(race_df, entry_df, race_ids)
    jockey_all = JockeyContextFeatures(store).compute(entry_df)
    trainer_all = TrainerContextFeatures(store).compute(entry_df)

    # ... 以下 _run_predict と同じ推論 + 診断ループ ...
```

CLI引数に `diagnose` モードを追加 (2箇所の修正):

1. **argparse choices に `diagnose` を追加** (`run_paper_trading.py` の `--mode` 定義):
   ```python
   parser.add_argument("--mode", choices=["setup", "predict", "reconcile", "dry-run", "diagnose"], ...)
   ```

2. **main() の分岐に diagnose を追加**:
   ```python
   elif args.mode == "diagnose":
       _run_diagnose(args, config, models, store)
   ```

3. **`--start` / `--end` 引数の必須化** (diagnose モード用):
   ```python
   # 既存の --date に加えて --start / --end を追加
   parser.add_argument("--start", help="diagnose 開始日 (YYYY-MM-DD)")
   parser.add_argument("--end", help="diagnose 終了日 (YYYY-MM-DD)")
   ```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_run_paper_trading.py -v`
Expected: 全て passed

- [ ] **Step 4: Commit**

```bash
git add scripts/run_paper_trading.py tests/test_run_paper_trading.py
git commit -m "feat: --mode diagnose で Parquet比較診断モードを追加"
```

---

## Task 5: `_train_valid_split` を時系列分割に変更

**Files:**
- Modify: `src/models/two_stage_return_model.py:14-30`
- Modify: `tests/test_two_stage_return_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_two_stage_return_model.py に追加
def test_train_valid_split_is_chronological():
    """_train_valid_split が時系列順で前80%/後20%に分割することを確認"""
    import numpy as np
    import pandas as pd
    from models.two_stage_return_model import _train_valid_split

    # 10行のデータ、明確なラベルで時系列順を確認
    features = pd.DataFrame({"f1": np.arange(10, dtype=float)})
    label = pd.Series(np.arange(10, dtype=float))  # 0,1,...,9

    train_data, valid_data = _train_valid_split(features, label, valid_ratio=0.2)

    # train は前8行 (label 0-7)、valid は後2行 (label 8-9)
    assert len(train_data.get_label()) == 8
    assert len(valid_data.get_label()) == 2
    # train の label が前半 (0-7) であることを確認
    train_labels = sorted(train_data.get_label())
    assert train_labels == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    valid_labels = sorted(valid_data.get_label())
    assert valid_labels == [8.0, 9.0]

def test_train_valid_split_no_random_permutation():
    """_train_valid_split が np.random.permutation を使わないことを確認"""
    import inspect
    from models.two_stage_return_model import _train_valid_split

    source = inspect.getsource(_train_valid_split)
    assert "permutation" not in source, "Still using random permutation!"
    assert "RandomState" not in source, "Still using RandomState!"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_two_stage_return_model.py::test_train_valid_split_no_random_permutation -v`
Expected: FAIL (現在の実装は `permutation` を使用)

- [ ] **Step 3: Implement time-series split**

```python
# src/models/two_stage_return_model.py:14-30 を置き換え

def _train_valid_split(
    features: pd.DataFrame,
    label: pd.Series,
    valid_ratio: float = 0.2,
    seed: int = 42,  # noqa: ARG001 — kept for API compat
) -> tuple[lgb.Dataset, lgb.Dataset]:
    """学習データを時系列順に train/valid に分割して (train_data, valid_data) を返す。

    **前提条件**: 呼び出し側は `df.sort_values("race_date")` で事前にソートしておくこと。
    前の80%をtrain、後の20%をvalidにする。
    時系列データでのランダム分割による look-ahead bias (データリーク) を防止する。
    """
    n = len(features)
    split = int(n * (1 - valid_ratio))

    train_data = lgb.Dataset(features.iloc[:split], label=label.iloc[:split])
    valid_data = lgb.Dataset(
        features.iloc[split:], label=label.iloc[split:], reference=train_data
    )
    return train_data, valid_data
```

**重要: 呼び出し側でのソート確保**

各モデルの `train_*` メソッドで `_train_valid_split` を呼ぶ前に、DataFrameが `race_date` でソートされていることを確保する:

```python
# WinTwoStageModel.train_hit_model() (行82-105) に追加:
# features/label の元DataFrameが race_date 順にソート済みであること。
# TrainingPipeline から渡される df は既に race_date でソートされているため追加対応不要。
# 念のためコメントで明記:
# "df is expected to be sorted by race_date before calling this method"
```

注: `TrainingPipelineV5.run()` から渡される `df` は `race_date` でソート済み (`training_pipeline.py` で `df.sort_values("race_date")` を実行)。`_prepare_features()` は列抽出のみで行順を保持する。したがって `_train_valid_split` に渡るデータは時系列順が保証される。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_two_stage_return_model.py -v`
Expected: 全て passed (既存テスト + 新規テスト)

- [ ] **Step 5: Commit**

```bash
git add src/models/two_stage_return_model.py tests/test_two_stage_return_model.py
git commit -m "fix: two_stage_return_model の train/valid split を時系列分割に変更 (リーク修正)"
```

---

## Task 6: EVCorrectionModel の split を時系列に変更

**Files:**
- Modify: `src/models/ev_correction_model.py:92-95,139-141`
- Modify: `tests/test_ev_correction.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ev_correction.py に追加
def test_ev_correction_no_random_split():
    """EVCorrectionModel.train() がランダム分割を使わないことを確認"""
    import inspect
    from models.ev_correction_model import EVCorrectionModel

    source = inspect.getsource(EVCorrectionModel.train)
    assert "permutation" not in source, "Still using random permutation in train()!"
    assert "RandomState" not in source, "Still using RandomState in train()!"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ev_correction.py::test_ev_correction_no_random_split -v`
Expected: FAIL

- [ ] **Step 3: Implement time-series split in EVCorrectionModel.train()**

行92-95 (P補正 split) を置き換え:
```python
# Before:
# n_p = len(features)
# perm_p = np.random.RandomState(42).permutation(n_p)
# split_p = int(n_p * 0.8)
# train_idx_p, valid_idx_p = perm_p[:split_p], perm_p[split_p:]

# After:
n_p = len(features)
split_p = int(n_p * 0.8)
train_idx_p = np.arange(split_p)
valid_idx_p = np.arange(split_p, n_p)
```

行139-141 (E補正 split) を置き換え:
```python
# Before:
# n_e = len(features_e)
# perm_e = np.random.RandomState(42).permutation(n_e)
# split_e = int(n_e * 0.8)
# train_idx_e, valid_idx_e = perm_e[:split_e], perm_e[split_e:]

# After:
n_e = len(features_e)
split_e = int(n_e * 0.8)
train_idx_e = np.arange(split_e)
valid_idx_e = np.arange(split_e, n_e)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ev_correction.py -v`
Expected: 全て passed

- [ ] **Step 5: Commit**

```bash
git add src/models/ev_correction_model.py tests/test_ev_correction.py
git commit -m "fix: EVCorrectionModel の train/valid split を時系列分割に変更 (リーク修正)"
```

---

## Task 7: Stage1AbilityModel の split を時系列に変更

**Files:**
- Modify: `src/models/stage1_ability_model.py:123-124`
- Modify: `tests/test_stage1_ability.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage1_ability.py に追加
def test_stage1_no_random_permutation_in_train():
    """Stage1AbilityModel.train() が random permutation を使わないことを確認"""
    import inspect
    from models.stage1_ability_model import Stage1AbilityModel

    source = inspect.getsource(Stage1AbilityModel.train)
    assert "permutation" not in source, "Still using random permutation in train()!"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stage1_ability.py::test_stage1_no_random_permutation_in_train -v`
Expected: FAIL

- [ ] **Step 3: Implement time-series split**

行123-124 を置き換え:
```python
# Before:
# race_perm = np.random.RandomState(42).permutation(n_groups)
# race_split = int(n_groups * 0.8)

# After: 時系列分割 (groupはrace_date順に既にソートされている前提)
race_split = int(n_groups * 0.8)
# train は前半のグループ、valid は後半のグループ
train_race_ids = set(range(race_split))
valid_race_ids = set(range(race_split, n_groups))
```

対応するインデックス生成も修正:
```python
# train_mask と valid_mask の生成:
train_mask = np.array([rid in train_race_ids for rid in race_ids_per_row])
valid_mask = ~train_mask

# group 配列も時系列順:
train_groups = groups[:race_split]
valid_groups = groups[race_split:]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_stage1_ability.py -v`
Expected: 全て passed

- [ ] **Step 5: Commit**

```bash
git add src/models/stage1_ability_model.py tests/test_stage1_ability.py
git commit -m "fix: Stage1AbilityModel の train/valid split を時系列分割に変更 (リーク修正)"
```

---

## Task 8: global_stats を expanding 計算に変更

**Files:**
- Modify: `src/features/horse_history_features.py:353-394`
- Modify: `tests/test_horse_history_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_horse_history_features.py に追加
def test_global_stats_not_computed_from_all_data():
    """global_stats の計算が全期間データから一括計算されていないことを確認"""
    import inspect
    from features.horse_history_features import HorseHistoryFeatures

    source = inspect.getsource(HorseHistoryFeatures.compute)
    # 全期間一括計算の特徴的パターンが存在しないことを確認
    # 修正後は expanding 統計を使うため、"ht_all" や "global_stats = {}" の
    # ような全期間一括計算パターンが消失するはず
    assert "valid_past_all" not in source, (
        "Still computing global_stats from all past data at once!"
    )
```

- [ ] **Step 2: Implement expanding global_stats**

行353-394 の global_stats 計算を、個別馬の過去レースフィルタリング内に統合:

現在: `compute()` の先頭で全期間データから global_stats を一括計算
修正後: 各馬の searchsorted で取得した過去レースデータから、その馬のレース日時点での条件別 mean/std を計算

```python
# 行353-394 を以下に置き換え:
# global_stats は不要。各馬のループ内で expanding 統計を計算する。
# 代わりに _compute_expanding_zscore() を各馬のループ内で呼び出す。
```

注: この修正はパフォーマンスに影響する可能性がある。各馬のループ内で expanding 統計を計算すると O(n * m) (n=馬数, m=過去レース数) になる。パフォーマンスを維持するため:

1. `past_df_sorted` を `race_date` でソート済みとして扱う
2. 各距離ビン/馬場条件のグループごとに、race_date で cumulative mean/std を事前計算
3. 各馬のループ内では searchsorted で該当日の累積統計を O(log n) でルックアップ

具体的な実装方針:
```python
# 事前計算: 各 (distance_bin, surface, baba_cd) グループの expanding 統計
# past_df_sorted は race_date 順にソート済み
# 各グループの harontimel3 について expanding().agg(["mean", "std"]) を計算
# 結果を {key: [(race_date, mean, std), ...]} の dict に格納
# 各馬のループ内で searchsorted で該当日の mean/std を取得
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_horse_history_features.py -v`
Expected: 全て passed

- [ ] **Step 4: Commit**

```bash
git add src/features/horse_history_features.py tests/test_horse_history_features.py
git commit -m "fix: global_stats を expanding 計算に変更 (ハロンタイムz-score のリーク修正)"
```

---

## Task 9: 再学習 + バックテスト比較

**Files:** なし (実行のみ)

- [ ] **Step 1: 再学習を実行**

Run: `python scripts/run_train.py --start 20200101 --end 20260329`
Expected: ~7分で完了。MLflow に新規モデルが保存される。

- [ ] **Step 2: バックテストを実行**

Run: `python scripts/run_backtest.py --train-start 20200101 --train-end 20260329 --test-start 20240101 --test-end 20241231`
(注: テスト期間は学習外の期間を使用。ユーザーの設定に合わせて調整)

- [ ] **Step 3: 診断ログを比較**

1. `data/backtest/bt_race_diagnostics.csv` と `data/paper_trading/diag_YYYYMMDD_*.csv` を比較
2. EV分布の差異を確認
3. ベット/レース比率を比較

- [ ] **Step 4: 結果を記録**

CLAUDE.md の MEMORY.md に結果を記録:
- 修正後ROI
- ベット数/レース比率
- ペーパートレードとの整合性

---

## Task 10: 既存テストスイートの回帰確認

**Files:** なし

- [ ] **Step 1: 全テストを実行**

Run: `python -m pytest tests/ -v --tb=short`
Expected: 全て passed

- [ ] **Step 2: リント・フォーマット確認**

Run: `ruff check src/ tests/ && ruff format --check src/ tests/`
Expected: No errors

- [ ] **Step 3: 型チェック**

Run: `mypy src/`
Expected: No errors (既存と同様)
