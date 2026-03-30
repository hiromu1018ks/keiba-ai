# Backtest Report Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a self-contained HTML report from backtest results with KPI cards, bankroll chart, monthly dashboard, condition analysis, and filterable bet detail table.

**Architecture:** `BacktestReportGenerator` class enriches bet_history from `BacktestEngine`, computes derived statistics (monthly, condition, bankroll series), and renders a Jinja2 template with Chart.js charts and DataTables. The `--report` CLI flag triggers the full pipeline.

**Tech Stack:** Python 3.11, Jinja2, Chart.js (CDN), DataTables (CDN), pytest

---

## File Structure

Create:
- `src/backtest/report.py` — `BacktestReportGenerator` class (computation + HTML generation)
- `src/backtest/templates/report.html` — Jinja2 HTML template (5 sections + footer)
- `tests/test_backtest_report.py` — Unit tests for all report functionality

Modify:
- `pyproject.toml:6-19` — Add `jinja2>=3.1` to dependencies
- `src/backtest/engine.py:211-232` — Enrich `bet_history.append()` with 5 new fields, restructure loop for `bankroll_after`
- `scripts/run_backtest.py` — Add `--report` flag, save `bet_history.json`, move output to `data/backtest/`, call generator

---

## Task 1: Add jinja2 dependency

**Files:**
- Modify: `pyproject.toml:6-19`

- [ ] **Step 1: Add jinja2 to dependencies**

In `pyproject.toml`, add `jinja2>=3.1` to the `dependencies` list (after `tqdm>=4.66`):

```toml
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "lightgbm>=4.3",
    "psycopg2-binary>=2.9",
    "sqlalchemy>=2.0",
    "pyarrow>=14.0",
    "pyyaml>=6.0",
    "mlflow>=2.12",
    "tqdm>=4.66",
    "jinja2>=3.1",
    "pytest>=8.0",
    "pytest-cov>=5.0",
]
```

- [ ] **Step 2: Install dependency**

Run: `pip install -e ".[dev]"`
Expected: Successfully installed jinja2-3.x

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: jinja2をdependenciesに追加"
```

---

## Task 2: Enrich bet_history in engine.py

The engine loop at `src/backtest/engine.py:211-232` currently appends 6 fields to `bet_history`. We need to add 5 new fields: `surface`, `distance`, `ev`, `popularity`, `bankroll_after`.

**Key change:** The loop must be restructured — `bankroll -= bet.stake` and payout must happen BEFORE `bet_history.append()` so `bankroll_after` reflects the post-settlement balance.

**Files:**
- Modify: `src/backtest/engine.py:211-232`
- Modify: `tests/test_backtest_engine.py` (add enrichment test)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_backtest_engine.py` — a new test class after the existing `TestBacktestEngine`:

```python
class TestBetHistoryEnrichment:
    """bet_history への surface/distance/ev/popularity/bankroll_after 付与テスト"""

    @patch("features.trainer_context_features.TrainerContextFeatures")
    @patch("features.jockey_context_features.JockeyContextFeatures")
    @patch("features.interaction_features.compute_interaction_features")
    @patch("features.horse_history_features.HorseHistoryFeatures")
    @patch("models.submodel_manager.SubModelManager")
    @patch("features.feature_engine.FeatureEngine")
    @patch("backtest.engine.DataRepository")
    def test_engine_populates_enriched_fields(
        self,
        mock_repo_cls: MagicMock,
        mock_feat_engine_cls: MagicMock,
        mock_submodel_mgr_cls: MagicMock,
        mock_hist_cls: MagicMock,
        mock_interaction_fn: MagicMock,
        mock_jockey_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """エンジンループが bet_history に拡張フィールドを付与する"""
        # --- repo mock ---
        mock_repo = MagicMock(spec=DataRepository)
        mock_repo_cls.return_value = mock_repo
        mock_repo.load_races.return_value = pd.DataFrame({
            "race_id": ["20240101010101"],
            "race_date": pd.to_datetime("2024-01-01"),
        })
        mock_repo.load_entries.return_value = pd.DataFrame({
            "race_id": ["20240101010101"],
            "umaban": [1],
            "ketto_num": [1234],
            "finish_pos": [2],
            "win_odds": [5.0],
            "popularity_rank": [3],
            "ba_taijyu": [480],
            "zogen_fugo": [0],
            "zogen_sa": [0],
            "kisyu_code": [100],
            "chokyosi_code": [200],
        })
        mock_repo.load_odds_snapshots.return_value = pd.DataFrame()

        # --- feat_df (complete columns for pipeline) ---
        feat_df = pd.DataFrame({
            "race_id": ["20240101010101"],
            "umaban": [1],
            "surface_key": ["turf"],
            "surface": ["turf"],
            "distance": [1200],
            "distance_bin": ["sprint"],
            "popularity_rank": [3],
            "ninki": [3],
            "ev_place": [1.5],
            "place_odds_actual": [2.4],
            "finish_pos": [2],
            "ketto_num": [1234],
            "win_odds": [5.0],
            "ba_taijyu": [480],
        })

        # --- FeatureEngine mock ---
        mock_feat_engine = MagicMock()
        mock_feat_engine_cls.return_value = mock_feat_engine
        mock_feat_engine.build_all.return_value = feat_df

        # --- SubModelManager mock ---
        mock_submodel_mgr = MagicMock()
        mock_submodel_mgr_cls.return_value = mock_submodel_mgr
        mock_submodel_mgr.add_distance_band_features.return_value = feat_df

        # --- pre-computation mocks (return empty → merges are no-ops) ---
        mock_hist = MagicMock()
        mock_hist_cls.return_value = mock_hist
        mock_hist.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])
        mock_hist.add_race_transforms = staticmethod(lambda df: df)

        mock_interaction_fn.side_effect = lambda df: df

        mock_jockey = MagicMock()
        mock_jockey_cls.return_value = mock_jockey
        mock_jockey.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        # --- submodel mocks (all methods return feat_df) ---
        submodel = mock_models.submodels["turf"]
        submodel.market.predict_and_calc_error.return_value = feat_df
        submodel.stage1.add_ability_probs.return_value = feat_df
        submodel.place_ability.predict.return_value = feat_df
        submodel.win.predict_ev.return_value = feat_df
        submodel.ev_corrector.correct_ev.return_value = feat_df
        submodel.place.predict_ev.return_value = feat_df
        submodel.confidence.predict_lower_bound.return_value = (
            feat_df,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )

        # --- run engine ---
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models, repo=mock_repo)
        result = engine.run("2024-01-01", "2024-12-31")

        # --- assertions ---
        assert result.total_bets >= 1, "Should place at least 1 bet"
        bet = result.bet_history[0]
        assert "surface" in bet
        assert bet["surface"] == "turf"
        assert "distance" in bet
        assert bet["distance"] == 1200
        assert "ev" in bet
        assert bet["ev"] == 1.5
        assert "popularity" in bet
        assert bet["popularity"] == 3
        assert "bankroll_after" in bet
        assert isinstance(bet["bankroll_after"], float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backtest_engine.py::TestBetHistoryEnrichment -v`
Expected: FAIL — `AssertionError: 'surface' not in bet` (field not yet added)

- [ ] **Step 3: Modify engine.py — restructure the bet settlement loop**

Replace lines 211-232 in `src/backtest/engine.py` (the `for bet in bets:` block) with:

```python
            for bet in bets:
                bet_result = self._settle_bet(bet, race_df_single)

                bankroll -= bet.stake
                if bet_result > 0:
                    bankroll += bet_result

                # 拡張フィールド: popularity は馬ごと、それ以外はレースごと
                horse_rows = race_df_single[race_df_single["umaban"] == bet.umaban]
                pop_val = (
                    horse_rows["popularity_rank"].iloc[0]
                    if not horse_rows.empty
                    and "popularity_rank" in horse_rows.columns
                    else 0
                )

                bet_history.append(
                    {
                        "race_id": race_id,
                        "bet_type": bet.bet_type.value,
                        "umaban": bet.umaban,
                        "stake": bet.stake,
                        "odds": bet.odds,
                        "result": bet_result,
                        "surface": surface_key,
                        "distance": int(race_df_single["distance"].iloc[0]),
                        "ev": float(bet.ev_lower_corrected),
                        "popularity": int(pop_val),
                        "bankroll_after": round(bankroll, 2),
                    }
                )

                # DD 追跡
                peak_bankroll = max(peak_bankroll, bankroll)
                dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                max_dd = max(max_dd, dd)
```

`✶ Insight ─────────────────────────────────────`
**Loop restructure:** The original code appended to `bet_history` BEFORE updating `bankroll`. Since `bankroll_after` needs the post-settlement balance, we move `bankroll -= bet.stake` and the payout addition BEFORE the append. The DD tracking remains unchanged — it still uses the same post-settlement `bankroll`.

**popularity_rank safety:** `popularity_rank` may be absent if the entry data lacks `ninki`. The safe `.iloc[0]` with a column-existence guard prevents KeyError on missing data.
`─────────────────────────────────────────────────`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_backtest_engine.py::TestBetHistoryEnrichment -v`
Expected: PASS

- [ ] **Step 5: Run all engine tests to check no regression**

Run: `python -m pytest tests/test_backtest_engine.py -v`
Expected: All tests PASS (existing tests still work with empty DataFrames)

- [ ] **Step 6: Commit**

```bash
git add src/backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat: bet_historyにsurface/distance/ev/popularity/bankroll_afterを追加"
```

---

## Task 3: Create report.py with _derive_fields

**Files:**
- Create: `src/backtest/report.py`
- Create: `tests/test_backtest_report.py`

- [ ] **Step 1: Write the failing tests for _derive_fields**

Create `tests/test_backtest_report.py`:

```python
"""BacktestReportGenerator のテスト"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestDeriveFields:
    """_derive_fields のテスト"""

    def test_adds_race_date(self) -> None:
        """race_id の先頭8文字から race_date (YYYY-MM-DD) を抽出"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        bets = [
            {"race_id": "20240105101011", "stake": 100.0, "result": 240.0},
            {"race_id": "20241225123456", "stake": 100.0, "result": 0.0},
        ]
        result = gen._derive_fields(bets)
        assert result[0]["race_date"] == "2024-01-05"
        assert result[1]["race_date"] == "2024-12-25"

    def test_computes_profit(self) -> None:
        """profit = result - stake"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        bets = [
            {"race_id": "20240101010101", "stake": 100.0, "result": 240.0},
            {"race_id": "20240102010101", "stake": 100.0, "result": 0.0},
        ]
        result = gen._derive_fields(bets)
        assert result[0]["profit"] == 140.0
        assert result[1]["profit"] == -100.0

    def test_computes_is_win(self) -> None:
        """is_win = result > 0"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        bets = [
            {"race_id": "20240101010101", "stake": 100.0, "result": 240.0},
            {"race_id": "20240102010101", "stake": 100.0, "result": 0.0},
        ]
        result = gen._derive_fields(bets)
        assert result[0]["is_win"] is True
        assert result[1]["is_win"] is False

    def test_preserves_original_fields(self) -> None:
        """元のフィールドが保持される"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        bets = [
            {"race_id": "20240101010101", "stake": 100.0, "result": 240.0, "surface": "turf"},
        ]
        result = gen._derive_fields(bets)
        assert result[0]["surface"] == "turf"
        assert result[0]["race_id"] == "20240101010101"

    def test_empty_input(self) -> None:
        """空リストは空リストを返す"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        assert gen._derive_fields([]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest_report.py::TestDeriveFields -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest.report'`

- [ ] **Step 3: Create report.py with _derive_fields implementation**

Create `src/backtest/report.py`:

```python
"""バックテストHTMLレポート生成器"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from backtest.engine import BacktestResult


class BacktestReportGenerator:
    """バックテスト結果から自己完結型HTMLレポートを生成"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        result: BacktestResult,
        bet_history: list[dict[str, Any]],
        train_period: str = "",
        test_period: str = "",
    ) -> Path:
        """HTMLレポートを生成し、ファイルパスを返す"""
        bets = self._derive_fields(bet_history)
        monthly = self._compute_monthly_stats(bets)
        conditions = self._compute_condition_stats(bets)
        bankroll = self._compute_bankroll_series(bets)

        summary = {
            "roi": result.total_roi,
            "win_rate": result.winning_bets / result.total_bets if result.total_bets > 0 else 0.0,
            "profit": result.profit,
            "max_dd": result.max_drawdown,
            "final_bankroll": result.final_bankroll,
            "total_bets": result.total_bets,
            "total_stake": result.total_stake,
            "total_return": result.total_return,
            "test_period": test_period,
            "train_period": train_period,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

        template_dir = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
        template = env.get_template("report.html")

        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit_hash = "unknown"

        footer_info = f"commit: {commit_hash}"

        html = template.render(
            summary=summary,
            bankroll_series=bankroll,
            monthly_stats=monthly,
            condition_stats=conditions,
            bet_details=bets,
            footer_info=footer_info,
        )

        outpath = self.output_dir / "backtest_report.html"
        outpath.write_text(html, encoding="utf-8")
        return outpath

    def _derive_fields(self, bet_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """race_date, profit, is_win を派生フィールドとして追加"""
        if not bet_history:
            return []
        enriched = []
        for bet in bet_history:
            d = dict(bet)
            d["race_date"] = f"{bet['race_id'][:4]}-{bet['race_id'][4:6]}-{bet['race_id'][6:8]}"
            d["profit"] = bet["result"] - bet["stake"]
            d["is_win"] = bet["result"] > 0
            enriched.append(d)
        return enriched
```

`✶ Insight ─────────────────────────────────────`
**`_derive_fields` as pure function:** This method takes `bet_history` and returns a new list — no data access, no side effects. It only derives `race_date` from `race_id` (first 8 chars → `YYYY-MM-DD`), `profit` (result - stake), and `is_win` (result > 0). The original fields are preserved via `dict(bet)` copy.

**`FileSystemLoader` over `PackageLoader`:** Using `FileSystemLoader` with a path relative to `__file__` avoids needing `package_data` in `pyproject.toml`. Works in both development (`pip install -e .`) and regular installs.
`─────────────────────────────────────────────────`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest_report.py::TestDeriveFields -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/report.py tests/test_backtest_report.py
git commit -m "feat: BacktestReportGenerator._derive_fieldsを実装"
```

---

## Task 4: Add _compute_monthly_stats

**Files:**
- Modify: `src/backtest/report.py` (add method)
- Modify: `tests/test_backtest_report.py` (add test class)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_backtest_report.py`:

```python
class TestComputeMonthlyStats:
    """_compute_monthly_stats のテスト"""

    def _make_bets(self) -> list[dict[str, Any]]:
        return [
            {"race_date": "2024-01-05", "stake": 100.0, "result": 240.0, "is_win": True},
            {"race_date": "2024-01-15", "stake": 100.0, "result": 0.0, "is_win": False},
            {"race_date": "2024-01-20", "stake": 100.0, "result": 180.0, "is_win": True},
            {"race_date": "2024-02-10", "stake": 100.0, "result": 0.0, "is_win": False},
            {"race_date": "2024-02-20", "stake": 100.0, "result": 0.0, "is_win": False},
        ]

    def test_monthly_aggregation(self) -> None:
        """月次集計が正しい ROI, 的中率, ベット数を返す"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        result = gen._compute_monthly_stats(self._make_bets())

        assert len(result) == 2  # 2 months

        jan = [m for m in result if m["month"] == "2024-01"][0]
        assert jan["bets"] == 3
        assert jan["wins"] == 2
        assert jan["win_rate"] == pytest.approx(2 / 3)
        assert jan["stake"] == 300.0
        assert jan["total_return"] == 420.0
        assert jan["roi"] == pytest.approx(420.0 / 300.0)

        feb = [m for m in result if m["month"] == "2024-02"][0]
        assert feb["bets"] == 2
        assert feb["wins"] == 0
        assert feb["roi"] == pytest.approx(0.0)

    def test_empty_input(self) -> None:
        """空リストは空リストを返す"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        assert gen._compute_monthly_stats([]) == []

    def test_all_losses(self) -> None:
        """全額ロスの月の ROI が 0.0"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        bets = [
            {"race_date": "2024-03-01", "stake": 100.0, "result": 0.0, "is_win": False},
            {"race_date": "2024-03-15", "stake": 100.0, "result": 0.0, "is_win": False},
        ]
        result = gen._compute_monthly_stats(bets)
        assert result[0]["roi"] == 0.0
        assert result[0]["win_rate"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest_report.py::TestComputeMonthlyStats -v`
Expected: FAIL — `AttributeError: 'BacktestReportGenerator' object has no attribute '_compute_monthly_stats'`

- [ ] **Step 3: Implement _compute_monthly_stats**

Add to `src/backtest/report.py` (inside the class, after `_derive_fields`):

```python
    def _compute_monthly_stats(self, bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """月次集計: ROI, 的中率, ベット数, 投資額, 払戻額"""
        if not bets:
            return []
        from collections import defaultdict

        monthly: dict[str, dict[str, float]] = defaultdict(lambda: {"bets": 0, "wins": 0, "stake": 0.0, "total_return": 0.0})
        for b in bets:
            month = b["race_date"][:7]  # "YYYY-MM"
            monthly[month]["bets"] += 1
            monthly[month]["stake"] += b["stake"]
            if b["result"] > 0:
                monthly[month]["wins"] += 1
                monthly[month]["total_return"] += b["result"]

        result = []
        for month, stats in sorted(monthly.items()):
            bets_count = stats["bets"]
            result.append({
                "month": month,
                "bets": bets_count,
                "wins": int(stats["wins"]),
                "win_rate": stats["wins"] / bets_count if bets_count > 0 else 0.0,
                "stake": stats["stake"],
                "total_return": stats["total_return"],
                "roi": stats["total_return"] / stats["stake"] if stats["stake"] > 0 else 0.0,
            })
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest_report.py::TestComputeMonthlyStats -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/report.py tests/test_backtest_report.py
git commit -m "feat: 月次統計計算メソッドを実装"
```

---

## Task 5: Add _compute_condition_stats

**Files:**
- Modify: `src/backtest/report.py` (add method + helper)
- Modify: `tests/test_backtest_report.py` (add test class)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_backtest_report.py`:

```python
class TestComputeConditionStats:
    """_compute_condition_stats のテスト"""

    def _make_bets(self) -> list[dict[str, Any]]:
        """多様な条件を持つテストデータ"""
        return [
            # turf sprint, popular, high EV, win
            {"surface": "turf", "distance": 1200, "popularity": 1, "ev": 1.8,
             "stake": 100.0, "result": 250.0, "is_win": True},
            # turf sprint, popular, high EV, lose
            {"surface": "turf", "distance": 1200, "popularity": 2, "ev": 1.6,
             "stake": 100.0, "result": 0.0, "is_win": False},
            # turf mile, mid-pop, mid EV, win
            {"surface": "turf", "distance": 1600, "popularity": 5, "ev": 1.3,
             "stake": 100.0, "result": 300.0, "is_win": True},
            # dirt sprint, low-pop, low EV, lose
            {"surface": "dirt", "distance": 1200, "popularity": 8, "ev": 0.9,
             "stake": 100.0, "result": 0.0, "is_win": False},
            # dirt mile, low-pop, high EV, win
            {"surface": "dirt", "distance": 1600, "popularity": 7, "ev": 1.5,
             "stake": 100.0, "result": 400.0, "is_win": True},
        ]

    def test_surface_distance_analysis(self) -> None:
        """路面×距離帯の集計が正しい"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        result = gen._compute_condition_stats(self._make_bets())
        sd = result["surface_distance"]

        # turf/sprint: 2 bets, 1 win
        turf_sprint = [r for r in sd if r["surface"] == "turf" and r["distance_band"] == "sprint"][0]
        assert turf_sprint["bets"] == 2
        assert turf_sprint["wins"] == 1
        assert turf_sprint["win_rate"] == pytest.approx(0.5)

        # dirt/sprint: 1 bet, 0 wins
        dirt_sprint = [r for r in sd if r["surface"] == "dirt" and r["distance_band"] == "sprint"][0]
        assert dirt_sprint["bets"] == 1
        assert dirt_sprint["wins"] == 0

    def test_popularity_bands(self) -> None:
        """人気帯 (1-3, 4-6, 7+) の集計"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        result = gen._compute_condition_stats(self._make_bets())
        bands = result["popularity_bands"]

        band_1_3 = [b for b in bands if b["band"] == "1-3"][0]
        assert band_1_3["bets"] == 2  # popularity 1, 2
        assert band_1_3["wins"] == 1

        band_4_6 = [b for b in bands if b["band"] == "4-6"][0]
        assert band_4_6["bets"] == 1  # popularity 5

        band_7p = [b for b in bands if b["band"] == "7+"][0]
        assert band_7p["bets"] == 2  # popularity 7, 8

    def test_ev_bands(self) -> None:
        """EV帯 (<1.0, 1.0-1.2, 1.2-1.5, 1.5+) の集計"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        result = gen._compute_condition_stats(self._make_bets())
        bands = result["ev_bands"]

        band_low = [b for b in bands if b["band"] == "<1.0"][0]
        assert band_low["bets"] == 1  # ev 0.9

        band_high = [b for b in bands if b["band"] == "1.5+"][0]
        assert band_high["bets"] == 2  # ev 1.8, 1.6

    def test_empty_input(self) -> None:
        """空リストは空の統計を返す"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        result = gen._compute_condition_stats([])
        assert result["surface_distance"] == []
        assert result["popularity_bands"] == []
        assert result["ev_bands"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest_report.py::TestComputeConditionStats -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Implement _compute_condition_stats and _distance_band helper**

Add to `src/backtest/report.py` (inside the class, after `_compute_monthly_stats`):

```python
    @staticmethod
    def _distance_band(surface: str, distance: int) -> str:
        """surface + distance → distance_band (FeatureEngine と同じロジック)"""
        if surface == "turf":
            if distance <= 1400:
                return "sprint"
            if distance <= 1700:
                return "mile"
            if distance <= 2100:
                return "intermediate"
            return "long"
        else:
            if distance <= 1400:
                return "sprint"
            if distance <= 1700:
                return "mile"
            return "intermediate"

    def _compute_condition_stats(self, bets: list[dict[str, Any]]) -> dict[str, Any]:
        """路面×距離帯、人気帯、EV帯の集計"""
        if not bets:
            return {"surface_distance": [], "popularity_bands": [], "ev_bands": []}

        # --- Surface × Distance Band ---
        from collections import defaultdict

        sd_groups: dict[str, dict[str, float]] = defaultdict(
            lambda: {"bets": 0, "wins": 0, "stake": 0.0, "total_return": 0.0}
        )
        for b in bets:
            band = self._distance_band(b["surface"], b["distance"])
            key = f"{b['surface']}|{band}"
            sd_groups[key]["bets"] += 1
            sd_groups[key]["stake"] += b["stake"]
            if b["result"] > 0:
                sd_groups[key]["wins"] += 1
                sd_groups[key]["total_return"] += b["result"]

        surface_distance = []
        for key, s in sorted(sd_groups.items()):
            surface, band = key.split("|")
            n = s["bets"]
            surface_distance.append({
                "surface": surface,
                "distance_band": band,
                "bets": n,
                "wins": int(s["wins"]),
                "win_rate": s["wins"] / n if n > 0 else 0.0,
                "roi": s["total_return"] / s["stake"] if s["stake"] > 0 else 0.0,
            })

        # --- Helper for banded stats ---
        def _band_stats(
            bets_list: list[dict[str, Any]],
            key_fn,
            band_order: list[str],
        ) -> list[dict[str, Any]]:
            groups: dict[str, dict[str, float]] = defaultdict(
                lambda: {"bets": 0, "wins": 0, "total_payout": 0.0}
            )
            for b in bets_list:
                band = key_fn(b)
                groups[band]["bets"] += 1
                if b["result"] > 0:
                    groups[band]["wins"] += 1
                    groups[band]["total_payout"] += b["result"]

            result = []
            for band in band_order:
                if band not in groups:
                    continue
                g = groups[band]
                n = g["bets"]
                result.append({
                    "band": band,
                    "bets": n,
                    "wins": int(g["wins"]),
                    "win_rate": g["wins"] / n if n > 0 else 0.0,
                    "avg_payout": g["total_payout"] / g["wins"] if g["wins"] > 0 else 0.0,
                    "roi": g["total_payout"] / (n * 100.0) if n > 0 else 0.0,
                })
            return result

        popularity_bands = _band_stats(
            bets,
            lambda b: "1-3" if b["popularity"] <= 3 else "4-6" if b["popularity"] <= 6 else "7+",
            ["1-3", "4-6", "7+"],
        )
        ev_bands = _band_stats(
            bets,
            lambda b: (
                "<1.0" if b["ev"] < 1.0
                else "1.0-1.2" if b["ev"] < 1.2
                else "1.2-1.5" if b["ev"] < 1.5
                else "1.5+"
            ),
            ["<1.0", "1.0-1.2", "1.2-1.5", "1.5+"],
        )

        return {
            "surface_distance": surface_distance,
            "popularity_bands": popularity_bands,
            "ev_bands": ev_bands,
        }
```

`✶ Insight ─────────────────────────────────────`
**distance_band consistency:** `_distance_band` replicates the exact same logic as `FeatureEngine._map_basic_features()` (lines 195-207 of `feature_engine.py`). This ensures the report's condition analysis matches the model's feature boundaries.

**ROI definition for banded stats:** `roi = total_payout / (n * stake)` where stake is always 100 in backtests. This gives the actual return-on-investment ratio (1.0 = break-even), not the raw profit ratio.
`─────────────────────────────────────────────────`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest_report.py::TestComputeConditionStats -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/report.py tests/test_backtest_report.py
git commit -m "feat: 条件分析（路面×距離帯/人気帯/EV帯）計算メソッドを実装"
```

---

## Task 6: Add _compute_bankroll_series

**Files:**
- Modify: `src/backtest/report.py` (add method)
- Modify: `tests/test_backtest_report.py` (add test class)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_backtest_report.py`:

```python
class TestComputeBankrollSeries:
    """_compute_bankroll_series のテスト"""

    def test_bankroll_trajectory(self) -> None:
        """資金推移とドローダウンが正しく計算される"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        bets = [
            {"race_date": "2024-01-05", "bankroll_after": 100500.0},
            {"race_date": "2024-01-10", "bankroll_after": 100200.0},
            {"race_date": "2024-01-15", "bankroll_after": 100800.0},
            {"race_date": "2024-02-01", "bankroll_after": 99500.0},
        ]
        result = gen._compute_bankroll_series(bets)

        assert len(result) == 4
        assert result[0]["date"] == "2024-01-05"
        assert result[0]["bankroll"] == 100500.0
        assert result[0]["drawdown"] == 0.0  # peak = 100500, no DD

        # At 2024-01-10: bankroll=100200, peak=100500 → DD = (100500-100200)/100500
        assert result[1]["drawdown"] == pytest.approx(300.0 / 100500.0)

        # At 2024-01-15: bankroll=100800 > peak=100500 → new peak, DD=0
        assert result[2]["drawdown"] == 0.0

        # At 2024-02-01: bankroll=99500, peak=100800 → DD = (100800-99500)/100800
        assert result[3]["drawdown"] == pytest.approx(1300.0 / 100800.0)

    def test_single_bet(self) -> None:
        """ベット1件の場合 DD=0"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        bets = [{"race_date": "2024-01-01", "bankroll_after": 100000.0}]
        result = gen._compute_bankroll_series(bets)
        assert len(result) == 1
        assert result[0]["drawdown"] == 0.0

    def test_empty_input(self) -> None:
        """空リストは空リストを返す"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        assert gen._compute_bankroll_series([]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest_report.py::TestComputeBankrollSeries -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Implement _compute_bankroll_series**

Add to `src/backtest/report.py` (inside the class, after `_compute_condition_stats`):

```python
    def _compute_bankroll_series(self, bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """日付ごとの資金推移とドローダウンを抽出"""
        if not bets:
            return []
        series = []
        peak = 0.0
        for b in bets:
            bal = b["bankroll_after"]
            peak = max(peak, bal)
            dd = (peak - bal) / peak if peak > 0 else 0.0
            series.append({
                "date": b["race_date"],
                "bankroll": bal,
                "drawdown": dd,
            })
        return series
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest_report.py::TestComputeBankrollSeries -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/report.py tests/test_backtest_report.py
git commit -m "feat: 資金推移・ドローダウン計算メソッドを実装"
```

---

## Task 7: Create Jinja2 HTML template

**Files:**
- Create: `src/backtest/templates/report.html`

- [ ] **Step 1: Create the templates directory and HTML template**

Create directory: `src/backtest/templates/`

Create `src/backtest/templates/report.html`:

```html
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>バックテストレポート</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
    <style>
        :root {
            --primary: #2563eb;
            --success: #16a34a;
            --danger: #dc2626;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg); color: var(--text);
            margin: 0; padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 5px; font-size: 1.8em; }
        .subtitle { text-align: center; color: var(--text-muted); margin-bottom: 30px; }
        .meta {
            display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;
            margin-bottom: 20px; font-size: 13px; color: var(--text-muted);
        }
        .section {
            background: var(--card-bg); border-radius: 8px; padding: 20px;
            margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .section h2 {
            margin-top: 0; border-bottom: 2px solid var(--primary); padding-bottom: 10px;
            font-size: 1.3em;
        }
        .section h3 { font-size: 1.1em; margin-top: 20px; margin-bottom: 10px; }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
        }
        .kpi-card {
            background: var(--bg); border-radius: 8px; padding: 15px; text-align: center;
        }
        .kpi-label {
            font-size: 12px; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 1px;
        }
        .kpi-value { font-size: 28px; font-weight: 700; margin-top: 5px; }
        .kpi-value.positive { color: var(--success); }
        .kpi-value.negative { color: var(--danger); }
        .chart-container { position: relative; height: 300px; }
        .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 768px) { .charts-row { grid-template-columns: 1fr; } }
        .heatmap-positive { background-color: #dcfce7; }
        .heatmap-negative { background-color: #fee2e2; }
        .win-row { background-color: #f0fdf4 !important; }
        .footer {
            text-align: center; color: var(--text-muted);
            font-size: 12px; margin-top: 30px;
        }
        .no-data { color: var(--text-muted); font-style: italic; }
    </style>
</head>
<body>
<div class="container">
    <h1>バックテストレポート</h1>
    <p class="subtitle">{{ summary.test_period or "テスト期間未設定" }}</p>

    <div class="meta">
        <span>学習期間: {{ summary.train_period or "-" }}</span>
        <span>生成日時: {{ summary.generated_at }}</span>
    </div>

    <!-- Section 1: KPI Cards -->
    <div class="section">
        <h2>サマリー</h2>
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">ROI</div>
                <div class="kpi-value {{ 'positive' if summary.roi >= 1 else 'negative' }}">
                    {{ "%.1f%%"|format(summary.roi * 100) }}
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">的中率</div>
                <div class="kpi-value">{{ "%.1f%%"|format(summary.win_rate * 100) }}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">総利益</div>
                <div class="kpi-value {{ 'positive' if summary.profit >= 0 else 'negative' }}">
                    &yen;{{ "{:,.0f}"|format(summary.profit) }}
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">最大DD</div>
                <div class="kpi-value negative">{{ "%.1f%%"|format(summary.max_dd * 100) }}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">最終資金</div>
                <div class="kpi-value">&yen;{{ "{:,.0f}"|format(summary.final_bankroll) }}</div>
            </div>
        </div>
    </div>

    <!-- Section 2: Bankroll Chart -->
    <div class="section">
        <h2>資金推移</h2>
        {% if bankroll_series %}
        <div class="chart-container">
            <canvas id="bankroll-chart"></canvas>
        </div>
        {% else %}
        <p class="no-data">データなし</p>
        {% endif %}
    </div>

    <!-- Section 3: Monthly Dashboard -->
    <div class="section">
        <h2>月次ダッシュボード</h2>
        {% if monthly_stats %}
        <div class="charts-row">
            <div class="chart-container"><canvas id="monthly-roi-chart"></canvas></div>
            <div class="chart-container"><canvas id="monthly-bets-chart"></canvas></div>
        </div>
        <table class="display" style="width:100%; margin-top: 20px;">
            <thead>
                <tr>
                    <th>月</th><th>ベット数</th><th>的中</th><th>的中率</th>
                    <th>投資額</th><th>払戻額</th><th>ROI</th>
                </tr>
            </thead>
            <tbody>
                {% for m in monthly_stats %}
                <tr>
                    <td>{{ m.month }}</td>
                    <td>{{ m.bets }}</td>
                    <td>{{ m.wins }}</td>
                    <td>{{ "%.1f%%"|format(m.win_rate * 100) }}</td>
                    <td>&yen;{{ "{:,.0f}"|format(m.stake) }}</td>
                    <td>&yen;{{ "{:,.0f}"|format(m.total_return) }}</td>
                    <td style="color: {{ 'green' if m.roi >= 1 else 'red' }}">
                        {{ "%.1f%%"|format(m.roi * 100) }}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p class="no-data">データなし</p>
        {% endif %}
    </div>

    <!-- Section 4: Condition Analysis -->
    <div class="section">
        <h2>条件分析</h2>
        {% if condition_stats.surface_distance %}
        <h3>路面 x 距離帯</h3>
        <table class="display" style="width:100%; margin-bottom: 20px;">
            <thead>
                <tr><th>路面</th><th>距離帯</th><th>ベット数</th><th>的中率</th><th>ROI</th></tr>
            </thead>
            <tbody>
                {% for row in condition_stats.surface_distance %}
                <tr class="{{ 'heatmap-positive' if row.roi >= 1 else 'heatmap-negative' }}">
                    <td>{{ "芝" if row.surface == "turf" else "ダート" }}</td>
                    <td>{{ row.distance_band }}</td>
                    <td>{{ row.bets }}</td>
                    <td>{{ "%.1f%%"|format(row.win_rate * 100) }}</td>
                    <td>{{ "%.1f%%"|format(row.roi * 100) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}

        {% if condition_stats.popularity_bands or condition_stats.ev_bands %}
        <div class="charts-row">
            {% if condition_stats.popularity_bands %}
            <div>
                <h3>人気帯</h3>
                <table class="display" style="width:100%;">
                    <thead>
                        <tr><th>人気帯</th><th>ベット数</th><th>的中率</th><th>平均払戻</th><th>ROI</th></tr>
                    </thead>
                    <tbody>
                        {% for row in condition_stats.popularity_bands %}
                        <tr class="{{ 'heatmap-positive' if row.roi >= 1 else 'heatmap-negative' }}">
                            <td>{{ row.band }}</td><td>{{ row.bets }}</td>
                            <td>{{ "%.1f%%"|format(row.win_rate * 100) }}</td>
                            <td>&yen;{{ "{:,.0f}"|format(row.avg_payout) }}</td>
                            <td>{{ "%.1f%%"|format(row.roi * 100) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% endif %}
            {% if condition_stats.ev_bands %}
            <div>
                <h3>EV帯</h3>
                <table class="display" style="width:100%;">
                    <thead>
                        <tr><th>EV帯</th><th>ベット数</th><th>的中率</th><th>平均払戻</th><th>ROI</th></tr>
                    </thead>
                    <tbody>
                        {% for row in condition_stats.ev_bands %}
                        <tr class="{{ 'heatmap-positive' if row.roi >= 1 else 'heatmap-negative' }}">
                            <td>{{ row.band }}</td><td>{{ row.bets }}</td>
                            <td>{{ "%.1f%%"|format(row.win_rate * 100) }}</td>
                            <td>&yen;{{ "{:,.0f}"|format(row.avg_payout) }}</td>
                            <td>{{ "%.1f%%"|format(row.roi * 100) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% endif %}
        </div>
        {% else %}
        <p class="no-data">データなし</p>
        {% endif %}
    </div>

    <!-- Section 5: Bet Detail Table -->
    <div class="section">
        <h2>ベット明細</h2>
        {% if bet_details %}
        <table id="bet-table" class="display" style="width:100%;">
            <thead>
                <tr>
                    <th>日付</th><th>race_id</th><th>馬番</th><th>路面</th>
                    <th>距離</th><th>人気</th><th>EV</th><th>オッズ</th>
                    <th>ベット額</th><th>払戻</th><th>利益</th><th>結果</th>
                </tr>
            </thead>
            <tbody>
                {% for bet in bet_details %}
                <tr class="{{ 'win-row' if bet.is_win }}">
                    <td>{{ bet.race_date }}</td>
                    <td>{{ bet.race_id }}</td>
                    <td>{{ bet.umaban }}</td>
                    <td>{{ "芝" if bet.surface == "turf" else "ダート" }}</td>
                    <td>{{ bet.distance }}</td>
                    <td>{{ bet.popularity }}</td>
                    <td>{{ "%.2f"|format(bet.ev) }}</td>
                    <td>{{ "%.1f"|format(bet.odds) }}</td>
                    <td>&yen;{{ "{:,.0f}"|format(bet.stake) }}</td>
                    <td>&yen;{{ "{:,.0f}"|format(bet.result) }}</td>
                    <td style="color: {{ 'green' if bet.profit > 0 else 'red' }}">
                        &yen;{{ "{:,.0f}"|format(bet.profit) }}
                    </td>
                    <td>{{ "的中" if bet.is_win else "外れ" }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p class="no-data">データなし</p>
        {% endif %}
    </div>

    <div class="footer">
        <p>Generated by keiba-ai backtest report generator</p>
        <p>{{ footer_info }}</p>
    </div>
</div>

<script>
// --- Bankroll Chart ---
{% if bankroll_series %}
(function() {
    const data = {{ bankroll_series | tojson }};
    new Chart(document.getElementById('bankroll-chart').getContext('2d'), {
        type: 'line',
        data: {
            labels: data.map(d => d.date),
            datasets: [{
                label: '資金 (¥)',
                data: data.map(d => d.bankroll),
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                fill: true, tension: 0.1, pointRadius: 0,
            }, {
                label: 'DD (%)',
                data: data.map(d => -(d.drawdown * 100)),
                borderColor: '#dc2626',
                backgroundColor: 'rgba(220, 38, 38, 0.1)',
                fill: true, tension: 0.1, pointRadius: 0,
                yAxisID: 'y1',
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: { title: { display: true, text: '資金 (¥)' } },
                y1: { position: 'right', title: { display: true, text: 'DD (%)' },
                      grid: { drawOnChartArea: false } },
            },
        },
    });
})();
{% endif %}

// --- Monthly Charts ---
{% if monthly_stats %}
(function() {
    const data = {{ monthly_stats | tojson }};
    new Chart(document.getElementById('monthly-roi-chart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: data.map(d => d.month),
            datasets: [{
                label: 'ROI (%)',
                data: data.map(d => (d.roi - 1) * 100),
                backgroundColor: data.map(d => d.roi >= 1 ? '#16a34a' : '#dc2626'),
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { title: { display: true, text: 'ROI (%)' } } },
        },
    });
    new Chart(document.getElementById('monthly-bets-chart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: data.map(d => d.month),
            datasets: [{
                label: 'ベット数',
                data: data.map(d => d.bets),
                backgroundColor: '#2563eb',
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { title: { display: true, text: 'ベット数' } } },
        },
    });
})();
{% endif %}

// --- DataTables ---
$(document).ready(function() {
    {% if bet_details %}
    $('#bet-table').DataTable({
        pageLength: 25,
        order: [[0, 'desc']],
        language: { url: '//cdn.datatables.net/plug-ins/1.13.7/i18n/ja.json' }
    });
    {% endif %}
    $('.section table.display').not('#bet-table').DataTable({
        paging: false, searching: false, info: false,
        language: { url: '//cdn.datatables.net/plug-ins/1.13.7/i18n/ja.json' }
    });
});
</script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add src/backtest/templates/report.html
git commit -m "feat: HTMLレポートテンプレートを作成"
```

`✶ Insight ─────────────────────────────────────`
**Self-contained design:** The template uses CDN resources (Chart.js, jQuery, DataTables) but embeds ALL data as JSON via Jinja2's `|tojson` filter. The resulting HTML file is fully self-contained — no external data files needed.

**Graceful offline handling:** `{% if bankroll_series %}` guards prevent Chart.js errors when data is empty. The CSS class `.no-data` provides a clear message instead of broken charts.

**DataTables i18n:** The `language: { url: '...ja.json' }` setting loads Japanese localization from the DataTables CDN, making column headers, search, and pagination labels display in Japanese.
`─────────────────────────────────────────────────`

---

## Task 8: Test and verify generate() method

The `generate()` method was already implemented in Task 3 (report.py). Now we test it end-to-end.

**Files:**
- Modify: `tests/test_backtest_report.py` (add test class)

- [ ] **Step 1: Write the failing tests for HTML generation and JSON serialization**

Add to `tests/test_backtest_report.py`:

```python
class TestHtmlGeneration:
    """HTMLレポート生成のテスト"""

    def _make_result_and_history(self) -> tuple:
        from backtest.engine import BacktestResult

        result = BacktestResult(
            total_bets=3, total_stake=300.0, total_return=420.0,
            winning_bets=1, total_roi=1.4, max_drawdown=0.05,
            final_bankroll=100200.0,
        )
        bet_history = [
            {"race_id": "20240105010101", "bet_type": "place", "umaban": 3,
             "stake": 100.0, "odds": 2.4, "result": 240.0,
             "surface": "turf", "distance": 1200, "ev": 1.5,
             "popularity": 3, "bankroll_after": 100200.0},
            {"race_id": "20240110010101", "bet_type": "place", "umaban": 5,
             "stake": 100.0, "odds": 3.0, "result": 0.0,
             "surface": "dirt", "distance": 1600, "ev": 1.3,
             "popularity": 6, "bankroll_after": 100100.0},
            {"race_id": "20240115010101", "bet_type": "place", "umaban": 1,
             "stake": 100.0, "odds": 1.8, "result": 180.0,
             "surface": "turf", "distance": 1800, "ev": 1.6,
             "popularity": 2, "bankroll_after": 100280.0},
        ]
        return result, bet_history

    def test_html_contains_sections(self, tmp_path: Path) -> None:
        """HTMLに全セクションが含まれる"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=tmp_path)
        result, bet_history = self._make_result_and_history()
        path = gen.generate(result, bet_history, train_period="2020-2023", test_period="2024")

        assert path.exists()
        html = path.read_text(encoding="utf-8")
        assert "サマリー" in html
        assert "資金推移" in html
        assert "月次ダッシュボード" in html
        assert "条件分析" in html
        assert "ベット明細" in html
        assert "140.0%" in html  # ROI

    def test_html_with_empty_history(self, tmp_path: Path) -> None:
        """空の bet_history でもHTMLが生成される"""
        from backtest.report import BacktestReportGenerator
        from backtest.engine import BacktestResult

        gen = BacktestReportGenerator(output_dir=tmp_path)
        result = BacktestResult()
        path = gen.generate(result, [])

        assert path.exists()
        html = path.read_text(encoding="utf-8")
        assert "サマリー" in html
        assert "データなし" in html

    def test_output_path(self, tmp_path: Path) -> None:
        """出力パスが data/backtest/backtest_report.html"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=tmp_path)
        result, bet_history = self._make_result_and_history()
        path = gen.generate(result, bet_history)

        assert path.name == "backtest_report.html"
        assert path.parent == tmp_path


class TestBetHistorySerialization:
    """bet_history JSON保存/読み込みのテスト"""

    def test_json_round_trip(self, tmp_path: Path) -> None:
        """JSON保存→読み込みでデータが保持される"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=tmp_path)
        original = [
            {"race_id": "20240101010101", "stake": 100.0, "result": 240.0,
             "surface": "turf", "distance": 1200, "ev": 1.5,
             "popularity": 3, "bankroll_after": 100200.0},
        ]
        json_path = gen.save_bet_history(original)
        loaded = gen.load_bet_history(json_path)

        assert len(loaded) == 1
        assert loaded[0]["race_id"] == "20240101010101"
        assert loaded[0]["ev"] == 1.5
        assert loaded[0]["bankroll_after"] == 100200.0

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """save_bet_history がJSONファイルを作成する"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=tmp_path)
        path = gen.save_bet_history([{"race_id": "20240101010101", "stake": 100.0, "result": 0.0}])
        assert path.exists()
        assert path.suffix == ".json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest_report.py::TestHtmlGeneration -v`
Expected: FAIL (if `save_bet_history` / `load_bet_history` are not yet implemented)

- [ ] **Step 3: Add save/load methods to report.py**

Add to `src/backtest/report.py` (inside the class, after `generate`):

```python
    def save_bet_history(self, bet_history: list[dict[str, Any]]) -> Path:
        """bet_history を JSON に保存"""
        path = self.output_dir / "bet_history.json"
        path.write_text(json.dumps(bet_history, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_bet_history(self, path: Path) -> list[dict[str, Any]]:
        """bet_history JSON を読み込み"""
        return json.loads(path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run all report tests**

Run: `python -m pytest tests/test_backtest_report.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/report.py tests/test_backtest_report.py
git commit -m "feat: HTML生成・bet_history JSON保存/読み込みのテストを追加"
```

---

## Task 9: Add --report CLI flag

**Files:**
- Modify: `scripts/run_backtest.py` (add --report, save bet_history, move output, call generator)
- Modify: `tests/test_backtest_report.py` (add CLI test)

- [ ] **Step 1: Write the failing CLI test**

Add to `tests/test_backtest_report.py`:

```python
class TestCliReportFlag:
    """--report フラグのテスト"""

    @patch("backtest.report.BacktestReportGenerator")
    @patch("pipelines.training_pipeline.TrainingPipelineV5")
    @patch("backtest.engine.BacktestEngine")
    @patch("db.repository.DataRepository")
    @patch("db.parquet_store.ParquetStore")
    def test_report_flag_triggers_generation(
        self,
        mock_store_cls: MagicMock,
        mock_repo_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_pipeline_cls: MagicMock,
        mock_report_gen_cls: MagicMock,
    ) -> None:
        """--report フラグでレポート生成が呼ばれる"""
        # Setup mocks
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.exists.return_value = True

        mock_repo = MagicMock()
        mock_repo_cls.return_value = mock_repo

        mock_models = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_pipeline.run.return_value = mock_models

        from backtest.engine import BacktestResult

        mock_result = BacktestResult(
            total_bets=10, total_stake=1000.0, total_return=1500.0,
            winning_bets=3, total_roi=1.5, max_drawdown=0.05,
            final_bankroll=101500.0,
            bet_history=[
                {"race_id": "20240101010101", "bet_type": "place", "umaban": 1,
                 "stake": 100.0, "odds": 2.4, "result": 240.0,
                 "surface": "turf", "distance": 1200, "ev": 1.5,
                 "popularity": 3, "bankroll_after": 100200.0},
            ],
        )
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.run.return_value = mock_result

        mock_gen = MagicMock()
        mock_report_gen_cls.return_value = mock_gen
        mock_gen.generate.return_value = MagicMock()
        mock_gen.save_bet_history.return_value = MagicMock()

        # sys.argv を直接操作 (main() は argparse で読む)
        with patch("sys.argv", [
            "run_backtest.py",
            "--train-start", "20200101", "--train-end", "20231231",
            "--test-start", "20240101", "--test-end", "20241231",
            "--report",
        ]):
            from scripts.run_backtest import main
            main()

        # Verify report generator was called
        mock_report_gen_cls.assert_called_once()
        mock_gen.generate.assert_called_once()
        mock_gen.save_bet_history.assert_called_once()
        call_args = mock_gen.generate.call_args
        assert call_args[0][0].total_roi == 1.5  # BacktestResult passed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backtest_report.py::TestCliReportFlag -v`
Expected: FAIL (argument error or report not generated)

- [ ] **Step 3: Modify run_backtest.py**

3 changes to `scripts/run_backtest.py`:

**Change 1:** Add `from pathlib import Path` to imports (after `import warnings`):

```python
from pathlib import Path
```

**Change 2:** Add `--report` argument after `--test-end` (after line 42):

```python
    parser.add_argument("--report", action="store_true", help="HTMLレポートを生成")
```

**Change 3:** Replace the JSON save section (lines 126-143) with conditional output logic:

```python
    # JSON出力
    out = {
        "before_roi": before_roi,
        "total_roi": result.total_roi,
        "total_bets": result.total_bets,
        "total_stake": result.total_stake,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "final_bankroll": result.final_bankroll,
        "train_period": [train_start, train_end],
        "test_period": [test_start, test_end],
        "train_seconds": round(elapsed_train),
        "test_seconds": round(elapsed_test),
    }

    # --report フラグ: 全出力を data/backtest/ に集約
    if args.report:
        from backtest.report import BacktestReportGenerator

        output_dir = os.path.join(ROOT, "data", "backtest")
        os.makedirs(output_dir, exist_ok=True)

        gen = BacktestReportGenerator(output_dir=Path(output_dir))
        bet_history_path = gen.save_bet_history(result.bet_history)
        print(f"\nbet_history保存: {bet_history_path}")

        result_path = os.path.join(output_dir, "backtest_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"結果保存: {result_path}")

        train_period_str = f"{train_start} ~ {train_end}"
        test_period_str = f"{test_start} ~ {test_end}"
        report_path = gen.generate(
            result, result.bet_history,
            train_period=train_period_str, test_period=test_period_str,
        )
        print(f"レポート生成: {report_path}")
    else:
        # 従来通りプロジェクトルートに保存
        outpath = os.path.join(ROOT, "backtest_result.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n結果保存: {outpath}")
```

Key changes:
1. Added `--report` argument (line with `store_true`)
2. Added `from pathlib import Path` import at top
3. When `--report` is active: saves to `data/backtest/`, generates HTML report
4. When `--report` is NOT active: existing behavior unchanged (saves to project root)

`✶ Insight ─────────────────────────────────────`
**Output consolidation:** When `--report` is active, ALL output files go to `data/backtest/` — this matches the design spec's requirement. Without `--report`, the existing behavior is preserved (saves `backtest_result.json` to project root).

**Incremental edits:** The 3 changes (import, argparse, conditional block) are applied as targeted edits to the existing file, not a full replacement. This preserves any other modifications and makes the git diff clean.
`─────────────────────────────────────────────────`

- [ ] **Step 4: Run CLI test**

Run: `python -m pytest tests/test_backtest_report.py::TestCliReportFlag -v`
Expected: PASS

- [ ] **Step 5: Run ALL tests**

Run: `python -m pytest tests/test_backtest_report.py tests/test_backtest_engine.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/run_backtest.py tests/test_backtest_report.py
git commit -m "feat: --reportフラグでHTMLレポート生成を追加"
```

---

## Task 10: Final verification

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS (no regressions)

- [ ] **Step 2: Run linter**

Run: `ruff check src/backtest/report.py scripts/run_backtest.py`
Expected: No errors

- [ ] **Step 3: Run formatter check**

Run: `ruff format --check src/backtest/report.py scripts/run_backtest.py`
Expected: No formatting issues

- [ ] **Step 4: Run type checker**

Run: `mypy src/backtest/report.py`
Expected: No type errors

- [ ] **Step 5: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: lint/format fixes for backtest report"
```
