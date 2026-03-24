# Phase A: 基盤構築 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 設計書 v5.5 の Phase A（基盤構築）を実装する。プロジェクトスケルトン・データクラス・DB接続モジュールを整え、以降の Phase B 以降で参照する基盤を完成させる。

**Architecture:** 3タスク構成。A-1（プロジェクトスケルトン）→ A-2（データクラス・型定義）→ A-3（PostgreSQL スキーマ + DB接続）の順に依存関係に従い実装。A-2 と A-3 は A-1 完了後に並行可能だが、A-2 を先に終わらせると A-3 のスキーマ定義で型を参照できる。

**Tech Stack:** Python 3.11+, pyproject.toml (uv/pip), PyYAML, dataclasses, psycopg2 (binary), SQLAlchemy 2.0 (Core only, ORM不使用), EveryDB2 (PostgreSQL)

---

## File Structure Overview

```
keiba-ai/
├── .gitignore                          # A-1
├── pyproject.toml                      # A-1
├── requirements.txt                    # A-1 (pyproject.toml から生成)
├── config/
│   └── settings.yaml                   # A-1
├── src/
│   ├── __init__.py                     # A-1
│   ├── domain/                         # A-2
│   │   ├── __init__.py
│   │   ├── types.py                    # Enum, 型エイリアス
│   │   └── models.py                   # Race, Entry, Bet, OddsSnapshot 等
│   └── db/                             # A-3
│       ├── __init__.py
│       ├── schema.py                   # CREATE TABLE DDL
│       └── connection.py               # DB接続 + 基本クエリ
├── tests/
│   ├── __init__.py
│   ├── test_domain.py                  # A-2
│   └── test_db.py                      # A-3
└── notebooks/                          # A-1 (空ディレクトリ)
```

---

### Task 1: A-1 プロジェクトスケルトン作成

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `config/settings.yaml`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `notebooks/.gitkeep`
- Test: `tests/test_settings.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "keiba-ai"
version = "0.1.0"
description = "競馬AI予測システム v5.5"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "lightgbm>=4.3",
    "psycopg2-binary>=2.9",
    "sqlalchemy>=2.0",
    "pyyaml>=6.0",
    "mlflow>=2.12",
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.4",
    "mypy>=1.10",
    "ipykernel>=6.29",
]

[build-system]
requires = ["setuptools>=69.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

- [ ] **Step 2: Create .gitignore**

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
.eggs/

# Virtual environments
.venv/
venv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Environment
.env
.env.local

# Data
data/
*.csv
*.parquet

# Models
models/
*.pkl
*.lgb
*.joblib

# MLflow
mlruns/

# Playwright
.playwright-mcp/

# Claude
.claude/
```

- [ ] **Step 3: Create config/settings.yaml**

```yaml
database:
  host: "localhost"
  port: 5432
  dbname: "everydb2"
  user: "postgres"
  password: ""  # 環境変数 PGPASSWORD で上書き

paths:
  data_dir: "data"
  model_dir: "models"
  mlflow_tracking_uri: "file:///mlruns"

logging:
  level: "INFO"
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

feature_engine:
  exclude_steeple: true  # 障害レース(TrackCD 51-59)を除外

late_money:
  cancel_threshold: 0.25     # 単勝オッズ25%以上急落→キャンセル
  add_rise_threshold: 0.30   # 単勝オッズ30%以上急騰→追加候補
  cancel_time_minutes: 3     # t-3min で判定
  log_time_minutes: 2        # t-2min でログ

submodel:
  surfaces: ["turf", "dirt"]
  distance_bands:
    turf:
      sprint: [0, 1400]
      mile: [1401, 1700]
      intermediate: [1701, 2100]
      long: [2101, 9999]
    dirt:
      sprint: [0, 1400]
      mile: [1401, 1700]
      intermediate: [1701, 9999]
```

- [ ] **Step 4: Create requirements.txt (from pyproject.toml dependencies)**

```
pandas>=2.2
numpy>=1.26
scikit-learn>=1.4
lightgbm>=4.3
psycopg2-binary>=2.9
sqlalchemy>=2.0
pyyaml>=6.0
mlflow>=2.12
pytest>=8.0
pytest-cov>=5.0
```

- [ ] **Step 5: Create empty __init__.py files and notebooks/.gitkeep**

Create:
- `src/__init__.py` (empty)
- `tests/__init__.py` (empty)
- `notebooks/.gitkeep` (empty)

- [ ] **Step 6: Write the failing test for settings loading**

```python
# tests/test_settings.py
"""config/settings.yaml のロードテスト"""

from pathlib import Path

import yaml


def test_settings_file_exists():
    settings_path = Path("config/settings.yaml")
    assert settings_path.exists(), "config/settings.yaml が存在しません"


def test_settings_has_required_sections():
    settings_path = Path("config/settings.yaml")
    with open(settings_path) as f:
        config = yaml.safe_load(f)

    required_sections = ["database", "paths", "logging", "feature_engine", "late_money", "submodel"]
    for section in required_sections:
        assert section in config, f"settings.yaml に '{section}' セクションがありません"


def test_settings_database_fields():
    settings_path = Path("config/settings.yaml")
    with open(settings_path) as f:
        config = yaml.safe_load(f)

    db = config["database"]
    required_fields = ["host", "port", "dbname", "user"]
    for field in required_fields:
        assert field in db, f"database セクションに '{field}' がありません"


def test_settings_submodel_surfaces():
    settings_path = Path("config/settings.yaml")
    with open(settings_path) as f:
        config = yaml.safe_load(f)

    surfaces = config["submodel"]["surfaces"]
    assert "turf" in surfaces
    assert "dirt" in surfaces
    assert len(surfaces) == 2  # 設計書§6: 2分割のみ
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd /c/Users/hirom/develop/keiba-ai && python -m pytest tests/test_settings.py -v`
Expected: 4 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore config/ requirements.txt src/__init__.py tests/__init__.py tests/test_settings.py notebooks/.gitkeep
git commit -m "feat: プロジェクトスケルトン作成 (A-1)

pyproject.toml, config/settings.yaml, .gitignore,
requirements.txt, ディレクトリ構成, settings.yamlロードテスト"
```

---

### Task 2: A-2 データクラス・型定義

**Files:**
- Create: `src/domain/__init__.py`
- Create: `src/domain/types.py`
- Create: `src/domain/models.py`
- Test: `tests/test_domain.py`

- [ ] **Step 1: Write the failing test for domain models**

```python
# tests/test_domain.py
"""src/domain モジュールのテスト"""

import math
from dataclasses import dataclass

import pytest

from domain.types import BetType, RecoveryState, Surface
from domain.models import (
    Race,
    Entry,
    Bet,
    OddsSnapshot,
    DDState,
    RegimeConfig,
    TwoStageConfig,
)


class TestEnums:
    def test_surface_values(self):
        assert Surface.TURF.value == "turf"
        assert Surface.DIRT.value == "dirt"

    def test_bet_type_values(self):
        assert BetType.WIN.value == "win"
        assert BetType.PLACE.value == "place"
        assert BetType.WIDE.value == "wide"

    def test_recovery_state_values(self):
        assert RecoveryState.NORMAL.value == "normal"
        assert RecoveryState.REDUCED.value == "reduced"
        assert RecoveryState.RECOVERING.value == "recovering"


class TestRace:
    def test_create_race_minimal(self):
        race = Race(
            year=2024,
            month_day="0324",
            jyo_cd="05",
            kaiji="03",
            nichiji="02",
            race_num="08",
            track_cd=11,
            distance=1600,
            tenko_cd=1,
            baba_cd=1,
            syubetu_cd=13,
            jyoken_cd=999,
            grade_cd="_",
            field_size=18,
        )
        assert race.surface == Surface.TURF
        assert race.distance == 1600
        assert race.distance_band == "mile"

    def test_surface_dirt(self):
        race = Race(
            year=2024, month_day="0324", jyo_cd="05", kaiji="03", nichiji="02",
            race_num="08", track_cd=23, distance=1200, tenko_cd=1, baba_cd=1,
            syubetu_cd=13, jyoken_cd=999, grade_cd="_", field_size=14,
        )
        assert race.surface == Surface.DIRT

    def test_race_id_format(self):
        race = Race(
            year=2024, month_day="0324", jyo_cd="05", kaiji="03", nichiji="02",
            race_num="08", track_cd=11, distance=1600, tenko_cd=1, baba_cd=1,
            syubetu_cd=13, jyoken_cd=999, grade_cd="_", field_size=18,
        )
        assert race.race_id == "2024032405030208"

    def test_is_good_track(self):
        race = Race(
            year=2024, month_day="0324", jyo_cd="05", kaiji="03", nichiji="02",
            race_num="08", track_cd=11, distance=1600, tenko_cd=1, baba_cd=1,
            syubetu_cd=13, jyoken_cd=999, grade_cd="_", field_size=18,
        )
        assert race.is_good_track is True

    def test_is_soft_track(self):
        race = Race(
            year=2024, month_day="0324", jyo_cd="05", kaiji="03", nichiji="02",
            race_num="08", track_cd=11, distance=1600, tenko_cd=1, baba_cd=3,
            syubetu_cd=13, jyoken_cd=999, grade_cd="_", field_size=18,
        )
        assert race.is_good_track is False
        assert race.is_soft_track is True


class TestEntry:
    def test_create_entry(self):
        entry = Entry(
            race_id="2024032405030208",
            umaban=5,
            ketto_num="0001234567",
            finish_pos=1,
            win_odds_actual=3.2,
            popularity_rank=2,
            running_style=2,
            ba_taijyu=480,
            zogen_fugo=2,
            zogen_sa=-4,
            kisyu_code="01056",
            chokyosi_code="01023",
        )
        assert entry.is_winner is True
        assert entry.is_place is True

    def test_entry_not_winner(self):
        entry = Entry(
            race_id="2024032405030208", umaban=5, ketto_num="0001234567",
            finish_pos=4, win_odds_actual=15.8, popularity_rank=8,
            running_style=4, ba_taijyu=476, zogen_fugo=1, zogen_sa=2,
            kisyu_code="01056", chokyosi_code="01023",
        )
        assert entry.is_winner is False
        assert entry.is_place is False

    def test_entry_cancelled(self):
        entry = Entry(
            race_id="2024032405030208", umaban=5, ketto_num="0001234567",
            finish_pos=0, win_odds_actual=0.0, popularity_rank=0,
            running_style=0, ba_taijyu=0, zogen_fugo=0, zogen_sa=0,
            kisyu_code="", chokyosi_code="",
        )
        assert entry.is_winner is False
        assert entry.is_cancelled is True


class TestBet:
    def test_create_bet(self):
        bet = Bet(
            race_id="2024032405030208",
            umaban=5,
            bet_type=BetType.WIN,
            odds=3.2,
            ev_lower_corrected=1.15,
            stake=200,
        )
        assert bet.bet_type == BetType.WIN
        assert bet.stake == 200

    def test_bet_minimum_stake(self):
        bet = Bet(
            race_id="2024032405030208", umaban=5, bet_type=BetType.WIN,
            odds=3.2, ev_lower_corrected=1.05, stake=50,
        )
        assert bet.stake < 100
        assert bet.is_valid is False


class TestOddsSnapshot:
    def test_create_snapshot(self):
        snapshot = OddsSnapshot(
            race_id="2024032405030208",
            happyo_time="03241505",
            umaban=5,
            tan_odds=3.2,
            fuku_odds=1.4,
        )
        assert snapshot.umaban == 5
        assert snapshot.tan_odds == 3.2


class TestDDState:
    def test_create_dd_state(self):
        state = DDState(
            current_dd=0.08,
            rolling_roi=1.05,
            n_bets_eval=150,
            recovery_state=RecoveryState.NORMAL,
        )
        assert state.recovery_state == RecoveryState.NORMAL


class TestTwoStageConfig:
    def test_defaults(self):
        config = TwoStageConfig()
        assert config.hit_metric == "auc"
        assert config.hit_rounds == 500
        assert config.return_rounds == 300
        assert config.min_hit_samples == 200


class TestRegimeConfig:
    def test_defaults(self):
        config = RegimeConfig()
        assert config.window == 200
        assert config.min_samples == 100
        assert config.fav_rate_aggressive == 0.28
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /c/Users/hirom/develop/keiba-ai && python -m pytest tests/test_domain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'domain'`

- [ ] **Step 3: Implement src/domain/types.py**

```python
# src/domain/types.py
"""Enum 定義と型エイリアス"""

from enum import Enum


class Surface(str, Enum):
    """芝/ダートのサーフェス"""
    TURF = "turf"
    DIRT = "dirt"


class BetType(str, Enum):
    """投票タイプ"""
    WIN = "win"
    PLACE = "place"
    WIDE = "wide"


class RecoveryState(str, Enum):
    """ドローダウン回復状態（DDコントローラー用）"""
    NORMAL = "normal"
    REDUCED = "reduced"
    RECOVERING = "recovering"


class RegimeState(str, Enum):
    """市場レジーム状態"""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    COLLAPSED = "collapsed"
```

- [ ] **Step 4: Implement src/domain/models.py**

```python
# src/domain/models.py
"""データクラス定義（Race, Entry, Bet, OddsSnapshot, DDState 等）"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from domain.types import BetType, RecoveryState, Surface


def _surface_from_track_cd(track_cd: int) -> Surface:
    """TrackCD から Surface を判定（設計書 everydb2-data-reference §3.1）"""
    if 10 <= track_cd <= 22:
        return Surface.TURF
    elif 23 <= track_cd <= 29:
        return Surface.DIRT
    else:
        raise ValueError(f"未対応の TrackCD: {track_cd} (障害51-59は除外前提)")


def _distance_band(surface: Surface, distance: int) -> str:
    """サーフェスと距離から距離帯を返す（設計書 everydb2-data-reference §3.1）"""
    if surface == Surface.TURF:
        if distance <= 1400:
            return "sprint"
        elif distance <= 1700:
            return "mile"
        elif distance <= 2100:
            return "intermediate"
        else:
            return "long"
    else:  # DIRT
        if distance <= 1400:
            return "sprint"
        elif distance <= 1700:
            return "mile"
        else:
            return "intermediate"


@dataclass(frozen=True)
class Race:
    """レース情報（n_race テーブル対応）

    複合主キー: (year, month_day, jyo_cd, kaiji, nichiji, race_num)
    """
    year: int
    month_day: str       # MMDD
    jyo_cd: str          # 場所コード 01-10
    kaiji: str           # 回次
    nichiji: str         # 日次
    race_num: str        # レース番号
    track_cd: int        # トラックコード
    distance: int        # 距離(m)
    tenko_cd: int        # 天候コード
    baba_cd: int         # 馬場状態コード
    syubetu_cd: str      # 種別コード
    jyoken_cd: str       # 条件コード
    grade_cd: str        # グレードコード
    field_size: int      # 頭数

    # --- 計算プロパティ ---
    @property
    def surface(self) -> Surface:
        return _surface_from_track_cd(self.track_cd)

    @property
    def distance_band(self) -> str:
        return _distance_band(self.surface, self.distance)

    @property
    def race_id(self) -> str:
        """複合主キーを文字列化: YYYYMMDDJyoKaiNiRace"""
        return f"{self.year}{self.month_day}{self.jyo_cd}{self.kaiji}{self.nichiji}{self.race_num}"

    @property
    def is_good_track(self) -> bool:
        """良 or 稍重"""
        return self.baba_cd in (1, 2)

    @property
    def is_soft_track(self) -> bool:
        """重 or 不良"""
        return self.baba_cd in (3, 4)

    @property
    def is_steeple(self) -> bool:
        """障害レース"""
        return self.track_cd >= 51

    @property
    def grade_name(self) -> str:
        grade_map = {"A": "G1", "B": "G2", "C": "G3", "D": "重賞", "E": "特別"}
        return grade_map.get(self.grade_cd, "一般")


@dataclass
class Entry:
    """出走馬情報（n_uma_race テーブル対応）"""
    race_id: str
    umaban: int              # 馬番
    ketto_num: str           # 血統番号
    finish_pos: int          # 確定着順 (1=1着, 0=取消等)
    win_odds_actual: float   # 確定単勝オッズ
    popularity_rank: int     # 人気順位
    running_style: int       # 脚質 (1=逃げ, 2=先行, 3=差し, 4=追込, 0=不明)
    ba_taijyu: float         # 馬体重
    zogen_fugo: int          # 体重増減符号 (1=増, 2=減, 3=不变)
    zogen_sa: float          # 体重増減幅
    kisyu_code: str          # 騎手コード
    chokyosi_code: str       # 調教師コード

    @property
    def is_winner(self) -> bool:
        return self.finish_pos == 1

    @property
    def is_place(self) -> bool:
        return 1 <= self.finish_pos <= 3

    @property
    def is_cancelled(self) -> bool:
        return self.finish_pos == 0

    @property
    def running_style_name(self) -> str:
        style_map = {1: "逃げ", 2: "先行", 3: "差し", 4: "追込"}
        return style_map.get(self.running_style, "不明")


@dataclass
class Bet:
    """投票情報"""
    race_id: str
    umaban: int
    bet_type: BetType
    odds: float                  # オッズ
    ev_lower_corrected: float    # EV下限値（補正済み）
    stake: float                 # 投票金額
    result: Optional[float] = None  # 払戻金（確定後）

    @property
    def is_valid(self) -> bool:
        """最低投票額 100円以上"""
        return self.stake >= 100

    @property
    def profit(self) -> float:
        """利益（払戻 - 投票額）"""
        if self.result is None:
            return 0.0
        return self.result - self.stake


@dataclass
class OddsSnapshot:
    """時系列オッズスナップショット（n_jodds_tanpuku テーブル対応）"""
    race_id: str
    happyo_time: str    # 発表時刻 MMDDHHmm
    umaban: int
    tan_odds: float     # 単勝オッズ
    fuku_odds: float    # 複勝オッズ


@dataclass
class DDState:
    """ドローダウン状態（§9 DDコントローラー用）"""
    current_dd: float
    rolling_roi: float
    n_bets_eval: int
    recovery_state: RecoveryState = RecoveryState.NORMAL


@dataclass
class RegimeConfig:
    """レジーム検知設定（§9.5）"""
    window: int = 200
    min_samples: int = 100
    fav_rate_aggressive: float = 0.28
    fav_rate_collapsed: float = 0.18
    overround_base: float = 0.20
    retrain_trigger: int = 100


@dataclass
class TwoStageConfig:
    """2段階モデルハイパーパラメータ（§2）"""
    hit_metric: str = "auc"
    hit_leaves: int = 31
    hit_lr: float = 0.03
    hit_rounds: int = 500
    return_metric: str = "mae"
    return_leaves: int = 15
    return_lr: float = 0.03
    return_rounds: int = 300
    min_hit_samples: int = 200
```

- [ ] **Step 5: Implement src/domain/__init__.py**

```python
# src/domain/__init__.py
from domain.types import BetType, RecoveryState, RegimeState, Surface
from domain.models import (
    Bet,
    DDState,
    Entry,
    OddsSnapshot,
    Race,
    RegimeConfig,
    TwoStageConfig,
)

__all__ = [
    "BetType",
    "RecoveryState",
    "RegimeState",
    "Surface",
    "Bet",
    "DDState",
    "Entry",
    "OddsSnapshot",
    "Race",
    "RegimeConfig",
    "TwoStageConfig",
]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /c/Users/hirom/develop/keiba-ai && python -m pytest tests/test_domain.py -v`
Expected: ALL PASSED

- [ ] **Step 7: Commit**

```bash
git add src/domain/ tests/test_domain.py
git commit -m "feat: データクラス・型定義 (A-2)

Race, Entry, Bet, OddsSnapshot, DDState, RegimeConfig,
TwoStageConfig, Surface/BetType/RecoveryState enum 定義"
```

---

### Task 3: A-3 PostgreSQL スキーマ定義 + DB接続モジュール

**Files:**
- Create: `src/db/__init__.py`
- Create: `src/db/schema.py`
- Create: `src/db/connection.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test for DB connection and schema**

```python
# tests/test_db.py
"""src/db モジュールのテスト（モックDB使用・実際のDB接続不要）"""

import os
from unittest.mock import MagicMock, patch

import pytest

from db.connection import DatabaseConnection
from db.schema import (
    SCHEMA_RAW,
    SCHEMA_ODDS_HISTORY,
    SCHEMA_FEATURE,
    SCHEMA_PREDICTION,
    SCHEMA_BETTING,
    ALL_CREATE_STATEMENTS,
)


class TestSchemaDefinitions:
    def test_raw_schema_contains_races_table(self):
        assert "CREATE TABLE IF NOT EXISTS raw.races" in SCHEMA_RAW

    def test_raw_schema_contains_entries_table(self):
        assert "CREATE TABLE IF NOT EXISTS raw.entries" in SCHEMA_RAW

    def test_odds_history_schema_contains_snapshots_table(self):
        assert "CREATE TABLE IF NOT EXISTS odds_history.odds_snapshots" in SCHEMA_ODDS_HISTORY

    def test_odds_history_schema_contains_time_series_table(self):
        assert "CREATE TABLE IF NOT EXISTS odds_history.odds_time_series" in SCHEMA_ODDS_HISTORY

    def test_feature_schema_exists(self):
        assert "CREATE TABLE IF NOT EXISTS feature.features" in SCHEMA_FEATURE

    def test_prediction_schema_exists(self):
        assert "CREATE TABLE IF NOT EXISTS prediction.predictions" in SCHEMA_PREDICTION

    def test_betting_schema_contains_bets_table(self):
        assert "CREATE TABLE IF NOT EXISTS betting.bets" in SCHEMA_BETTING

    def test_all_schemas_list(self):
        assert len(ALL_CREATE_STATEMENTS) == 5

    def test_race_primary_key(self):
        """複合主キーの確認"""
        assert "PRIMARY KEY (year, month_day, jyo_cd, kaiji, nichiji, race_num)" in SCHEMA_RAW

    def test_entries_foreign_key(self):
        assert "REFERENCES raw.races" in SCHEMA_RAW

    def test_raw_schema_race_id_generated(self):
        """race_id は GENERATED ALWAYS AS で複合PKから自動生成"""
        assert "race_id" in SCHEMA_RAW
        assert "GENERATED ALWAYS AS" in SCHEMA_RAW
        assert "UNIQUE" in SCHEMA_RAW

    def test_raw_schema_surface_computed_column(self):
        """surface は GENERATED COLUMN で計算"""
        assert "surface" in SCHEMA_RAW


class TestDatabaseConnection:
    def test_connection_string_from_settings(self):
        """settings.yaml から接続文字列を正しく生成"""
        mock_settings = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "dbname": "everydb2",
                "user": "postgres",
                "password": "",
            }
        }
        with patch("db.connection._load_settings", return_value=mock_settings):
            conn = DatabaseConnection()
            expected = "postgresql+psycopg2://postgres@localhost:5432/everydb2"
            assert conn._connection_url == expected

    def test_connection_string_with_password(self):
        mock_settings = {
            "database": {
                "host": "db.example.com",
                "port": 5433,
                "dbname": "everydb2",
                "user": "app_user",
                "password": "secret",
            }
        }
        with patch("db.connection._load_settings", return_value=mock_settings):
            conn = DatabaseConnection()
            expected = "postgresql+psycopg2://app_user:secret@db.example.com:5433/everydb2"
            assert conn._connection_url == expected

    def test_connection_string_uses_env_password(self):
        """環境変数 PGPASSWORD で password を上書き"""
        mock_settings = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "dbname": "everydb2",
                "user": "postgres",
                "password": "",
            }
        }
        with (
            patch.dict(os.environ, {"PGPASSWORD": "env_secret"}),
            patch("db.connection._load_settings", return_value=mock_settings),
        ):
            conn = DatabaseConnection()
            assert "env_secret" in conn._connection_url
            assert conn._connection_url.startswith("postgresql+psycopg2://postgres:env_secret@")

    def test_get_engine_returns_engine(self):
        """engine はキャッシュされる"""
        mock_settings = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "dbname": "everydb2",
                "user": "postgres",
                "password": "",
            }
        }
        with (
            patch("db.connection._load_settings", return_value=mock_settings),
            patch("db.connection.create_engine") as mock_create_engine,
        ):
            conn = DatabaseConnection()
            engine1 = conn.get_engine()
            engine2 = conn.get_engine()
            mock_create_engine.assert_called_once()
            assert engine1 is engine2

    def test_create_schemas_executes_all(self):
        """全ステートメントが個別に実行される（DDL分割対応）

        SCHEMA_RAW: 4文, SCHEMA_ODDS_HISTORY: 4文, SCHEMA_FEATURE: 2文,
        SCHEMA_PREDICTION: 2文, SCHEMA_BETTING: 5文 = 合計17文
        """
        mock_settings = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "dbname": "everydb2",
                "user": "postgres",
                "password": "",
            }
        }
        with (
            patch("db.connection._load_settings", return_value=mock_settings),
            patch("db.connection.create_engine") as mock_create_engine,
        ):
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            conn = DatabaseConnection()
            conn.create_schemas()

            # 17個の個別SQLステートメントが実行される
            assert mock_engine.begin.call_count == 17
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /c/Users/hirom/develop/keiba-ai && python -m pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'db'`

- [ ] **Step 3: Implement src/db/schema.py**

```python
# src/db/schema.py
"""PostgreSQL スキーマ定義（EveryDB2 対応）

5つのスキーマ:
- raw: EveryDB2 生データのローカルコピー
- odds_history: 時系列オッズ（JODDS_TANPUKU対応）
- feature: 特徴量エンジン出力
- prediction: モデル予測結果
- betting: 投票記録

EveryDB2の外部テーブル(n_race, n_uma_race 等)は読み取り専用。
本スキーマは特徴量・予測・投票の保存用。
"""

SCHEMA_RAW = """
CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE IF NOT EXISTS raw.races (
    year          INTEGER NOT NULL,
    month_day     VARCHAR(4) NOT NULL,
    jyo_cd        VARCHAR(2) NOT NULL,
    kaiji         VARCHAR(2) NOT NULL,
    nichiji       VARCHAR(2) NOT NULL,
    race_num      VARCHAR(2) NOT NULL,
    track_cd      INTEGER NOT NULL,
    distance      INTEGER NOT NULL,
    tenko_cd      INTEGER NOT NULL,
    baba_cd       INTEGER NOT NULL,
    syubetu_cd    VARCHAR(4) NOT NULL,
    jyoken_cd     VARCHAR(4) NOT NULL,
    grade_cd      VARCHAR(1) NOT NULL DEFAULT '_',
    field_size    INTEGER NOT NULL,
    -- 複合PKを文字列化したrace_id（子テーブルからのFK参照用）
    race_id       VARCHAR(16) GENERATED ALWAYS AS (
        year::text || month_day || jyo_cd || kaiji || nichiji || race_num
    ) STORED UNIQUE,
    surface       VARCHAR(5) GENERATED ALWAYS AS (
        CASE
            WHEN track_cd BETWEEN 10 AND 22 THEN 'turf'
            WHEN track_cd BETWEEN 23 AND 29 THEN 'dirt'
            ELSE 'exclude'
        END
    ) STORED,
    distance_band VARCHAR(20) GENERATED ALWAYS AS (
        CASE
            WHEN track_cd BETWEEN 10 AND 22 THEN
                CASE
                    WHEN distance <= 1400 THEN 'sprint'
                    WHEN distance <= 1700 THEN 'mile'
                    WHEN distance <= 2100 THEN 'intermediate'
                    ELSE 'long'
                END
            WHEN track_cd BETWEEN 23 AND 29 THEN
                CASE
                    WHEN distance <= 1400 THEN 'sprint'
                    WHEN distance <= 1700 THEN 'mile'
                    ELSE 'intermediate'
                END
            ELSE NULL
        END
    ) STORED,
    PRIMARY KEY (year, month_day, jyo_cd, kaiji, nichiji, race_num)
);

CREATE TABLE IF NOT EXISTS raw.entries (
    race_id       VARCHAR(16) NOT NULL REFERENCES raw.races ON DELETE CASCADE,
    umaban        INTEGER NOT NULL,
    ketto_num     VARCHAR(10) NOT NULL,
    finish_pos    INTEGER NOT NULL DEFAULT 0,
    finish_time   FLOAT,
    haron_time_l3 FLOAT,
    ninki         INTEGER,
    win_odds      FLOAT,
    ba_taijyu     FLOAT,
    zogen_fugo    INTEGER,
    zogen_sa      FLOAT,
    kisyu_code    VARCHAR(5),
    chokyosi_code VARCHAR(5),
    kyakusitu     INTEGER,
    honsyokin     INTEGER,
    PRIMARY KEY (race_id, umaban)
);

CREATE TABLE IF NOT EXISTS raw.payouts (
    race_id       VARCHAR(16) NOT NULL REFERENCES raw.races ON DELETE CASCADE,
    tan_umaban    INTEGER,
    tan_pay       FLOAT,
    fuku_umaban1  INTEGER,  fuku_pay1  FLOAT,
    fuku_umaban2  INTEGER,  fuku_pay2  FLOAT,
    fuku_umaban3  INTEGER,  fuku_pay3  FLOAT,
    fuku_umaban4  INTEGER,  fuku_pay4  FLOAT,
    fuku_umaban5  INTEGER,  fuku_pay5  FLOAT,
    PRIMARY KEY (race_id)
);
"""

SCHEMA_ODDS_HISTORY = """
CREATE SCHEMA IF NOT EXISTS odds_history;

CREATE TABLE IF NOT EXISTS odds_history.odds_snapshots (
    race_id    VARCHAR(16) NOT NULL,
    umaban     INTEGER NOT NULL,
    tan_odds   FLOAT,
    fuku_odds  FLOAT,
    PRIMARY KEY (race_id, umaban)
);

CREATE TABLE IF NOT EXISTS odds_history.odds_time_series (
    race_id     VARCHAR(16) NOT NULL,
    happyo_time VARCHAR(8) NOT NULL,  -- MMDDHHmm
    umaban      INTEGER NOT NULL,
    tan_odds    FLOAT,
    fuku_odds   FLOAT,
    PRIMARY KEY (race_id, happyo_time, umaban)
);

CREATE TABLE IF NOT EXISTS odds_history.wide_odds (
    race_id     VARCHAR(16) NOT NULL,
    kumi        VARCHAR(5) NOT NULL,  -- "3-7" 形式
    odds_low    FLOAT,
    odds_high   FLOAT,
    PRIMARY KEY (race_id, kumi)
);
"""

SCHEMA_FEATURE = """
CREATE SCHEMA IF NOT EXISTS feature;

CREATE TABLE IF NOT EXISTS feature.features (
    race_id     VARCHAR(16) NOT NULL,
    umaban      INTEGER NOT NULL,
    surface     VARCHAR(5) NOT NULL,
    feature_data JSONB NOT NULL,  -- 特徴量の辞書をJSONで保存
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (race_id, umaban)
);
"""

SCHEMA_PREDICTION = """
CREATE SCHEMA IF NOT EXISTS prediction;

CREATE TABLE IF NOT EXISTS prediction.predictions (
    race_id               VARCHAR(16) NOT NULL,
    umaban                INTEGER NOT NULL,
    surface               VARCHAR(5) NOT NULL,
    p_ability_win         FLOAT,
    p_ability_place       FLOAT,
    signed_log_error_win  FLOAT,
    abs_log_error_win     FLOAT,
    p_win_pred            FLOAT,
    ev_win                FLOAT,
    p_win_corrected       FLOAT,
    ev_win_corrected      FLOAT,
    ev_lower_win_corrected FLOAT,
    p_place_pred          FLOAT,
    ev_place              FLOAT,
    ev_lower_place        FLOAT,
    wide_score_adj        FLOAT,
    predicted_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (race_id, umaban)
);
"""

SCHEMA_BETTING = """
CREATE SCHEMA IF NOT EXISTS betting;

CREATE TABLE IF NOT EXISTS betting.bets (
    id                    SERIAL PRIMARY KEY,
    race_id               VARCHAR(16) NOT NULL,
    umaban                INTEGER NOT NULL,
    bet_type              VARCHAR(5) NOT NULL,
    odds                  FLOAT NOT NULL,
    ev_lower_corrected    FLOAT NOT NULL,
    stake                 INTEGER NOT NULL,
    result_payout         FLOAT,
    profit                FLOAT,
    regime_state          VARCHAR(15),
    recovery_state        VARCHAR(15),
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settled_at            TIMESTAMP
);

CREATE INDEX idx_bets_race_id ON betting.bets (race_id);
CREATE INDEX idx_bets_created_at ON betting.bets (created_at);
CREATE INDEX idx_bets_bet_type ON betting.bets (bet_type);
"""

ALL_CREATE_STATEMENTS = [
    SCHEMA_RAW,
    SCHEMA_ODDS_HISTORY,
    SCHEMA_FEATURE,
    SCHEMA_PREDICTION,
    SCHEMA_BETTING,
]
```

- [ ] **Step 4: Implement src/db/connection.py**

```python
# src/db/connection.py
"""PostgreSQL DB接続モジュール（SQLAlchemy Core 使用）"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from db.schema import ALL_CREATE_STATEMENTS


def _load_settings() -> dict:
    """config/settings.yaml をロード"""
    settings_path = Path("config/settings.yaml")
    if not settings_path.exists():
        raise FileNotFoundError("config/settings.yaml が見つかりません")
    with open(settings_path) as f:
        return yaml.safe_load(f)


class DatabaseConnection:
    """データベース接続を管理するクラス（シングルトンエンジン）"""

    def __init__(self, settings_path: Optional[str] = None):
        settings = _load_settings()
        db = settings["database"]

        password = os.environ.get("PGPASSWORD", db.get("password", ""))
        if password:
            self._connection_url = (
                f"postgresql+psycopg2://{db['user']}:{password}"
                f"@{db['host']}:{db['port']}/{db['dbname']}"
            )
        else:
            self._connection_url = (
                f"postgresql+psycopg2://{db['user']}"
                f"@{db['host']}:{db['port']}/{db['dbname']}"
            )
        self._engine: Optional[Engine] = None

    def get_engine(self) -> Engine:
        """SQLAlchemy エンジンを取得（キャッシュ）"""
        if self._engine is None:
            self._engine = create_engine(
                self._connection_url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )
        return self._engine

    def create_schemas(self) -> None:
        """全スキーマとテーブルを作成（冪等）

        SQLAlchemy text() は単一SQL文のみ実行可能なため、
        セミコロンで分割して個別に実行する。
        """
        engine = self.get_engine()
        for ddl in ALL_CREATE_STATEMENTS:
            for statement in ddl.split(";"):
                stmt = statement.strip()
                if stmt:
                    with engine.begin() as conn:
                        conn.execute(text(stmt))

    def load_races(self, start_date: str, end_date: str) -> "pd.DataFrame":
        """指定期間のレースデータをDataFrameで取得"""
        import pandas as pd

        engine = self.get_engine()
        sql = text("""
            SELECT * FROM raw.races
            WHERE (year || month_day)::int BETWEEN :start AND :end
            AND track_cd NOT BETWEEN 51 AND 59
            ORDER BY year, month_day, jyo_cd, kaiji, nichiji, race_num
        """)
        return pd.read_sql(sql, engine, params={"start": start_date, "end": end_date})

    def load_entries_with_results(self, start_date: str, end_date: str) -> "pd.DataFrame":
        """指定期間の出走馬データをDataFrameで取得"""
        import pandas as pd

        engine = self.get_engine()
        sql = text("""
            SELECT e.* FROM raw.entries e
            JOIN raw.races r ON e.race_id = r.race_id
            WHERE (r.year || r.month_day)::int BETWEEN :start AND :end
            AND r.track_cd NOT BETWEEN 51 AND 59
            AND e.finish_pos > 0
            ORDER BY e.race_id, e.umaban
        """)
        return pd.read_sql(sql, engine, params={"start": start_date, "end": end_date})

    def load_odds_snapshots(self, start_date: str, end_date: str) -> "pd.DataFrame":
        """指定期間のオッズスナップショットをDataFrameで取得"""
        import pandas as pd

        engine = self.get_engine()
        sql = text("""
            SELECT o.* FROM odds_history.odds_snapshots o
            JOIN raw.races r ON o.race_id = r.race_id
            WHERE (r.year || r.month_day)::int BETWEEN :start AND :end
            AND r.track_cd NOT BETWEEN 51 AND 59
            ORDER BY o.race_id, o.umaban
        """)
        return pd.read_sql(sql, engine, params={"start": start_date, "end": end_date})

    def load_odds_time_series(self, race_id: str) -> "pd.DataFrame":
        """特定レースの時系列オッズを取得"""
        import pandas as pd

        engine = self.get_engine()
        sql = text("""
            SELECT * FROM odds_history.odds_time_series
            WHERE race_id = :race_id
            ORDER BY happyo_time, umaban
        """)
        return pd.read_sql(sql, engine, params={"race_id": race_id})

    def save_predictions(self, df: "pd.DataFrame") -> None:
        """予測結果を prediction.predictions に保存"""
        engine = self.get_engine()
        df.to_sql("predictions", engine, schema="prediction", if_exists="append", index=False)

    def save_bets(self, df: "pd.DataFrame") -> None:
        """投票記録を betting.bets に保存"""
        engine = self.get_engine()
        df.to_sql("bets", engine, schema="betting", if_exists="append", index=False)
```

- [ ] **Step 5: Implement src/db/__init__.py**

```python
# src/db/__init__.py
from db.connection import DatabaseConnection
from db.schema import ALL_CREATE_STATEMENTS

__all__ = [
    "DatabaseConnection",
    "ALL_CREATE_STATEMENTS",
]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /c/Users/hirom/develop/keiba-ai && python -m pytest tests/test_db.py -v`
Expected: ALL PASSED

- [ ] **Step 7: Run all tests together**

Run: `cd /c/Users/hirom/develop/keiba-ai && python -m pytest tests/ -v`
Expected: ALL PASSED (test_settings.py + test_domain.py + test_db.py)

- [ ] **Step 8: Commit**

```bash
git add src/db/ tests/test_db.py
git commit -m "feat: PostgreSQL スキーマ定義 + DB接続モジュール (A-3)

5スキーマ(raw/odds_history/feature/prediction/betting)、
SQLAlchemy Core 接続、race/entry/odds データローダー"
```

---

## Summary

| Task | Description | Files | Dependencies |
|------|-------------|-------|-------------|
| A-1 | プロジェクトスケルトン | pyproject.toml, .gitignore, config/settings.yaml, requirements.txt | なし |
| A-2 | データクラス・型定義 | src/domain/types.py, src/domain/models.py | A-1 |
| A-3 | DB スキーマ + 接続 | src/db/schema.py, src/db/connection.py | A-1 |

**全テスト実行コマンド:** `python -m pytest tests/ -v`

**完了後の次のステップ:** Phase B（特徴量エンジン）へ移行可能。
