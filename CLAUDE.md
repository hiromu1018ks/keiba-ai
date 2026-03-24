# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

競馬AI予測システム v5.5 — 統計的 horse racing prediction system (単勝/複勝/ワイド)。
LightGBM + scikit-learn で ML モデルを構築し、PostgreSQL (EveryDB2/JRA-VAN DataLab) をデータソースとする。
MLflow で実験管理。設計書は `docs/design.md` (v5.5, ~2900行)。

## Development Environment

### Python (mise)

```bash
# Python 3.11 を mise でインストール・アクティベート
mise install
mise activate

# 依存インストール
pip install -e ".[dev]"
```

`mise.toml` で Python 3.11 を固定。`pip install -e ".[dev]"` で ruff, mypy, ipykernel も含む。

### Common Commands

```bash
# テスト実行（DB不要、全テスト mock 使用）
python -m pytest tests/ -v

# 単一テストファイル
python -m pytest tests/test_domain.py -v

# カバレッジ付き
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# リント (ruff)
ruff check src/ tests/

# フォーマット
ruff format --check src/ tests/

# 型チェック (mypy)
mypy src/
```

### Database

PostgreSQL が `localhost:5432/everydb2` で稼働前提。パスワードは環境変数 `PGPASSWORD` で上書き。
設定ファイル: `config/settings.yaml`

## Architecture

### Current State (Phase A: Foundation)

```
src/
├── domain/          # データクラス・型定義 (Race, Entry, Bet, OddsSnapshot, DDState, etc.)
│   ├── types.py     # Enum: Surface, BetType, RecoveryState, RegimeState
│   └── models.py    # frozen dataclass群 (computed properties付き)
└── db/
    ├── schema.py    # PostgreSQL DDL (5スキーマ: raw, odds_history, feature, prediction, betting)
    └── connection.py # SQLAlchemy Core接続 (ORM不使用), データローダー/セーバー
```

- **SQLAlchemy Core のみ使用** — ORM は不使用
- **Race識別子**: 複合PK `(year, month_day, jyo_cd, kaiji, nichiji, race_num)` + `GENERATED ALWAYS AS` で `race_id` 文字列生成
- **EveryDB2外部テーブル** (読取専用): `n_race`, `n_uma_race`, `n_uma`, `n_harai`, `n_odds_tanpuku`, `n_odds_wide` 等
- **import path**: `from domain.types import ...` (pythonpath = `[".", "src"]` 設定済み)

### Planned Phases (design.md 参照)

Phase A (done) → B (features/) → C (models/) → D (betting/) → E (backtest/) → F (automation/, monitoring/)

## Key Design Decisions

- **2段階モデル**: P(hit) × E(odds|hit) でゼロ偏重を排除。Stage1=能力モデル, Stage2=的中時払戻回帰
- **Market Model**: 差分 log_error のみ出力（p_market_pred は Stage2 に入れない）
- **EV補正**: P補正 (binary objective, init_score=logit(p_pred)) × E補正 (weight=1/√p) の2モデルに分解
- **RegimeDetector**: 市場指標 (fav_rate × overround) で 3状態分類 (aggressive/conservative/collapsed)
- **サブモデル**: 芝/ダートの2分割のみ

## Code Style

- Ruff: target py311, line-length=100, rules=E/F/I/N/W
- Mypy: `disallow_untyped_defs = true` (全関数に型アノテーション必須)
- テストは DB不要 (全て mock) — `unittest.mock` を使用
- コミットメッセージ: Conventional Commits (日本語)

## Configuration

`config/settings.yaml` — database, paths, logging, feature_engine, late_money, submodel の設定。
DB接続時は `_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent` でプロジェクトルートを解決。
