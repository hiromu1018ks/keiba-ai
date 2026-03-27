# 開発ガイド

本プロジェクトへの開発参加に必要な環境セットアップ、コーディング規約、テスト実行方法を説明します。
詳細な設定やアーキテクチャについては [CLAUDE.md](../../CLAUDE.md) および [設計書](../../docs/design.md) を参照してください。

---

## 開発環境のセットアップ

### 前提条件

| ソフトウェア | バージョン | 用途 |
|---|---|---|
| Python | 3.11 | 実行環境（`mise.toml` で固定） |
| PostgreSQL | 15+ | データソース（EveryDB2） |
| mise | 最新版 | Python バージョン管理 |
| Git | 最新版 | バージョン管理 |

PostgreSQL は `localhost:5432/everydb2` で稼働していることを前提とします。
データベース設定の詳細は [設定リファレンス](03_configuration.md) および [CLAUDE.md > Database](../../CLAUDE.md) を参照してください。

### セットアップ手順

```bash
# 1. リポジトリのクローン
git clone <repository-url>
cd keiba-ai

# 2. Python 3.11 のインストール・アクティベート
mise install
mise activate

# 3. 依存パッケージのインストール（dev ツール含む）
pip install -e ".[dev]"
```

`pip install -e ".[dev]"` により、プロジェクト本体に加えて以下の開発ツールがインストールされます：

- **ruff** — Linter / Formatter
- **mypy** — 静的型チェッカー
- **ipykernel** — Jupyter カーネル（ノートブック開発用）

初回セットアップの詳細な手順は [Getting Started ガイド](../guide/04_getting_started.md) を参照してください。

---

## コーディング規約

### Python バージョン

Python 3.11 を固定で使用します。`mise.toml` および `pyproject.toml` でバージョンを指定しています。

### Ruff（Linter / Formatter）

Ruff は高速な Python Linter および Formatter です。

```bash
# リントチェック
ruff check src/ tests/

# フォーマットチェック
ruff format --check src/ tests/

# フォーマット自動修正
ruff format src/ tests/
```

設定（`pyproject.toml`）：

| 設定項目 | 値 |
|---|---|
| `target-version` | `py311` |
| `line-length` | `100` |
| `lint.select` | `E`, `F`, `I`, `N`, `W` |

各ルールの概要：

- **E** / **W** — pycodestyle（エラー / 警告）
- **F** — pyflakes（未使用インポート等）
- **I** — isort（import 順序）
- **N** — pep8-naming（命名規則）

### Mypy（型チェッカー）

全関数に型アノテーションが必須です。

```bash
# 型チェック
mypy src/
```

主要な設定（`pyproject.toml`）：

| 設定項目 | 値 |
|---|---|
| `python_version` | `3.11` |
| `disallow_untyped_defs` | `true` |
| `warn_return_any` | `true` |
| `warn_unused_configs` | `true` |

### import パス

`pythonpath = [".", "src"]` が設定されています。以下のようにインポートします：

```python
from domain.types import Surface, BetType
from domain.models import Race, Entry
```

### その他の規約

- **SQLAlchemy Core のみ使用** — ORM は使用しない
- **テストは全て mock** — `unittest.mock` を使用し、DB に依存しない

これらの設計方針の詳細は [CLAUDE.md > Architecture](../../CLAUDE.md) を参照してください。

---

## コミットメッセージ

[Conventional Commits](https://www.conventionalcommits.org/)（日本語）に従います。

### プレフィックス一覧

| プレフィックス | 用途 |
|---|---|
| `feat` | 新機能・新規モジュール |
| `fix` | バグ修正 |
| `docs` | ドキュメント追加・修正 |
| `test` | テスト追加・修正 |
| `refactor` | リファクタリング（機能変更なし） |
| `chore` | ビルド・ツール・設定変更 |

### メッセージ形式

```
<type>: <日本語の説明> (<phase-id>)
```

- `type` — 上記プレフィックス
- `日本語の説明` — 変更内容を簡潔に記述
- `phase-id`（任意） — 設計書のフェーズ識別子（例: `A-1`, `C-5`, `G-2d`）

### 例

```
feat: EV補正モデル P/E分解で独立性破綻を解決 (C-5)
fix: Race複合PKの Generated Always AS カラムを修正 (A-3)
docs: 開発ガイドを作成 (reference/04)
test: BacktestValidationSuite 全テスト統合 (G-1)
refactor: domain models の computed properties を整理 (A-2)
```

---

## テスト

### 特徴

- **DB 不要** — 全テスト `unittest.mock` でモック化
- **即座に実行可能** — クローン直後から `pytest` が動作
- **43 テストファイル / 6,163 行**（2026年3月時点）

### 実行方法

```bash
# 全テスト実行
python -m pytest tests/ -v

# 単一テストファイル
python -m pytest tests/test_domain.py -v

# カバレッジ付き実行
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

### テストの書き方

- 標準ライブラリ `unittest.mock` を使用（pytest-mock 等の追加ライブラリは不要）
- 外部リソース（DB、API）へのアクセスは必ずモック化
- テストファイルは `tests/` 配下に配置し、`test_*.py` の命名に従う

---

## 関連ドキュメント

| ドキュメント | 説明 |
|---|---|
| [CLAUDE.md](../../CLAUDE.md) | プロジェクト概要・アーキテクチャ・開発環境の全体設定 |
| [設計書 design.md](../../docs/design.md) | システム設計書 v5.5（〜2,900行） |
| [設定リファレンス](03_configuration.md) | `config/settings.yaml` の設定項目解説 |
| [Getting Started](../guide/04_getting_started.md) | 初回セットアップの詳細ガイド |

---

> **前のドキュメント:** [設定リファレンス](03_configuration.md) | **ドキュメント一覧:** [README](../../README.md)
