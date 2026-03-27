# セットアップガイド

このシステムを実際に動かすための手順を説明します。各ステップのコマンドはそのままコピー＆ペーストで実行できます。

## 必要なもの

| ソフトウェア | バージョン | 用途 |
|-------------|-----------|------|
| Python | 3.11 | 実行環境 |
| PostgreSQL | 14+ | データベース |
| EveryDB2 | 最新版 | 競馬データソース |
| mise | 最新版 | Pythonバージョン管理 |

> **前提**: OSは Windows / macOS / Linux のいずれか。Gitがインストール済みであること。

## インストール手順

### 1. リポジトリのクローン

```bash
git clone <リポジトリURL>
cd keiba-ai
```

### 2. miseのインストール

**Windows (PowerShell):**

```powershell
irm https://mise.run | iex
```

**macOS:**

```bash
brew install mise
```

**Linux:**

```bash
curl https://mise.run | sh
```

インストール後、シェルを再起動してください（`mise` コマンドが使えることを確認）。

### 3. Pythonのセットアップ

```bash
# Python 3.11のインストール（mise.tomlで固定）
mise install

# Pythonをアクティベート
mise activate

# バージョン確認
python --version  # Python 3.11.x が表示されればOK
```

### 4. 依存パッケージのインストール

```bash
pip install -e ".[dev]"
```

これで本体パッケージ（pandas, lightgbm, sqlalchemy等）と開発ツール（ruff, mypy, ipykernel）の両方がインストールされます。

### 5. PostgreSQLのセットアップ

PostgreSQLをインストール後、データベースを作成します。

```bash
# everydb2データベースを作成
createdb everydb2
```

接続テスト:

```bash
psql -d everydb2 -c "SELECT 1;"
```

`?column? | 1` が返れば接続成功です。

### 6. EveryDB2データのインポート

EveryDB2のインポート手順はツールの公式ドキュメントに従ってください。インポートが完了すると、以下の外部テーブルが参照可能になります。

- `n_race` — レース情報
- `n_uma_race` — 馬のレース結果
- `n_uma` — 馬の基本情報
- `n_harai` — 払戻情報
- `n_odds_tanpuku` — 単勝・複勝オッズ
- `n_odds_wide` — ワイドオッズ

確認:

```bash
psql -d everydb2 -c "SELECT count(*) FROM n_race;"
```

## 設定

### config/settings.yaml

最小限の設定は `config/settings.yaml` の `database` セクションのみです。

```yaml
database:
  host: "localhost"
  port: 5432
  dbname: "everydb2"
  user: "postgres"
  password: ""  # 環境変数 PGPASSWORD で上書き
```

デフォルトでは `localhost:5432` の `postgres` ユーザーで接続します。環境に合わせて変更してください。

### 環境変数 PGPASSWORD

パスワードを設定ファイルに直接書きたくない場合は、環境変数で指定します。

**Windows (PowerShell):**

```powershell
$env:PGPASSWORD = "あなたのパスワード"
```

**macOS / Linux:**

```bash
export PGPASSWORD="あなたのパスワード"
```

永続的に設定する場合は、シェルの設定ファイル（`.bashrc`, `.zshrc` 等）に追記してください。

## 最初の予測

システムが正しく動作するかテストします。

```bash
# 1. テスト実行（DB不要、全テストmock使用）
python -m pytest tests/ -v

# 2. リントチェック
ruff check src/ tests/

# 3. 型チェック
mypy src/
```

全てパスすれば、セットアップは完了です。

## よくあるエラーと対処法

### PostgreSQL接続エラー

| エラーメッセージ | 原因 | 対処法 |
|-----------------|------|--------|
| `connection refused` | PostgreSQLが起動していない | `pg_ctl start` またはサービスを再起動 |
| `database "everydb2" does not exist` | データベース未作成 | `createdb everydb2` を実行 |
| `password authentication failed` | パスワードが間違っている | `PGPASSWORD` 環境変数を確認、または `config/settings.yaml` の `password` を確認 |
| `FATAL: role "postgres" does not exist` | ユーザー名が違う | `config/settings.yaml` の `user` をPostgreSQLのユーザー名に変更 |

### Python/mise環境エラー

| エラーメッセージ | 原因 | 対処法 |
|-----------------|------|--------|
| `python: command not found` | miseがアクティベートされていない | `mise activate` を実行。シェルを再起動しても解決しない場合、miseのPATH設定を確認 |
| `ModuleNotFoundError: No module named 'pandas'` | 依存パッケージ未インストール | `pip install -e ".[dev]"` を再実行 |
| `Python 3.11 not found` | miseがPythonをダウンロードできていない | `mise install` を再実行。ネットワーク接続を確認 |
| `ERROR: Could not build wheels for psycopg2` | Cコンパイラ未インストール | `psycopg2-binary` がインストールされるはず。それでも失敗する場合、Cコンパイラ（Windows: Build Tools, Mac: Xcode CLI Tools）をインストール |

### EveryDB2データエラー

| エラーメッセージ | 原因 | 対処法 |
|-----------------|------|--------|
| `relation "n_race" does not exist` | EveryDB2のテーブルが未インポート | EveryDB2のインポート手順に従ってデータをインポート |
| `permission denied for table n_race` | DBユーザーに読取権限がない | `GRANT SELECT ON ALL TABLES IN SCHEMA public TO postgres;` を実行 |
| テーブルはあるがデータが0件 | インポートが不完全 | EveryDB2を再インポート。JRA-VAN DataLabの契約状態を確認 |

### importエラー（pythonpath）

| エラーメッセージ | 原因 | 対処法 |
|-----------------|------|--------|
| `ModuleNotFoundError: No module named 'domain'` | pythonpathが通っていない | `pip install -e .` でパッケージを編集可能モードでインストール。または `PYTHONPATH=src:.` を設定 |
| `ImportError: cannot import name 'Race' from 'domain.models'` | 古いキャッシュ | `pip install -e ".[dev]"` を再実行し、`.pyc` ファイルを削除 |

## FAQ

**Q: miseを使わずに直接Python 3.11をインストールしても動きますか？**

A: はい、動作します。ただしPython 3.11である必要があります。miseはバージョン管理を簡単にするための推奨ツールです。pyenvなど他のツールでも構いません。

**Q: PostgreSQLをDockerで動かすことはできますか？**

A: はい、可能です。以下のコマンドで起動できます。

```bash
docker run -d --name keiba-pg \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=yourpassword \
  -p 5432:5432 \
  postgres:16
```

その後 `PGPASSWORD=yourpassword createdb -h localhost everydb2` でデータベースを作成してください。

**Q: EveryDB2以外のデータソースは使えますか？**

A: 現在はEveryDB2（およびJRA-VAN DataLab）に依存しています。`src/db/schema.py` で定義されている外部テーブル（`n_race`, `n_uma_race`等）と互換性のあるスキーマであれば、他のデータソースでも動作します。

**Q: テストはDBなしで動くと聞きましたが、本当ですか？**

A: はい、全てのテストは `unittest.mock` を使用しているため、PostgreSQLがなくても実行できます。`python -m pytest tests/ -v` でDB接続なしに全テストが走ります。

---

> **次のドキュメント:** [データの流れ](../concepts/01_data_pipeline.md) | **前のドキュメント:** [システム概要](03_system_overview.md) | **ドキュメント一覧:** [README](../../README.md)
