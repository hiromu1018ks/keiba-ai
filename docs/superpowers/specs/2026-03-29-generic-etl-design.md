# Generic ETL: EveryDB2 全テーブル Parquet 出力 + 差分更新

## 背景

EveryDB2 (JRA-VAN DataLab) の PostgreSQL 外部テーブル103個をすべて Parquet ファイルに出力する。現在の ETL (`src/db/etl.py`) は9テーブルのみ対応で、各テーブルごとにハードコードされたカラム選択・リネーム・型変換を持つ。103テーブルに同じ手法はスケールしないため、汎用 ETL エンジンに置き換える。

## 要件

1. PostgreSQL の103テーブル (n_ 53個 + s_ 50個) をすべて Parquet に出力
2. s_ テーブルを利用した差分更新（upsert/delete）に対応
3. カラムはそのまま出力（リネーム・型変換なし）。後から必要テーブルのみ設定追加
4. 既存の ParquetStore / DatabaseConnection インフラを活用
5. 既存の `etl.py` を汎用版に置き換え

## 設計

### テーブル分類

| タイプ | 説明 | 日付フィルタ | 対象テーブル数 |
|--------|------|-------------|---------------|
| `raced` | レースに紐づくデータ | `year + monthday` | ~30 (race, uma_race, harai, odds系, hyosu系, jodds系) |
| `master` | マスターデータ | なし（フルダンプ） | ~20 (uma, kisyu, chokyo, seisan, banusi等) |
| `delta` | s_ テーブル差分 | なし（前回取得からの差分） | ~50 (s_race, s_uma_race等) |

**Why:** EveryDB2 は `n_` (normal/全件) と `s_` (scratch/差分) の2系統で配信される。`s_` テーブルの行数は `n_` のごく一部（例: n_race 70,733行 → s_race 72行）。

### 差分マージロジック

差分データ (`s_` テーブル) を既存 Parquet ファイルにマージする:

- `datakubun = '0'` → 該当 PK の行を削除
- `datakubun ≠ '0'` → 該当 PK の行を更新（なければ挿入）

マージ後は常に最新状態の1ファイルを維持する。利用者側は差分を意識しなくてよい。

**How to apply:** `pandas.DataFrame.merge()` で既存データと差分を結合。PK で `indicator=True` を使い、削除対象を除外した後、upsert 対象で上書き。

### 設定ファイル (`config/etl_tables.yaml`)

```yaml
tables:
  # === n_ 系: レース紐付けデータ ===
  - db_table: n_race
    parquet_key: races
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: n_uma_race
    parquet_key: entries
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban, kettonum]

  # ... (全 n_ テーブル)

  # === n_ 系: マスターデータ ===
  - db_table: n_uma
    parquet_key: horses
    category: raw
    type: master
    pk: [kettonum]

  # ... (全マスタテーブル)

  # === s_ 系: 差分データ ===
  - db_table: s_race
    parquet_key: races        # ← マージ先は同じ parquet_key
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  # ... (全 s_ テーブル)
```

各エントリのフィールド:

| フィールド | 必須 | 説明 |
|-----------|------|------|
| `db_table` | Yes | PostgreSQL テーブル名 |
| `parquet_key` | Yes | 出力先 Parquet ファイル名（拡張子なし） |
| `category` | Yes | `raw` / `odds` 等（ディレクトリ分類） |
| `type` | Yes | `raced` / `master` / `delta` |
| `pk` | Yes | 主キーカラムのリスト（差分マージ用） |
| `partition_cols` | No | パーティションカラム（時系列オッズ等） |

### ファイル構成

```
src/db/
├── etl.py              ← リプレイス: 汎用ETLエンジン
├── parquet_store.py    ← そのまま
├── repository.py       ← そのまま
├── connection.py       ← そのまま
└── schema.py           ← そのまま

config/
├── settings.yaml       ← そのまま
└── etl_tables.yaml     ← 新規

scripts/
└── run_etl.py          ← 更新: --mode full|delta 対応

data/
├── raw/*.parquet       ← 全テーブル
├── odds/*.parquet      ← オッズ系
└── etl_state.json      ← 差分管理状態
```

### CLI インターフェース

```bash
# 初回フルロード（全 n_ テーブル）
python scripts/run_etl.py --mode full --start 20140101 --end 20231231

# 差分更新（全 s_ テーブル → 既存 Parquet にマージ）
python scripts/run_etl.py --mode delta

# 特定テーブルのみフル再ロード
python scripts/run_etl.py --mode full --tables races entries --start 20140101 --end 20231231
```

### ETL状態ファイル (`data/etl_state.json`)

```json
{
  "last_delta_applied": "2026-03-29T15:30:00",
  "tables": {
    "races": {"rows": 70733, "last_full": "2026-03-28T20:59:00"},
    "entries": {"rows": 820000, "last_full": "2026-03-28T20:59:00"}
  }
}
```

**Why:** 差分更新の実行履歴を記録し、二重適用や欠落を防止する。

### etl.py の主要関数

```python
def load_table_config(path: str) -> list[dict]: ...
def run_full_load(store: ParquetStore, engine: Engine, config: list[dict], start: str, end: str, tables: list[str] | None = None) -> dict[str, int]: ...
def run_delta_update(store: ParquetStore, engine: Engine, config: list[dict]) -> dict[str, int]: ...
def _merge_delta(existing: pd.DataFrame, delta: pd.DataFrame, pk: list[str]) -> pd.DataFrame: ...
def _read_db_table(engine: Engine, table_name: str, start: str | None = None, end: str | None = None) -> pd.DataFrame: ...
```

### 影響範囲

- `src/db/etl.py` — フルリプレイス（既存のカスタム変換ロジックを削除）
- `scripts/run_etl.py` — `--mode` 引数追加
- `config/etl_tables.yaml` — 新規作成（103テーブル定義）
- `src/db/repository.py` — 変更なし（同じ Parquet ファイルを読む）
- `src/db/parquet_store.py` — 変更なし
- MLパイプライン — 変更なし（DataRepository 経由で同じファイルを読む）

### 既存カスタム変換の移行先

既存 `etl.py` のカスタム変換（race_id 計算、オッズ変換等）は、必要になった時に `DataRepository` 側に移行する:

- `_compute_race_id()` → `DataRepository.load_races()` 内で計算
- `_to_odds()` → `DataRepository` の読み込み時変換
- `_compute_race_date()` → `DataRepository` の読み込み時変換

**Why:** ETL 層は「データのダンプ」に専念し、ドメイン変換はデータアクセス層で実行する責務分離。

## PostgreSQL テーブル一覧 (103テーブル)

### n_ 系 (53テーブル)

n_bameiorigin, n_banusi, n_chokyo, n_chokyo_seiseki, n_course, n_hanro, n_hansyoku, n_harai, n_hyosu, n_hyosu2, n_hyosu_sanren, n_hyosu_sanrentan, n_hyosu_tanpuku, n_hyosu_umarenwide, n_hyosu_umatan, n_hyosu_waku, n_jodds_tanpuku, n_jodds_tanpukuwaku_head, n_jodds_umaren, n_jodds_umaren_head, n_jodds_waku, n_jogaiba, n_jyusyosiki, n_jyusyosiki_head, n_keito, n_kisyu, n_kisyu_seiseki, n_mining, n_odds_sanren, n_odds_sanren_head, n_odds_sanrentan, n_odds_sanrentan_head, n_odds_tanpuku, n_odds_tanpukuwaku_head, n_odds_umaren, n_odds_umaren_head, n_odds_umatan, n_odds_umatan_head, n_odds_waku, n_odds_wide, n_odds_wide_head, n_race, n_record, n_sale, n_sanku, n_schedule, n_seisan, n_taisengata_mining, n_toku, n_toku_race, n_uma, n_uma_race, n_wood_chip

### s_ 系 (50テーブル)

s_banusi, s_bataijyu, s_chokyo, s_chokyo_seiseki, s_course_change, s_harai, s_hassou_jikoku_change, s_hyosu, s_hyosu2, s_hyosu_sanren, s_hyosu_sanrentan, s_hyosu_tanpuku, s_hyosu_umarenwide, s_hyosu_umatan, s_hyosu_waku, s_jodds_tanpuku, s_jodds_tanpukuwaku_head, s_jodds_umaren, s_jodds_umaren_head, s_jodds_waku, s_jogaiba, s_jyusyosiki, s_jyusyosiki_head, s_kisyu, s_kisyu_change, s_kisyu_seiseki, s_mining, s_odds_sanren, s_odds_sanren_head, s_odds_sanrentan, s_odds_sanrentan_head, s_odds_tanpuku, s_odds_tanpukuwaku_head, s_odds_umaren, s_odds_umaren_head, s_odds_umatan, s_odds_umatan_head, s_odds_waku, s_odds_wide, s_odds_wide_head, s_race, s_record, s_seisan, s_taisengata_mining, s_tenko_baba, s_toku, s_toku_race, s_torikesi_jyogai, s_uma, s_uma_race
