# Generic ETL: EveryDB2 全テーブル Parquet 出力 + 差分更新

## 背景

EveryDB2 (JRA-VAN DataLab) の PostgreSQL 外部テーブル103個をすべて Parquet ファイルに出力する。現在の ETL (`src/db/etl.py`) は9テーブルのみ対応で、各テーブルごとにハードコードされたカラム選択・リネーム・型変換を持つ。103テーブルに同じ手法はスケールしないため、汎用 ETL エンジンに置き換える。

## 要件

1. PostgreSQL の103テーブル (n_ 53個 + s_ 50個) をすべて Parquet に出力
2. s_ テーブルを利用した差分更新（upsert/delete）に対応
3. カラムはそのまま出力（リネーム・型変換なし）。後から必要テーブルのみ設定追加
4. 既存の ParquetStore / DatabaseConnection インフラを活用
5. 既存の `etl.py` を汎用版に置き換え

## 制約と確認済み事実

- **カラム名はすべて小文字**: `trackcd`, `jyocd`, `datakubun` 等。アンダースコアなし
- **`datakubun` は小文字で確定**: PostgreSQL 外部テーブルの実際のカラム名
- **`x_` テーブルは存在しない**: 現在の etl.py が参照する `x_uma`, `x_kisyu_seiseki`, `x_chokyo_seiseki` は DB に存在しない（バグ）。正しくは `n_uma`, `n_kisyu_seiseki`, `n_chokyo_seiseki`
- **全カラム varchar**: EveryDB2 はすべてのカラムを `character varying` で定義

## 設計

### テーブル分類

| タイプ | 説明 | 日付フィルタ | 対象テーブル数 |
|--------|------|-------------|---------------|
| `raced` | レースに紐づくデータ | `year + monthday` | ~30 (race, uma_race, harai, odds系, hyosu系, jodds系) |
| `master` | マスターデータ | なし（フルダンプ） | ~20 (uma, kisyu, chokyo, seisan, banusi等) |
| `delta` | s_ テーブル差分 | なし（前回取得からの差分） | ~50 (s_race, s_uma_race等) |

**Why:** EveryDB2 は `n_` (normal/全件) と `s_` (scratch/差分) の2系統で配信される。`s_` テーブルの行数は `n_` のごく一部（例: n_race 70,733行 → s_race 72行）。

### ETL 出力の列ルール

汎用 ETL は **基本的にカラムをそのまま出力** するが、以下の例外を設ける:

| 列 | 対象テーブル | 処理 | 理由 |
|----|------------|------|------|
| `race_date` | `type: raced` の全テーブル | `year + monthday` から datetime64 計算して追加 | PyArrow 述語プッシュダウンに必須。これがないと日付フィルタがフルスキャンになる |

その他の変換（`race_id` 計算、カラムリネーム、オッズ型変換等）はすべて DataRepository 側に移行する。

**Why `race_date` only:** `race_date` は ParquetStore.read() の predicate pushdown で使われ、これがないと DataRepository の全 load_* メソッドがフルスキャンになる。`race_id` は join key だが、DataRepository が読み込み時に計算可能。

### 差分マージロジック

差分データ (`s_` テーブル) を既存 Parquet ファイルにマージする:

1. 既存 Parquet ファイルを `existing_df` として読む
2. `s_` テーブルを `delta_df` として読む
3. `delta_df` を `deletes` (`datakubun == '0'`) と `upserts` (`datakubun != '0'`) に分割
4. `existing_df` から `deletes` の PK に一致する行を除外
5. 除外後の `existing_df` から `upserts` の PK に一致する行を除外（古い行を捨てる）
6. 手順5の結果と `upserts` を `pd.concat()` で結合
7. `datakubun` 列は出力から除外（消費者は不要）
8. `race_date` を再計算（type=raced の場合）
9. 結果を Parquet に書き戻し（ParquetStore のアトミック書き込みで安全）

**エッジケース:** 既存 Parquet が存在しない場合（フルロード前に差分更新が実行された場合）はエラーとする。`--mode delta` はフルロード後にのみ実行可能。

### 設定ファイル (`config/etl_tables.yaml`)

```yaml
# テーブル共通フィールド:
#   db_table:     PostgreSQL テーブル名（必須）
#   parquet_key:  出力先ファイル名（必須）
#   category:     ディレクトリ分類 raw/odds 等（必須）
#   type:         raced/master/delta（必須）
#   pk:           主キーカラムリスト（必須、差分マージ用）
#   partition_cols: パーティション列（任意、時系列オッズ等）

tables:
  # ============================================================
  # n_ 系: レース紐付けデータ (type: raced)
  # ============================================================
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

  - db_table: n_harai
    parquet_key: payouts
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: n_odds_tanpuku
    parquet_key: odds_tanpuku
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban]

  - db_table: n_odds_wide
    parquet_key: odds_wide
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2]

  - db_table: n_jodds_tanpuku
    parquet_key: jodds_tanpuku
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban, happyotime]
    partition_cols: [year, month]

  - db_table: n_odds_tanpukuwaku_head
    parquet_key: odds_tanpukuwaku_head
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: n_odds_waku
    parquet_key: odds_waku
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, wakuban1, wakuban2]

  - db_table: n_odds_umaren_head
    parquet_key: odds_umaren_head
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: n_odds_umaren
    parquet_key: odds_umaren
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2]

  - db_table: n_odds_umatan_head
    parquet_key: odds_umatan_head
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: n_odds_umatan
    parquet_key: odds_umatan
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2]

  - db_table: n_odds_sanren_head
    parquet_key: odds_sanren_head
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: n_odds_sanren
    parquet_key: odds_sanren
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2, umaban3]

  - db_table: n_odds_sanrentan_head
    parquet_key: odds_sanrentan_head
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: n_odds_sanrentan
    parquet_key: odds_sanrentan
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2, umaban3]

  - db_table: n_odds_wide_head
    parquet_key: odds_wide_head
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: n_hyosu
    parquet_key: hyosu
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: n_hyosu_tanpuku
    parquet_key: hyosu_tanpuku
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban]

  - db_table: n_hyosu_waku
    parquet_key: hyosu_waku
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, wakuban1, wakuban2]

  - db_table: n_hyosu_umarenwide
    parquet_key: hyosu_umarenwide
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2]

  - db_table: n_hyosu_umatan
    parquet_key: hyosu_umatan
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2]

  - db_table: n_hyosu_sanren
    parquet_key: hyosu_sanren
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2, umaban3]

  - db_table: n_hyosu_sanrentan
    parquet_key: hyosu_sanrentan
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2, umaban3]

  - db_table: n_hyosu2
    parquet_key: hyosu2
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: n_jodds_tanpukuwaku_head
    parquet_key: jodds_tanpukuwaku_head
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: n_jodds_umaren_head
    parquet_key: jodds_umaren_head
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: n_jodds_umaren
    parquet_key: jodds_umaren
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2, happyotime]

  - db_table: n_jodds_waku
    parquet_key: jodds_waku
    category: odds
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, wakuban1, wakuban2, happyotime]

  - db_table: n_toku_race
    parquet_key: toku_race
    category: raw
    type: raced
    pk: [year, tokunum]

  - db_table: n_toku
    parquet_key: toku
    category: raw
    type: raced
    pk: [year, tokunum, kettonum]

  - db_table: n_mining
    parquet_key: mining
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban]

  - db_table: n_taisengata_mining
    parquet_key: taisengata_mining
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2]

  - db_table: n_jyusyosiki_head
    parquet_key: jyusyosiki_head
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: n_jyusyosiki
    parquet_key: jyusyosiki
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, jyuni]

  # ============================================================
  # n_ 系: マスターデータ (type: master)
  # ============================================================
  - db_table: n_uma
    parquet_key: horses
    category: raw
    type: master
    pk: [kettonum]

  - db_table: n_kisyu
    parquet_key: kisyu
    category: raw
    type: master
    pk: [kisyucode]

  - db_table: n_chokyo
    parquet_key: chokyo
    category: raw
    type: master
    pk: [chokyosicode]

  - db_table: n_kisyu_seiseki
    parquet_key: kisyu_seiseki
    category: raw
    type: master
    pk: [kisyucode, setyear]

  - db_table: n_chokyo_seiseki
    parquet_key: chokyo_seiseki
    category: raw
    type: master
    pk: [chokyosicode, setyear]

  - db_table: n_seisan
    parquet_key: seisan
    category: raw
    type: master
    pk: [seisancode]

  - db_table: n_banusi
    parquet_key: banusi
    category: raw
    type: master
    pk: [banusicode]

  - db_table: n_hansyoku
    parquet_key: hansyoku
    category: raw
    type: master
    pk: [kettonum]

  - db_table: n_sanku
    parquet_key: sanku
    category: raw
    type: master
    pk: [kettonum]

  - db_table: n_keito
    parquet_key: keito
    category: raw
    type: master
    pk: [keitoucode]

  - db_table: n_course
    parquet_key: course
    category: raw
    type: master
    pk: [coursecd]

  - db_table: n_bameiorigin
    parquet_key: bameiorigin
    category: raw
    type: master
    pk: [kettonum]

  - db_table: n_record
    parquet_key: record
    category: raw
    type: master
    pk: [jyocd, trackcd, kyori]

  - db_table: n_hanro
    parquet_key: hanro
    category: raw
    type: master
    pk: [jyocd, trackcd, kyori]

  - db_table: n_bataijyu
    parquet_key: bataijyu
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: n_sale
    parquet_key: sale
    category: raw
    type: master
    pk: [salecode]

  - db_table: n_schedule
    parquet_key: schedule
    category: raw
    type: master
    pk: [year, monthday, jyocd]

  - db_table: n_wood_chip
    parquet_key: wood_chip
    category: raw
    type: master
    pk: [kettonum]

  - db_table: n_jogaiba
    parquet_key: jogaiba
    category: raw
    type: raced
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban]

  # ============================================================
  # s_ 系: 差分データ (type: delta)
  # parquet_key は対応する n_ テーブルと同じ（マージ先）
  # ============================================================
  - db_table: s_race
    parquet_key: races
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_uma_race
    parquet_key: entries
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban, kettonum]

  - db_table: s_harai
    parquet_key: payouts
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_odds_tanpuku
    parquet_key: odds_tanpuku
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban]

  - db_table: s_odds_wide
    parquet_key: odds_wide
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2]

  - db_table: s_odds_tanpukuwaku_head
    parquet_key: odds_tanpukuwaku_head
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_odds_waku
    parquet_key: odds_waku
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, wakuban1, wakuban2]

  - db_table: s_odds_umaren_head
    parquet_key: odds_umaren_head
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_odds_umaren
    parquet_key: odds_umaren
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2]

  - db_table: s_odds_umatan_head
    parquet_key: odds_umatan_head
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_odds_umatan
    parquet_key: odds_umatan
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2]

  - db_table: s_odds_sanren_head
    parquet_key: odds_sanren_head
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_odds_sanren
    parquet_key: odds_sanren
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2, umaban3]

  - db_table: s_odds_sanrentan_head
    parquet_key: odds_sanrentan_head
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_odds_sanrentan
    parquet_key: odds_sanrentan
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2, umaban3]

  - db_table: s_odds_wide_head
    parquet_key: odds_wide_head
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_jodds_tanpuku
    parquet_key: jodds_tanpuku
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban, happyotime]

  - db_table: s_jodds_tanpukuwaku_head
    parquet_key: jodds_tanpukuwaku_head
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_jodds_umaren_head
    parquet_key: jodds_umaren_head
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_jodds_umaren
    parquet_key: jodds_umaren
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2, happyotime]

  - db_table: s_jodds_waku
    parquet_key: jodds_waku
    category: odds
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, wakuban1, wakuban2, happyotime]

  - db_table: s_hyosu
    parquet_key: hyosu
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_hyosu_tanpuku
    parquet_key: hyosu_tanpuku
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban]

  - db_table: s_hyosu_waku
    parquet_key: hyosu_waku
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, wakuban1, wakuban2]

  - db_table: s_hyosu_umarenwide
    parquet_key: hyosu_umarenwide
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2]

  - db_table: s_hyosu_umatan
    parquet_key: hyosu_umatan
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2]

  - db_table: s_hyosu_sanren
    parquet_key: hyosu_sanren
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2, umaban3]

  - db_table: s_hyosu_sanrentan
    parquet_key: hyosu_sanrentan
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2, umaban3]

  - db_table: s_hyosu2
    parquet_key: hyosu2
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_uma
    parquet_key: horses
    category: raw
    type: delta
    pk: [kettonum]

  - db_table: s_kisyu
    parquet_key: kisyu
    category: raw
    type: delta
    pk: [kisyucode]

  - db_table: s_chokyo
    parquet_key: chokyo
    category: raw
    type: delta
    pk: [chokyosicode]

  - db_table: s_kisyu_seiseki
    parquet_key: kisyu_seiseki
    category: raw
    type: delta
    pk: [kisyucode, setyear]

  - db_table: s_chokyo_seiseki
    parquet_key: chokyo_seiseki
    category: raw
    type: delta
    pk: [chokyosicode, setyear]

  - db_table: s_seisan
    parquet_key: seisan
    category: raw
    type: delta
    pk: [seisancode]

  - db_table: s_banusi
    parquet_key: banusi
    category: raw
    type: delta
    pk: [banusicode]

  - db_table: s_record
    parquet_key: record
    category: raw
    type: delta
    pk: [jyocd, trackcd, kyori]

  - db_table: s_toku_race
    parquet_key: toku_race
    category: raw
    type: delta
    pk: [year, tokunum]

  - db_table: s_toku
    parquet_key: toku
    category: raw
    type: delta
    pk: [year, tokunum, kettonum]

  - db_table: s_mining
    parquet_key: mining
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban]

  - db_table: s_taisengata_mining
    parquet_key: taisengata_mining
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2]

  - db_table: s_jyusyosiki_head
    parquet_key: jyusyosiki_head
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_jyusyosiki
    parquet_key: jyusyosiki
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, jyuni]

  - db_table: s_jogaiba
    parquet_key: jogaiba
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban]

  # s_ 独自テーブル（n_ に対応なし）
  - db_table: s_bataijyu
    parquet_key: bataijyu
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_tenko_baba
    parquet_key: tenko_baba
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_kisyu_change
    parquet_key: kisyu_change
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban]

  - db_table: s_course_change
    parquet_key: course_change
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_hassou_jikoku_change
    parquet_key: hassou_jikoku_change
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum]

  - db_table: s_torikesi_jyogai
    parquet_key: torikesi_jyogai
    category: raw
    type: delta
    pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban]
```

### ファイル構成

```
src/db/
├── etl.py              ← リプレイス: 汎用ETLエンジン（~300行）
├── parquet_store.py    ← そのまま
├── repository.py       ← 更新: EveryDB2 生カラム名対応 + race_id 計算追加
├── connection.py       ← そのまま
└── schema.py           ← そのまま

config/
├── settings.yaml       ← そのまま
└── etl_tables.yaml     ← 新規: 103テーブル定義

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
    "races": {"rows": 70733, "last_full": "2026-03-28T20:59:00", "last_delta": "2026-03-29T15:30:00"},
    "entries": {"rows": 820000, "last_full": "2026-03-28T20:59:00", "last_delta": "2026-03-29T15:30:00"}
  }
}
```

**Why:** 差分更新の実行履歴を記録し、二重適用や欠落を防止する。テーブルごとに個別トラッキングし、部分失敗時に再実行可能にする。

### etl.py の主要関数

```python
def load_table_config(path: str) -> list[dict]: ...
def run_full_load(store: ParquetStore, engine: Engine, config: list[dict], start: str, end: str, tables: list[str] | None = None) -> dict[str, int]: ...
def run_delta_update(store: ParquetStore, engine: Engine, config: list[dict]) -> dict[str, int]: ...
def _merge_delta(existing: pd.DataFrame, delta: pd.DataFrame, pk: list[str]) -> pd.DataFrame: ...
def _read_db_table(engine: Engine, table_name: str, start: str | None = None, end: str | None = None) -> pd.DataFrame: ...
def _add_race_date(df: pd.DataFrame) -> pd.DataFrame: ...
```

### 影響範囲

- `src/db/etl.py` — フルリプレイス（既存のカスタム変換ロジックを削除）
- `scripts/run_etl.py` — `--mode` 引数追加
- `config/etl_tables.yaml` — 新規作成（103テーブル定義）
- `src/db/repository.py` — **要更新**: EveryDB2 生カラム名（`trackcd`等）→ リネーム（`track_cd`等）+ `race_id` 計算を追加
- `src/db/parquet_store.py` — 変更なし
- MLパイプライン — 変更なし（DataRepository 経由で同じ列名でアクセス）

### DataRepository の更新内容

汎用 ETL が出力する Parquet ファイルは EveryDB2 の生カラム名（`trackcd`, `jyocd`等）を含む。DataRepository は読み込み時に以下の変換を行う:

1. **カラムリネーム**: `trackcd` → `track_cd`, `jyocd` → `jyo_cd` 等の snake_case 変換
2. **`race_id` 計算**: `year + monthday + jyocd + kaiji + nichiji + racenum` → 16桁文字列
3. **型変換**: varchar → int/float（オッズ、着順等）
4. **フィルタ**: 障害除外（`trackcd NOT IN (51-59)`）等

**Why:** ETL 層は「データのダンプ」に専念し、ドメイン変換はデータアクセス層で実行する責務分離。カラムリネームは `_rename_columns(df, mapping)` ヘルパーで一括処理。

### 大規模テーブルの考慮事項

- **`n_jodds_tanpuku`** (~83M行): `partition_cols: [year, month]` で Hive パーティション出力。年ごとにチャンク読み込みしてメモリ使用量を制御
- **`n_uma`** (212K行, 227列): 全カラムそのまま出力で問題なし
- **`n_uma_race`** (~800K行): 全カラムそのまま出力で問題なし

## PostgreSQL テーブル一覧 (103テーブル)

### n_ 系 (53テーブル)

n_bameiorigin, n_banusi, n_chokyo, n_chokyo_seiseki, n_course, n_hanro, n_hansyoku, n_harai, n_hyosu, n_hyosu2, n_hyosu_sanren, n_hyosu_sanrentan, n_hyosu_tanpuku, n_hyosu_umarenwide, n_hyosu_umatan, n_hyosu_waku, n_jodds_tanpuku, n_jodds_tanpukuwaku_head, n_jodds_umaren, n_jodds_umaren_head, n_jodds_waku, n_jogaiba, n_jyusyosiki, n_jyusyosiki_head, n_keito, n_kisyu, n_kisyu_seiseki, n_mining, n_odds_sanren, n_odds_sanren_head, n_odds_sanrentan, n_odds_sanrentan_head, n_odds_tanpuku, n_odds_tanpukuwaku_head, n_odds_umaren, n_odds_umaren_head, n_odds_umatan, n_odds_umatan_head, n_odds_waku, n_odds_wide, n_odds_wide_head, n_race, n_record, n_sale, n_sanku, n_schedule, n_seisan, n_taisengata_mining, n_toku, n_toku_race, n_uma, n_uma_race, n_wood_chip

### s_ 系 (50テーブル)

s_banusi, s_bataijyu, s_chokyo, s_chokyo_seiseki, s_course_change, s_harai, s_hassou_jikoku_change, s_hyosu, s_hyosu2, s_hyosu_sanren, s_hyosu_sanrentan, s_hyosu_tanpuku, s_hyosu_umarenwide, s_hyosu_umatan, s_hyosu_waku, s_jodds_tanpuku, s_jodds_tanpukuwaku_head, s_jodds_umaren, s_jodds_umaren_head, s_jodds_waku, s_jogaiba, s_jyusyosiki, s_jyusyosiki_head, s_kisyu, s_kisyu_change, s_kisyu_seiseki, s_mining, s_odds_sanren, s_odds_sanren_head, s_odds_sanrentan, s_odds_sanrentan_head, s_odds_tanpuku, s_odds_tanpukuwaku_head, s_odds_umaren, s_odds_umaren_head, s_odds_umatan, s_odds_umatan_head, s_odds_waku, s_odds_wide, s_odds_wide_head, s_race, s_record, s_seisan, s_taisengata_mining, s_tenko_baba, s_toku, s_toku_race, s_torikesi_jyogai, s_uma, s_uma_race
