# Phase 35: ETL 品質確認手順 (D-03)

**Document:** ETL Quality Verification
**Created:** 2026-05-19
**Status:** Verification procedure (PostgreSQL環境依存、手動実施)

## 概要

Phase 35 のコード変更完了後、PostgreSQL環境で `run_etl.py` を実行する際の
Claude 品質確認手順を定義する。全テストは mock 使用のため、実際の Parquet 品質は
手動検証が必要 (D-04)。

## 1. ETL 実行コマンド

```bash
python scripts/run_etl.py --mode full --start 20140101 --end 20251231
```

- 所要時間: ~10分
- PostgreSQL 接続必須 (`localhost:5432/everydb2`)
- `PGPASSWORD` 環境変数でパスワードを上書き可能

## 2. 検証対象と期待結果

### 2.1 entries.parquet — HaronTimeL3/L4 (ETL-01)

| 項目 | 期待値 |
|------|--------|
| 列名 | `harontimel3`, `harontimel4` |
| dtype | `float64` (varchar ではない) |
| センチネルNaN | "000" と "999" が NaN 化されている |
| 有効値範囲 | 33.0 ~ 45.0 秒程度 (÷10前は 330.0 ~ 450.0) |
| 禁止値 | 0.0, 999.0 (センチネル残存の兆候) |

**検証コマンド:**
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/raw/entries.parquet')
print('=== HaronTimeL3/L4 (entries) ===')
print(df[['harontimel3','harontimel4']].describe())
print()
print('NaN rate:')
print(df[['harontimel3','harontimel4']].isna().mean())
print()
print('Sentinel check (must be 0):')
print(f'  L3 == 0.0: {(df[\"harontimel3\"] == 0.0).sum()}')
print(f'  L3 == 999.0: {(df[\"harontimel3\"] == 999.0).sum()}')
print(f'  L4 == 0.0: {(df[\"harontimel4\"] == 0.0).sum()}')
print(f'  L4 == 999.0: {(df[\"harontimel4\"] == 999.0).sum()}')
print(f'dtypes: L3={df[\"harontimel3\"].dtype}, L4={df[\"harontimel4\"].dtype}')
"
```

### 2.2 entries.parquet — Jyuni2c/3c (ETL-03)

| 項目 | 期待値 |
|------|--------|
| 列名 | `jyuni2c`, `jyuni3c` |
| dtype | `float64` (sentinel_float 処理後) |
| センチネルNaN | "000"/"00" が NaN 化されている |
| 有効値範囲 | 1.0 ~ 18.0 (コーナー通過順位) |

**検証コマンド:**
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/raw/entries.parquet')
print('=== Jyuni2c/3c (entries) ===')
print(df[['jyuni2c','jyuni3c']].describe())
print()
print('NaN rate:')
print(df[['jyuni2c','jyuni3c']].isna().mean())
print(f'dtypes: jyuni2c={df[\"jyuni2c\"].dtype}, jyuni3c={df[\"jyuni3c\"].dtype}')
"
```

### 2.3 races.parquet — LapTime1~25 (ETL-02)

| 項目 | 期待値 |
|------|--------|
| 列名 | `laptime1` ~ `laptime25` |
| dtype | `float64` (varchar ではない) |
| センチネルNaN | "000" が NaN 化されている |
| 有効値範囲 | 10.0 ~ 99.0 秒程度 (divisor=10 で除算済みの場合)、または 100.0 ~ 999.0 (÷10なしの場合) |
| 禁止値 | 0.0 (センチネル "000" の残存) |

**検証コマンド:**
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/raw/races.parquet')
lap_cols = [f'laptime{i}' for i in range(1, 26)]
existing = [c for c in lap_cols if c in df.columns]
missing = [c for c in lap_cols if c not in df.columns]
print(f'=== LapTime1~25 (races) ===')
print(f'Existing: {len(existing)} columns')
if missing:
    print(f'Missing: {missing}')
if existing:
    print(df[existing].describe())
    print()
    print('NaN rate (sample):')
    print(df[existing[:5]].isna().mean())
    print()
    print('Sentinel check (LapTime1 == 0.0 must be 0):')
    print(f'  LapTime1 == 0.0: {(df[\"laptime1\"] == 0.0).sum()}')
    print(f'  dtype LapTime1: {df[\"laptime1\"].dtype}')
"
```

### 2.4 races.parquet — RA HaronTimeL3/L4 (Plan 35-01 追加対応)

| 項目 | 期待値 |
|------|--------|
| 列名 | `harontimel3`, `harontimel4` |
| dtype | `float64` |
| センチネルNaN | "000" と "999" が NaN 化されている |
| 有効値範囲 | 33.0 ~ 45.0 秒程度 |

**検証コマンド:**
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/raw/races.parquet')
print('=== RA HaronTimeL3/L4 (races) ===')
if 'harontimel3' in df.columns:
    print(df[['harontimel3','harontimel4']].describe())
    print()
    print('NaN rate:')
    print(df[['harontimel3','harontimel4']].isna().mean())
else:
    print('WARNING: harontimel3 column not found in races.parquet')
"
```

## 3. 品質確認チェックリスト

以下のチェックリストはETL実行後に1つずつ確認する:

- [ ] **HaronTimeL3 (entries)**: float64 型、0.0と999.0が存在しない (min > 0.0)
- [ ] **HaronTimeL4 (entries)**: float64 型、NaN率が合理的 (< 50%)
- [ ] **Jyuni2c/3c (entries)**: float64 型、NaN率が合理的
- [ ] **LapTime1~25 (races)**: float64 型、有効値範囲が合理的
- [ ] **RA HaronTimeL3/L4 (races)**: float64 型、センチネルNaN化済み
- [ ] **CI テスト通過**: `python -m pytest tests/ -v` が全て通過

## 4. 問題発見時の対応

### 4.1 センチネルがNaN化されていない場合

**症状:** describe() で min=0.0 または max=999.0 が表示される

**原因:** `_TABLE_TYPE_RULES` の `sentinel_float` sentinels リストに該当値が含まれていない

**対応:**
1. `src/db/etl.py` の該当 sentinel_float ルールの `sentinels` リストを確認
2. 欠落しているセンチネル値を追加 (例: "00" が含まれていない場合)
3. ETL を再実行 (`--mode full`)

### 4.2 dtype が object の場合

**症状:** `.dtype` が `object` と表示される

**原因:** `_apply_type_conversions` の該当ルールが適用されていない

**対応:**
1. `src/db/etl.py` の `_TABLE_TYPE_RULES` で該当列が正しく登録されているか確認
2. テーブル名のマッピングが正しいか確認 (entries/races)
3. ETL を再実行

### 4.3 NaN率が極端に高い場合

**症状:** NaN率が 90% を超える

**原因:** センチネル定義の誤り (有効値をセンチネルとして扱っている可能性)

**対応:**
1. EveryDB2 スキーマドキュメントで初期値とセンチネル値を再確認
   - `docs/everydb2/04-UMA_RACE.md` (SE table)
   - `docs/everydb2/03-RACE.md` (RA table)
2. センチネル値リストが実際のデータに合致しているか確認
3. 必要に応じてセンチネル値を修正し、ETLを再実行

## 5. 全テスト通過確認

ETL品質確認後、CIテストが全て通過することを確認:

```bash
python -m pytest tests/ -v
```

特に以下のテストファイルが重要:

- `tests/test_post_race_leakage.py` — 3層CI漏洩テスト (POST_RACE安全性)
- `tests/test_paper_trading_guards.py` — POST_RACE DROP確認
- `tests/test_etl_type_conversion.py` — sentinel rule テスト (Plan 35-01)

---

*Phase: 35-ETL Data Foundation*
*D-03: ETL品質確認手順*
*Created: 2026-05-19*
