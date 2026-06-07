# 血統Parquetファイルの復元手順

## 現状

血統セッション（2026-06-03〜06-04）で生成した parquet を一時退避中。
また、DamPedigreeFeatures のバグ修正コミットが main にマージされていない。

### 退避中のparquet

- `data/raw/sanku.parquet` → `tmp/pedigree_parquet_backup/sanku.parquet`
- `data/raw/hansyoku.parquet` → `tmp/pedigree_parquet_backup/hansyoku.parquet`
- `data/raw/keito.parquet` → `tmp/pedigree_parquet_backup/keito.parquet`

退避状態では `DamPedigreeFeatures`（dam_wr等）と `BloodlineFeatures`（keito_cd）が
空DFを返し、全馬 "unknown" / NaN 扱いになる。

### 未マージのコミット

- `a616720` — fix: DamPedigreeFeatures の2バグ修正 (merge後index不一致 + kettonum dtype不一致)

このコミットは `a23cbb3` 系のブランチにあり、main に取り込まれていない。

## 復元手順

### Step 1: parquet ファイルを戻す

```bash
mv tmp/pedigree_parquet_backup/sanku.parquet data/raw/
mv tmp/pedigree_parquet_backup/hansyoku.parquet data/raw/
mv tmp/pedigree_parquet_backup/keito.parquet data/raw/
```

### Step 2: dam_pedigree バグ修正を main に取り込む

```bash
git cherry-pick a616720
```

修正内容:
1. `dam_pedigree_features.py` — merge() 後に valid boolean mask を再計算（indexリセット対応）
2. `dam_pedigree_features.py` — kettonum の Categorical/Object dtype を str に統一してから merge_asof

### Step 3: 確認

```bash
python -c "
from db.parquet_store import ParquetStore
from features.dam_pedigree_features import DamPedigreeFeatures
store = ParquetStore()
print('sanku exists:', store.exists('raw', 'sanku'))
dpf = DamPedigreeFeatures(store)
print('DamPedigreeFeatures loaded OK')
"
```

## ETLからの再生成（ファイル紛失時）

```bash
$env:PGPASSWORD = 'aa8940aa'
python scripts/run_etl.py --mode full --start 20200101 --end 20251231 --tables sanku hansyoku keito
```

DB内の行数（確認用）:
- `n_sanku`: 82,875 rows
- `n_hansyoku`: 23,021 rows
- `n_keito`: 92 rows

## 2025 BT ROI への影響（2026-06-04 計測）

| 条件 | n | ROI | 利益 |
|------|---|-----|------|
| 血統なし (EV>=1.03, odds>=3) | 1,071 | 95.4% | -4,920 |
| 血統あり (EV>=1.03, odds>=3) | 708 | 94.7% | -3,740 |

差は 0.7pt と小さく、血統修正自体はROI悪化の主因ではない。
