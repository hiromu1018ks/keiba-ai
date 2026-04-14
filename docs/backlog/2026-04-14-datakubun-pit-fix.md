# DataKubun フィルタ修正 — PIT (Predictive Information Theory) 対応

**日付:** 2026-04-14
**ステータス:** 実装済（コミット前）

## 問題

Paper Trading (PT) で `track_condition_code` (TCC) が常に 0 となり、
Backtest (BT) との特徴量乖離が発生。PT ROI 74.4% vs BT ROI 136.6%。

## 原因

EveryDB2 `s_race` / `s_uma_race` テーブルは同一レースに複数の DataKubun レコードを持つ。
旧コードは全レコードを取得していたため、DataKubun=1（木曜出走馬名表、馬場状態なし）
が混入し、TCC=0 となっていた。

## EveryDB2 DataKubun 定義 (03-RACE.md より)

| DataKubun | 内容 | TCC有無 | リークリスク |
|-----------|------|---------|-------------|
| 1 | 出走馬名表（木曜） | **なし** | なし |
| **2** | 出馬表（金・土曜） | **あり** | **なし** ← 採用 |
| 3 | 速報成績（3着まで確定） | あり | 着順リーク |
| 4 | 速報成績（5着まで確定） | あり | 着順リーク |
| 5 | 速報成績（全馬着順確定） | あり | 着順+タイムリーク |
| 6 | 速報成績（全馬着順+コーナ通過順） | あり | コーナー通過順位リーク |
| 7 | 成績（月曜／確定） | あり | 全確定情報リーク |

## 修正内容

**ファイル:** `src/db/everydb2_queries.py`

### `get_races()` (line ~172)
```python
# Before:
sql = "SELECT * FROM s_race WHERE year || monthday = %s"

# After:
sql = "SELECT * FROM s_race WHERE year || monthday = %s AND DataKubun = '2'"
```

### `get_entries()` (line ~193)
```python
# Before:
sql = "SELECT * FROM s_uma_race WHERE year || monthday = %s"

# After:
sql = "SELECT * FROM s_uma_race WHERE year || monthday = %s AND DataKubun = '2'"
```

## 検証結果 (4/11, 4/12)

| テーブル | 日付 | 行数 | SibaBabaCD 分布 |
|----------|------|------|----------------|
| s_race (DK=2) | 4/11 | 36 | {1: 12, 2: 24} |
| s_uma_race (DK=2) | 4/11 | 526 | — |
| s_race (DK=2) | 4/12 | 36 | {1: 36} |

## 設計原則

- **DataKubun=2 のみ使用**: 出馬表は発走前に発表され、馬場状態を含む
- **DataKubun>=3 除外**: 速報/確定成績に着順・タイム等の結果情報を含むため PIT 違反
- **n_XXX フォールバック**: s_ テーブルが空の場合のみ蓄積系テーブルを使用
  （フォールバック先には DataKubun 列がないためフィルタ不可だが、
   s_ テーブルが存在する限り DK=2 のみを使用）

## 残タスク

- [ ] コミット
- [ ] Paper Trading 再実行で TCC が正しく流れることをエンドツーエンド検証
- [ ] PT vs BT の ROI 乖離が縮小することを確認
