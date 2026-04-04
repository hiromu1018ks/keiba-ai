# Paper Trading 予測障害修正

## 概要

ペーパートレード (`run_paper_trading.py`) の predict / dry-run モードで、オッズ時系列特徴量が全馬 NaN になる等の障害を修正する。修正により Stage2 return model の予測値が均一になる問題が解消され、正常な EV 計算が可能になる。

## 障害一覧

| # | 障害 | 影響 | 優先度 |
|---|------|------|--------|
| 1 | odds_ts_df=None で build_all() 呼び出し | オッズ動態特徴量が全馬 NaN | Critical |
| 2 | load_odds_time_series_range() が2025-2026で空を返す | time_series のみ使用、jodds_tanpuku へのフォールバックなし | Critical |
| 3 | jodds_tanpuku の odds10 変換 (検証済み: 正常動作) | 異常値ガード追加のみ | Low |
| 4 | bamei が Shift-JIS バイト列で文字化け | 表示のみの問題 | Low |

## Fix 1: odds_ts_df を build_all() に渡す

**ファイル**: `scripts/run_paper_trading.py`
**対象**: `_run_predict()` (line 229), `_run_dry_run()` (line 519)

### 現状

```python
feat_df = feat_engine.build_all(race_df, entry_df, odds_df)
```

`TrainingPipelineV5` では `odds_ts_df` と `store` を渡しているが、ペーパートレードでは渡していない。その結果:

- `compute_odds_dynamics(df, odds_ts_df=None)` → 全カラム NaN
- Stage2 return model の最重要特徴量 top 3 (`odds_drop_rate_60_10`, `odds_drop_rate_30_10`, `odds_velocity`) が NaN
- `e_return_place_pred` が全馬 ~11.5 で均一

### 変更

両関数で `load_odds_time_series_range()` を呼び出し、結果を `build_all()` に渡す。

**`_run_predict()`** — `ymd` (line 213) を `all_start`/`all_end` として使用:

```python
from db.readers import load_odds_time_series_range

# line 219 (odds_df 読み込みの直後) に追加
odds_ts_df = load_odds_time_series_range(store, ymd, ymd)

# line 229 の build_all 呼び出しを変更
feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df)
```

**`_run_dry_run()`** — 既に `all_start`/`all_end` が定義済み (line 505-506):

```python
# line 511 (odds_df 読み込みの直後) に追加
odds_ts_df = load_odds_time_series_range(store, all_start, all_end)

# line 519 の build_all 呼び出しを変更
feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df)
```

## Fix 2: load_odds_time_series_range() の空結果フォールバック

**ファイル**: `src/db/readers.py`
**対象**: `load_odds_time_series_range()` (line 105-122), `load_odds_time_series()` (line 125-133)

### 現状

```python
subpath = "time_series" if store.exists("odds", "time_series") else "jodds_tanpuku"
df = store.read("odds", subpath, filters=filters)
```

問題:
- `time_series` ディレクトリが存在する → 常に `time_series` を選択
- `time_series` は 2015-2024 のみ (更新停止済み)
- 2025-2026 のクエリ → 0行 → フォールバック発生せず
- `jodds_tanpuku` (2015-2026、ETLで自動更新) が使われない

### データソース比較

| | time_series (旧ETL) | jodds_tanpuku (新ETL) |
|---|---|---|
| ソース | 旧々ETLで手動生成 | EveryDB2 n_jodds_tanpuku |
| 期間 | 2015-2024 | 2015-2026 |
| 更新 | 停止 | run_etl.py で自動 |
| 列名 | happyo_time, tan_odds | happyotime, tanodds, tanninki |

### 変更

`src/db/readers.py` に `import logging` と `logger = logging.getLogger(__name__)` を追加。

```python
subpath = "time_series" if store.exists("odds", "time_series") else "jodds_tanpuku"
df = store.read("odds", subpath, filters=filters)

# time_series が要求範囲のデータを持たない場合、jodds_tanpuku にフォールバック
# jodds_tanpuku も year/month パーティションなので、同一 filters がそのまま適用可能
if df.empty and subpath == "time_series" and store.exists("odds", "jodds_tanpuku"):
    logger.debug("time_series empty for %s-%s, falling back to jodds_tanpuku", start, end)
    df = store.read("odds", "jodds_tanpuku", filters=filters)
```

`load_odds_time_series()` (単一 race_id 版) にも同様のフォールバックを追加:

```python
subpath = "time_series" if store.exists("odds", "time_series") else "jodds_tanpuku"
df = store.read("odds", subpath, filters=[("race_id", "==", race_id)])

if df.empty and subpath == "time_series" and store.exists("odds", "jodds_tanpuku"):
    df = store.read("odds", "jodds_tanpuku", filters=[("race_id", "==", race_id)])
```

### 設計判断: パーティション事前チェックは不採用

代替案として「要求年のパーティションが存在するか事前チェック」も検討したが不採用:
- 年ディレクトリの存在 ≠ データの存在 (空パーティションの可能性)
- フォールバックロジックで十分対応可能
- 余計な複雑さを避ける

## Fix 3: odds10 異常値ガード

**ファイル**: `src/features/odds_dynamics_features.py`

### 現状

データ検証の結果、jodds_tanpuku の tanodds は odds10 変換済み (max=999.9、サンプル値 1.0, 2.7, 14.0 等)。直近の ETL 修正 (4229da0) で 2重適用バグも解消済み。

### 変更

念のため `compute_odds_dynamics()` に合理的オッズ範囲のガードを追加。

挿入位置: `ts = odds_ts.sort_values(...).copy()` (line 52) の直後、`tanninki` 正規化 (line 56) の前:

```python
# 合理的オッズ範囲外を NaN にする (1.0-999.9)
ts.loc[ts["tanodds"] < 1.0, "tanodds"] = np.nan
ts.loc[ts["tanodds"] > 999.9, "tanodds"] = np.nan
```

既存の `first_odds.replace(0, np.nan)` は残す (互換性のため)。

## Fix 4: bamei 文字化け修正

**ファイル**: `scripts/run_paper_trading.py`
**対象**: `_run_predict()` (line 266), `_run_dry_run()` (line 567-568)

### 現状

EveryDB2 の `bamei` 列は Shift-JIS バイト列が文字列として格納されている。推論結果の Slack 通知や JSON 保存で文字化けする。

### 変更

ヘルパー関数を追加:

```python
def _decode_bamei(name: str) -> str:
    """Shift-JIS バイト列の bamei をデコードする。"""
    if not isinstance(name, str):
        return str(name)
    try:
        return name.encode("latin-1").decode("shift_jis")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name
```

bamei を使用する箇所で `_decode_bamei(horse_name)` を呼び出す。

## 変更ファイル一覧

| ファイル | Fix | 変更内容 |
|----------|-----|----------|
| `scripts/run_paper_trading.py` | 1, 4 | odds_ts_df の読み込み・渡し、bamei デコード |
| `src/db/readers.py` | 2 | load_odds_time_series_range/load の空結果フォールバック |
| `src/features/odds_dynamics_features.py` | 3 | tanodds 異常値ガード |

## 影響範囲

- **TrainingPipelineV5**: 影響なし (既に odds_ts_df を正しく渡している)
- **BacktestEngine**: 影響なし (意図的に odds_ts_df=None でメモリ節約)
- **PaperPredictor.setup()**: 影響なし (別クラス、修正対象外)
- **FeatureEngine.build_features()**: 本修正対象外。`build_features()` は `compute_odds_dynamics()` を呼び出さない (line 182 はプレースホルダー)。`BettingOrchestrator` 経由のライブ予測で同問題が発生する場合は別途対応が必要。
- **テスト**: readers.py の既存テストにフォールバックのテストを追加

## テスト

### readers.py フォールバックテスト

```python
def test_load_odds_time_series_range_falls_back_to_jodds_tanpuku(self):
    """time_series が空の場合、jodds_tanpuku にフォールバックする。"""
    store = MagicMock()
    # time_series exists but returns empty; jodds_tanpuku has data
    store.exists.side_effect = lambda cat, name: name in ("time_series", "jodds_tanpuku")
    empty_df = pd.DataFrame()
    fallback_df = pd.DataFrame({
        "race_id": ["20260401010101"], "happyotime": ["03241000"],
        "umaban": [1], "tanodds": [3.0],
    })
    store.read.side_effect = [empty_df, fallback_df]
    result = load_odds_time_series_range(store, "20260401", "20260401")
    assert store.read.call_count == 2
    assert len(result) == 1

def test_load_odds_time_series_no_fallback_when_data_exists(self):
    """time_series にデータがある場合、フォールバックしない。"""
    store = MagicMock()
    store.exists.return_value = True
    valid_df = pd.DataFrame({
        "race_id": ["20240701010101"], "happyotime": ["03241000"],
        "umaban": [1], "tanodds": [5.4],
    })
    store.read.return_value = valid_df
    result = load_odds_time_series_range(store, "20240701", "20240701")
    assert store.read.call_count == 1
    assert len(result) == 1
```

### odds_dynamics_features.py 異常値ガードテスト

```python
def test_out_of_range_tanodds_produces_nan_features(self):
    """tanodds が範囲外 (1.0未満, 999.9超) の場合、特徴量が NaN になる。"""
    df = pd.DataFrame({"race_id": ["R1"], "umaban": [1]})
    ts = pd.DataFrame({
        "race_id": ["R1", "R1"], "umaban": [1, 1],
        "happyotime": [1, 2], "tanodds": [0.5, 1500.0],  # 範囲外
    })
    result = compute_odds_dynamics(df, ts)
    assert pd.isna(result["odds_drop_rate_60_10"].iloc[0])
```

## 検証方法

```bash
# dry-run で 2024-07-13 をテスト (time_series にデータあり)
python scripts/run_paper_trading.py --mode dry-run --date 2024-07-13

# dry-run で 2026-04-05 をテスト (jodds_tanpuku フォールバック)
python scripts/run_paper_trading.py --mode dry-run --date 2026-04-05

# 確認: odds_drop_rate_*, odds_velocity が NaN でないこと
# 確認: e_return_place_pred が馬ごとに異なる値を持つこと
```
