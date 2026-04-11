# ペーパートレード予測安定化 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** EV判定オッズを発走5分前に固定し、予測結果を確定的にする。

**Architecture:** `odds_ts_df` の時系列オッズから各レースの `post_time - 5min` 時点のスナップショットを抽出し、`FeatureEngine.build_all()` に渡す `odds_df` を差し替える。これにより何回実行しても同じ予測結果が得られる。

**Tech Stack:** Python 3.11, pandas, pytest (mock-based)

**Spec:** `docs/superpowers/specs/2026-04-11-predict-stability-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/run_paper_trading.py` | Modify | `_run_predict()` のオッズフロー変更、`_extract_pre_post_odds()` 追加、出力2セクション化、argparse 追加 |
| `tests/test_predict_stability.py` | Create | `_extract_pre_post_odds()` の単体テスト + `_run_predict` の統合テスト |

---

### Task 1: `_extract_pre_post_odds()` の実装

**Files:**
- Modify: `scripts/run_paper_trading.py` (新関数追加、ファイル末尾 `_run_predict` の前に挿入)
- Test: `tests/test_predict_stability.py` (新規作成)

**背景:**
- `odds_ts_df` の `happyotime` は `"MMDDHHmm"` (8桁文字列)。年情報は `year` 列から取得
- `race_df` の `hassotime` は int `hhmm` (例: `930` = 09:30)。日付は `race_id` の先頭8桁から取得
- `build_all()` は `odds_df[["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"]]` のみ使用
- オッズは x10 スケール (DB仕様。変換は readers.py で既に済み)

- [ ] **Step 1: テストファイル作成 + 失敗テストを書く**

`tests/test_predict_stability.py` を新規作成:

```python
"""tests/test_predict_stability.py — ペーパートレード予測安定化のテスト"""

from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# _extract_pre_post_odds() のテスト
# ---------------------------------------------------------------------------


def _make_odds_ts_df(entries: list[dict]) -> pd.DataFrame:
    """テスト用の odds_ts_df を構築するヘルパー。"""
    return pd.DataFrame(entries)


def _make_race_df(entries: list[dict]) -> pd.DataFrame:
    """テスト用の race_df を構築するヘルパー。"""
    return pd.DataFrame(entries)


class TestExtractPrePostOdds:
    """_extract_pre_post_odds() のテスト群。"""

    def test_basic_extraction(self) -> None:
        """発走5分前のスナップショットが正しく抽出される。"""
        # import は関数定義後に成功する。ここでは関数名だけでテスト構造を示す。
        # 実装後に import してテストする。
        from pathlib import sys
        # テスト対象のインポート
        import importlib
        # scripts/run_paper_trading.py はスクリプトなので直接 import できない。
        # 代わりにテスト用ヘルパーとして同じロジックを実装してテストする。
        # → 実装後に関数を抽出してテスト可能にする。
        pass

    def test_no_snapshot_before_cutoff_skips_race(self) -> None:
        """cutoff 以前のエントリがないレースは結果に含まれない。"""
        pass

    def test_stale_snapshot_excluded(self) -> None:
        """cutoff の60分以上前のスナップショットは除外される。"""
        pass

    def test_output_schema_compatible_with_build_all(self) -> None:
        """出力 DataFrame が race_id, umaban, tanodds, fukuoddslow, tanninki を含む。"""
        pass
```

※ テストは関数実装後に有効化する。まずはスケルトンとして作成。

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_predict_stability.py -v`
Expected: テストは `pass` する（スケルトンなので）。実装後にアサーションを追加。

- [ ] **Step 3: `_extract_pre_post_odds()` の実装**

`scripts/run_paper_trading.py` の `_run_predict()` の前に追加:

```python
def _extract_pre_post_odds(
    odds_ts_df: pd.DataFrame,
    race_df: pd.DataFrame,
    minutes_before: int = 5,
    max_staleness_minutes: int = 60,
) -> pd.DataFrame:
    """各レースの発走N分前時点のオッズスナップショットを抽出.

    Parameters
    ----------
    odds_ts_df : DataFrame
        時系列オッズ。happyotime (str "MMDDHHmm"), year, umaban 等を含む。
    race_df : DataFrame
        レース情報。hassotime (int "hhmm"), race_id 等を含む。
    minutes_before : int
        発走何分前のオッズを使うか (デフォルト: 5)。
    max_staleness_minutes : int
        cutoff から何分以上前のスナップショットを除外するか (デフォルト: 60)。

    Returns
    -------
    DataFrame
        build_all() と互換のスキーマ:
        race_id, umaban, tanodds, fukuoddslow, tanninki
    """
    import pandas as pd
    from datetime import datetime, timedelta

    if odds_ts_df.empty or race_df.empty:
        return pd.DataFrame(
            columns=["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"]
        )

    # 1. race_id → post_datetime のマッピング
    post_time_map: dict[str, datetime] = {}
    for _, r in race_df.iterrows():
        ht = r.get("hassotime")
        if pd.isna(ht) or str(ht).strip() == "":
            continue
        ht_str = f"{int(ht):04d}"  # 930 → "0930"
        # race_id の先頭8桁 = YYYYMMDD
        rid = r["race_id"]
        race_date_str = rid[:8]
        post_time_map[rid] = datetime(
            int(race_date_str[:4]),
            int(race_date_str[4:6]),
            int(race_date_str[6:8]),
            int(ht_str[:2]),
            int(ht_str[2:]),
        )

    # 2. odds_ts_df の各行について happyotime → datetime
    def _parse_happyotime(row: pd.Series) -> datetime | None:
        ht = row.get("happyotime")
        if pd.isna(ht) or not isinstance(ht, str) or len(ht) != 8:
            return None
        year = int(row["year"])
        month = int(ht[:2])
        day = int(ht[2:4])
        hour = int(ht[4:6])
        minute = int(ht[6:8])
        return datetime(year, month, day, hour, minute)

    odds_ts_df = odds_ts_df.copy()
    odds_ts_df["_ht_datetime"] = odds_ts_df.apply(_parse_happyotime, axis=1)
    odds_ts_df = odds_ts_df[odds_ts_df["_ht_datetime"].notna()]

    # 3. 各行に cutoff を付与し、cutoff 以前のエントリのみ残す
    def _is_before_cutoff(row: pd.Series) -> bool:
        post_time = post_time_map.get(row["race_id"])
        if post_time is None:
            return False
        cutoff = post_time - timedelta(minutes=minutes_before)
        min_cutoff = cutoff - timedelta(minutes=max_staleness_minutes)
        ht_dt = row["_ht_datetime"]
        return min_cutoff <= ht_dt <= cutoff

    mask = odds_ts_df.apply(_is_before_cutoff, axis=1)
    valid = odds_ts_df[mask]

    if valid.empty:
        return pd.DataFrame(
            columns=["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"]
        )

    # 4. (race_id, umaban) ごとに最新エントリを取得
    idx = valid.groupby(["race_id", "umaban"])["_ht_datetime"].idxmax()
    snapshot = valid.loc[idx]

    # 5. build_all() と互換のスキーマで返す
    result = snapshot[["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"]].copy()
    result = result.reset_index(drop=True)
    return result
```

- [ ] **Step 4: テストを有効化して通ることを確認**

テストの `pass` を実際のアサーションに置き換える。Run: `python -m pytest tests/test_predict_stability.py -v`
Expected: 全テスト PASS

- [ ] **Step 5: コミット**

```bash
git add scripts/run_paper_trading.py tests/test_predict_stability.py
git commit -m "feat: _extract_pre_post_odds() 追加 (発走N分前オッズスナップショット抽出)"
```

### Task 2: `_run_predict()` のオッズフロー変更 + `--minutes-before` 引数追加

**Files:**
- Modify: `scripts/run_paper_trading.py:_run_predict()` (lines ~246-262)
- Modify: `scripts/run_paper_trading.py:parse_args()` (line ~53)

**現在のコード (lines 246-262):**
```python
# EveryDB2からデータ読み込み
db = EveryDB2Queries(config.everydb2_connection_string)
race_df = load_races_from_db(db, ymd)
entry_df = load_entries_from_db(db, ymd)
odds_df = load_odds_snapshots_from_db(db, ymd)
odds_ts_df = load_odds_time_series_from_db(db, ymd)

if race_df.empty or entry_df.empty or odds_df.empty or odds_ts_df.empty:
    logger.error("EveryDB2 からデータ取得失敗: %s", ymd)
    return

# 特徴量生成
feat_engine = FeatureEngine()
submodel_mgr = SubModelManager()
feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df)
```

**変更後:**
```python
# EveryDB2からデータ読み込み
db = EveryDB2Queries(config.everydb2_connection_string)
race_df = load_races_from_db(db, ymd)
entry_df = load_entries_from_db(db, ymd)
odds_snapshot_df = load_odds_snapshots_from_db(db, ymd)  # fallback用
odds_ts_df = load_odds_time_series_from_db(db, ymd)

if race_df.empty or entry_df.empty:
    logger.error("EveryDB2 からデータ取得失敗: %s", ymd)
    return

# 発走N分前のオッズスナップショットを生成
minutes_before = getattr(args, "minutes_before", 5)
if odds_ts_df.empty:
    logger.warning("No odds time series for %s, falling back to snapshots", ymd)
    odds_df = odds_snapshot_df
else:
    odds_df = _extract_pre_post_odds(odds_ts_df, race_df, minutes_before=minutes_before)
    if odds_df.empty:
        logger.warning("No pre-post odds extracted for %s, falling back to snapshots", ymd)
        odds_df = odds_snapshot_df

# 5分前スナップショットがないレースの race_id を特定
if not odds_ts_df.empty and odds_df is not odds_snapshot_df:
    all_race_ids = set(race_df["race_id"].unique())
    covered_race_ids = set(odds_df["race_id"].unique())
    skipped_race_ids = all_race_ids - covered_race_ids
    for rid in sorted(skipped_race_ids):
        post_time = _race_time_map.get(rid, "??")
        logger.info("Skipping %s: no pre-post odds snapshot yet (post_time=%s)", rid, post_time)
else:
    skipped_race_ids = set()

# 特徴量生成 (odds_df を差し替え)
feat_engine = FeatureEngine()
submodel_mgr = SubModelManager()
feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df)
```

**`parse_args()` の変更 (line ~68 に追加):**
```python
parser.add_argument(
    "--minutes-before", type=int, default=5,
    help="発走何分前のオッズを使用するか (デフォルト: 5)",
)
```

**`main()` の変更:** `_run_predict(args, config, models, store)` は既に args を受け取っているので変更不要。

- [ ] **Step 1: `parse_args()` に `--minutes-before` を追加**
- [ ] **Step 2: `_run_predict()` のオッズ読み込み部分を上記に差し替え**
- [ ] **Step 3: `skipped_race_ids` を使って推論ループでスキップ**

`_run_predict()` の推論ループ (line ~294) で `skipped_race_ids` を除外:

```python
for race_id in race_ids:
    if race_id in skipped_race_ids:
        continue  # 5分前スナップショットなし → スキップ
    # ... 既存の推論ロジック
```

- [ ] **Step 4: コミット**

```bash
git add scripts/run_paper_trading.py
git commit -m "feat: _run_predict() のオッズを発走N分前に固定"
```

### Task 3: 出力の2セクション化 + `predicted_at` 列追加

**Files:**
- Modify: `scripts/run_paper_trading.py:_run_predict()` (lines ~292-449)

**変更概要:**
1. 推論前に既存の `predictions/{ymd}.parquet` を読み込み
2. 新しいベットに `predicted_at` タイムスタンプを付与
3. 新規 + 既存を追記保存
4. 出力を "New Predictions" と "Previous Predictions" に分離

**Step 1: 既存予測の読み込み + `predicted_at` 列追加**

推論ループの前に追加 (line ~292 の前):

```python
# 既存予測の読み込み
pred_path = config.paper_trading_dir / "predictions" / f"{ymd}.parquet"
existing_pred_df = pd.DataFrame()
existing_race_ids: set[str] = set()
if pred_path.exists():
    existing_pred_df = pd.read_parquet(pred_path)
    existing_race_ids = set(existing_pred_df["race_id"].unique())
```

各ベットに `predicted_at` を付与 (line ~376 の `all_bets.append` 内):

```python
all_bets.append({
    ...  # 既存のフィールド
    "predicted_at": datetime.now().isoformat(),
})
```

**Step 2: 保存ロジックの変更 (既存 line 396-399)**

```python
# 予測結果を保存 (追記)
new_pred_df = pd.DataFrame(all_bets)
if not existing_pred_df.empty:
    combined_pred_df = pd.concat([existing_pred_df, new_pred_df], ignore_index=True)
else:
    combined_pred_df = new_pred_df
combined_pred_df.to_parquet(pred_path, index=False)
logger.info("Predictions saved: %d new + %d existing → %s",
            len(all_bets), len(existing_pred_df), pred_path)
```

**Step 3: 出力フォーマット変更 (既存 lines 422-449)**

```python
# --- New Predictions ---
new_bets = [b for b in all_bets if b["race_id"] not in existing_race_ids]
# --- Previous Predictions ---
prev_bets_from_df = existing_pred_df.to_dict("records") if not existing_pred_df.empty else []

lines: list[str] = []
lines.append("")
lines.append("=" * 60)
lines.append(f"  Predict: {args.date}  -  {len(new_bets)} new bets  ({len(skipped_race_ids)} races skipped)")
lines.append("=" * 60)

if new_bets:
    lines.append("  --- New Predictions ---")
    new_bets.sort(key=lambda b: b.get("post_time", "99:99"))
    for b in new_bets:
        t = b.get("post_time", "--:--")
        lines.append(
            f"  {t}  {_fmt_race_id(b['race_id'])}  "
            f"馬番{int(b['umaban']):2d}  {b['horse_name']:<16s}  "
            f"複勝{b['odds']:5.1f}  EV={b['ev']:.2f}"
        )

if prev_bets_from_df:
    lines.append(f"  --- Previous Predictions ({len(prev_bets_from_df)} bets) ---")
    prev_bets_from_df.sort(key=lambda b: b.get("post_time", "99:99"))
    for b in prev_bets_from_df:
        t = b.get("post_time", "--:--")
        name = b.get("horse_name", "")
        lines.append(
            f"  {t}  {_fmt_race_id(b['race_id'])}  "
            f"馬番{int(b['umaban']):2d}  {name:<16s}  "
            f"複勝{b['odds']:5.1f}  EV={b['ev']:.2f}"
        )

lines.append("")
```

※ `_fmt_race_id` と `_venue_map` は既存コード (lines 403-420) をそのまま使用。

- [ ] **Step 4: コミット**

```bash
git add scripts/run_paper_trading.py
git commit -m "feat: predict出力の2セクション化 + predicted_at列追加"
```

### Task 4: 動作確認

**Files:**
- None (testing only)

- [ ] **Step 1: `python -m pytest tests/ -v` で全テスト通過確認**

Run: `python -m pytest tests/ -v`
Expected: 全テスト PASS (既存テスト + 新規テスト)

- [ ] **Step 2: コミット (不要ならスキップ)**
