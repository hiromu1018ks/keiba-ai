# ペーパートレード予測安定化 — 設計書

Date: 2026-04-11

## Problem

`run_paper_trading.py --mode predict` を複数回実行すると、同じ日の予測結果が毎回変わる。

原因: EveryDB2 のオッズが更新されるたびにスナップショットオッズ (`fukuoddslow`) が変化し、EV 計算結果が変わるため。終了したレースも再予測対象になる。

## Solution

EV 判定に使うオッズを「発走5分前」の時点に固定する。時系列オッズデータから各レースの `post_time - 5min` 時点のスナップショットを抽出し、`FeatureEngine.build_all()` に渡す `odds_df` を差し替える。

## Why This Works

1. **確定的**: 同じレースは常に同じ5分前オッズ → 何回実行しても同じ予測
2. **自然なフィルタ**: 5分前スナップショットが存在しないレース（まだ発走5分前になっていない）は自動的にスキップ
3. **バックテスト整合**: 発走直前のオッズで評価 = バックテストと同じ条件
4. **上書き安全**: 同じレースは常に同じ予測結果になるため、parquet の上書きでも問題ない
5. **シンプル**: タイムウィンドウ、フリーズ機構、追記ロジック不要

## Implementation

### 変更対象: `_run_predict()` in `scripts/run_paper_trading.py`

#### 1. 5分前オッズスナップショットの抽出

新関数 `_extract_pre_post_odds()` を追加。`odds_ts_df` から各レースの発走N分前時点のオッズスナップショットを抽出し、`FeatureEngine.build_all()` と互換性のある DataFrame を返す。

**出力スキーマ** (`load_odds_snapshots_from_db` と同じ列構成):

```
race_id, umaban, tanodds, fukuoddslow, tanninki, ...
```

`FeatureEngine.build_all()` は `odds_df[["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"]]` を merge するため、これらの列が含まれていればそのまま渡せる。

**タイムスタンプの比較方法** (`happyotime` と `hassotime` の形式差異を解消):

```python
# odds_ts_df の happyotime: "MMDDHHmm" (例: "04110930" = 4月11日09:30)
# race_df の hassotime: int/str "hhmm" (例: 930 → "09:30")
# race_df の race_date: YYYYMMDD (race_id の先頭8桁)

# hassotime → datetime 変換
ht_str = f"{int(hassotime):04d}"  # 930 → "0930"
post_time = datetime(race_year, race_month, race_day, int(ht_str[:2]), int(ht_str[2:]))

# happyotime → datetime 変換
mmddhhmm = str(happyotime)  # "04110930"
ht_full = datetime(year, int(mmddhhmm[:2]), int(mmddhhmm[2:4]),
                   int(mmddhhmm[4:6]), int(mmddhhmm[6:8]))

# cutoff 計算
cutoff = post_time - timedelta(minutes=minutes_before)
# cutoff 以前の最新エントリを取得
```

**鮮度ルール** (カットオフより古すぎるスナップショットを除外):

```python
max_staleness_minutes = 60  # スナップショットがカットオフの60分以上前なら除外
min_cutoff = cutoff - timedelta(minutes=max_staleness_minutes)
# min_cutoff < snapshot_time <= cutoff のエントリのみ採用
```

これにより、前日夜のオッズスナップショットが誤って採用されるのを防ぐ。

#### 2. _run_predict() のフロー変更

現状:
```python
odds_df = load_odds_snapshots_from_db(db, ymd)
odds_ts_df = load_odds_time_series_from_db(db, ymd)
feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df)
```

変更後:
```python
odds_ts_df = load_odds_time_series_from_db(db, ymd)
odds_snapshot_df = load_odds_snapshots_from_db(db, ymd)  # fallback用
odds_df = _extract_pre_post_odds(odds_ts_df, race_df, minutes_before=args.minutes_before)
# odds_df は build_all と同じスキーマ → そのまま渡す
feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df)
```

- `odds_ts_df` が空でない限り `_extract_pre_post_odds()` で生成した DataFrame を使用
- `odds_ts_df` が空の場合は `load_odds_snapshots_from_db()` の結果を fallback として使用（従来動作）
- 5分前スナップショットが取得できないレースの race_id を特定し、推論ループでスキップ

#### 3. スナップショットがないレースの扱い

- ログ: `[INFO] Skipping RACE_ID: no pre-post odds snapshot yet (post_time=10:05)`
- 予測対象から除外
- 除外レース数を最後にサマリ表示

#### 4. 出力の2セクション構成

```
============================================================
  Predict: 20260411  -  8 new bets (3 races skipped)
============================================================
  --- New Predictions ---
  10:35  中山 2R  馬番 5  ...  複勝  2.1  EV=1.94
  10:45  福島 3R  馬番12  ...  複勝  5.9  EV=1.84
  ...

  --- Previous Predictions (5 races) ---
  09:55  阪神 1R  馬番 8  ...  複勝  9.2  EV=2.16
  10:05  中山 1R  馬番 6  ...  複勝  5.7  EV=1.44
  ...
```

- **New Predictions**: 今回新たに予測したベット
- **Previous Predictions**: 過去に予測済みのベット（見返し用）
- **Skipped**: 5分前スナップショットがないレースの数

**New と Previous の判定方法**:

1. 既存の `{ymd}.parquet` を読み込み（存在する場合）、既存の `race_id` 一覧を取得
2. 新しい予測結果のうち、既存 `race_id` に含まれないものを "New" と判定
3. 既存のものを "Previous" として表示
4. 新しい予測を既存に追記して保存

#### 5. predictions parquet の扱い

- 既存の parquet を読み込み
- 新しい予測結果を追記（`pd.concat([existing, new])`）
- 同じ race_id の予測は同じ結果になるため、重複が発生しても実害なし
- `predicted_at` タイムスタンプ列 (ISO8601) を追加し、各予測の実行時刻を記録

#### 6. 設定

- `--minutes-before 5`: デフォルト5分。変更可能。
- `--reset` フラグは不要（同じオッズ → 同じ結果 → リセットの意味がない）。除外。

### 初期 empty チェックの扱い

現状 (254行目):
```python
if race_df.empty or entry_df.empty or odds_df.empty or odds_ts_df.empty:
    logger.error("EveryDB2 からデータ取得失敗: %s", ymd)
    return
```

変更後:
```python
# odds_df は odds_ts_df から生成されるため、空チェックは odds_ts_df で判定
if race_df.empty or entry_df.empty:
    logger.error("EveryDB2 からデータ取得失敗: %s", ymd)
    return
if odds_ts_df.empty:
    logger.warning("No odds time series for %s, falling back to snapshots", ymd)
    odds_df = load_odds_snapshots_from_db(db, ymd)
    # 従来動作で続行
```

### 影響範囲

| ファイル | 変更 | 影響 |
|---------|------|------|
| `scripts/run_paper_trading.py` | `_run_predict()` のオッズ抽出ロジック変更、出力フォーマット変更 | メイン変更箇所 |
| `scripts/run_paper_trading.py` | 新関数 `_extract_pre_post_odds()` 追加 | ~40行の新規関数 |
| `scripts/run_paper_trading.py` | `--minutes-before` 引数追加 | argparse 変更 |

reconcile, report, dry-run, diagnose は **変更不要**。

- reconcile は `result == 0.0` で未確定を判定（`predicted_at` 列は無視される）
- report は `bets.parquet` から生成（スキーマ互換）
- dry-run/diagnose は Parquet データを使用（EveryDB2 フローとは別）

### reconcile との整合性

reconcile は予測時のオッズ（5分前固定）と実際の払戻オッズを別々に扱っている:
- 予測 parquet の `odds` 列 = 5分前オッズ（ベット判定に使用）
- reconcile が払戻テーブルから取得する `actual_odds` = 確定払戻オッズ（結果計算に使用）

この分離は正しく機能する。reconcile は「予測したオッズでベットした場合、実際にどうだったか」を評価する。

## Out of Scope

- Phase 2 自動化 (別設計: `race-day-automation-design.md`)
- `dry-run` / `diagnose` モードのオッズ固定（将来的に検討。現在は Parquet データを使用するため直接の影響なし）
- Slack 通知フォーマットの変更（出力に合わせて自動追従）
