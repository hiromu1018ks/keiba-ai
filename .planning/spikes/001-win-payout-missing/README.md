---
spike: 001
name: win-payout-missing
type: standard
validates: "Given run_backtest in WIN mode, when 'Win payout missing' warnings appear, then the root cause can be identified as noise vs real data gap"
verdict: VALIDATED
related: []
tags: [backtest, payouts, data-quality, logging]
---

# Spike 001: Win Payout Missing 原因調査

## What This Validates

Given run_backtest in WIN mode, when "Win payout missing" warnings appear, then the root cause can be identified and quantified.

## Research

調査対象:
- `src/backtest/engine.py` — `_settle_bet()` (line 1312-1327), `build_win_payout_map()` (line 175-195)
- `src/db/readers.py` — `load_payouts()` (line 241-243)
- `src/db/everydb2_queries.py` — `get_payouts()` (line 260-305)
- `data/raw/payouts.parquet` — 払戻データ
- `data/raw/entries.parquet` — 出走データ

## How to Run

```bash
cd <project root>
python .planning/spikes/001-win-payout-missing/diagnose_payouts.py
python .planning/spikes/001-win-payout-missing/diagnose_jra_only.py
```

## Investigation Trail

### Iteration 1: 全期間データ分析

- payouts: 38,835 rows, **paytansyoumaban1/paytansyopay1 の NULL は 0件** — payouts テーブル自体はクリーン
- entries: 852,216 rows, 69,457 unique races
- 30,634 races in entries but NOT in payouts (44%)

### Iteration 2: jyocd で JRA/地方 を分離

**payouts は全件 JRA (jyocd 1-10)**。entries には地方競馬 (jyocd 30-55等) も含まれる。
JRA限定でのカバレッジ:
- JRA entries races: 39,013
- JRA payout races: 38,835
- **JRA races missing from payouts: 190 (0.5%)**
  - うち **151件は 2026年** (未ETLの最新レース)
  - 残り **39件は 2015-2025年** (ETLギャップ)
- **JRA winners missing from payout map: 67 (0.2%)**

### Iteration 3: ノイズ分析

`_settle_bet()` の WIN 分岐 (engine.py:1312-1327):
1. `(race_id, umaban)` を `win_payout_map` から lookup
2. **見つからなければ常に警告** → フォールバックへ
3. フォールバック: 着順確認 → 1着なら odds 使用、それ以外は 0.0

`build_win_payout_map()` は `paytansyoumaban1` (1着馬番) のみをキーに登録する。
**したがって、1着以外の馬にベットした場合は常に lookup miss → 警告が発火する**。
これは正常動作（返り値 0.0 は正しい）だが、警告ログが大量のノイズとなる。

## Results

**Verdict: VALIDATED**

3つの原因が特定できた:

### 原因1: ノイズ警告（主原因、~95%以上の警告）

`_settle_bet()` が非1着馬の WIN ベットでも「Win payout missing」警告を出す。
`win_payout_map` は1着馬しか含まないため、1着以外のベットは全て miss する。
フォールバックの返り値 (0.0) は正しいが、**警告が大量に出る**。

**修正案:** 1着確認を先に行い、1着の場合のみ payout lookup → miss の場合のみ警告とする。
または、lookup miss 時は DEBUG レベルに下げる。

### 原因2: 過去データのETLギャップ（軽微、0.2%）

2015-2025年の JRA レースで **67レース** (0.2%) の1着馬が payouts に存在しない。
これらは取消レース・データ欠損等が原因と推定。
フォールバックで final_odds を使用するため、精算額にわずかな誤差が生じる。

### 原因3: 2026年データの未ETL（151レース）

2026年のレースは payouts がまだロードされていない。
現在日付以降のレースを実行した場合に発生。
