# バックテスト・ペーパートレード オッズ整合設計

日付: 2026-04-11

## 背景

バックテスト（2025テスト, ensemble）の結果: 2,056 bets / ~3,455 races = **0.6 bets/race**
ペーパートレード（2026-04-11）の結果: 44 bets / 26 races = **1.7 bets/race** (3倍の乖離)

原因はオッズソースの違い:
- **バックテスト**: `n_odds_tanpuku`（蓄積テーブル, レース・馬ごとに1行 = 確定オッズ）を使用
- **ペーパートレード**: `n_jodds_tanpuku`（時系列テーブル）から `_extract_pre_post_odds()` で発走5分前のオッズを抽出

確定オッズは市場が完全に収束した後の値なため、モデルが「割安」と判定する馬が少なくなる。
発走5分前のオッズはまだ変動途中で、EVが高く計算されやすく、より多くの馬が閾値を通過する。

さらに、バックテストは **確定オッズを使ってベット判定 AND 精算** を行っており、
これはルックアヘッドバイアス（未来の情報を使った判定）に該当する。

## 設計

### セクション1: 共通オッズ抽出モジュール

**新規ファイル:** `src/db/odds_extractor.py`

`_extract_pre_post_odds()` を `scripts/run_paper_trading.py:224-329` から移動。

```python
def extract_pre_post_odds(
    odds_ts_df: pd.DataFrame,
    race_df: pd.DataFrame,
    minutes_before: int = 5,
    max_staleness_minutes: int = 60,
) -> pd.DataFrame:
    """各レースの発走N分前時点のオッズスナップショットを抽出。

    戻り値: race_id, umaban, tanodds, fukuoddslow, tanninki 列を持つ DataFrame
    """
```

`run_paper_trading.py` 側は `from db.odds_extractor import extract_pre_post_odds` に変更。

### セクション2: バックテストエンジンのオッズ変更

**ファイル:** `src/backtest/engine.py`

#### 変更前の流れ

```
1. load_odds_snapshots(store, start, end)        → 確定オッズ
2. load_odds_time_series_range(store, start, end) → 時系列オッズ
3. feat_df = build_all(race_df, entry_df, odds_df, odds_ts_df)
4. for race in races:
       select_bets(race_df, bankroll)   # fukuoddslow = 確定オッズ
       settle(bet)                      # payout = stake × 確定オッズ
```

#### 変更後の流れ

```
1. load_odds_snapshots(store, start, end)         → 確定オッズ（精算用に保持）
2. load_odds_time_series_range(store, start, end)  → 時系列オッズ
3. pre_post_odds = extract_pre_post_odds(odds_ts_df, race_df)
4. feat_df = build_all(race_df, entry_df, pre_post_odds, odds_ts_df)
5. for race in races:
       select_bets(race_df, bankroll)    # fukuoddslow = 発走5分前オッズ
       settle(bet, final_odds_map)       # payout = stake × 確定オッズ
```

**重要な分離:**
- ベット判定（EV計算、閾値判定）: 発走前オッズ
- 精算（払戻計算）: 確定オッズ

確定オッズは別列 `fukuoddslow_final` として保持。`Bet.odds` には発走前オッズを格納し、
精算時に `final_odds_map[(race_id, umaban)]` から確定オッズを引いて payout を計算。

**フォールバック:** 時系列データが存在しないレースでは、確定オッズをそのまま使用
（旧来の動作と同じ）。ログで警告を出力。

### セクション3: その他の修正

#### 3a. `bankroll_after` バグ修正（ペーパートレード）

**ファイル:** `scripts/run_paper_trading.py`
**現状:** line 534 で `bankroll_after: bet.stake`（賭け金を誤設定）
**修正:** バンクロールを正しく追跡して `bankroll_after: round(bankroll, 2)` にする

#### 3b. BloodlineFeatures の扱い

今回は変更しない。学習時に使われていない列は `predict()` で無視されるため動作に影響なし。
将来的に血統特徴量をモデルに組み込む際にバックテストにも追加を検討。

#### 3c. 時系列オッズデータの可用性

実装前に以下を確認:
- `data/odds/jodds_tanpuku/` または `data/odds/time_series/` にテスト期間のデータが存在するか
- データが欠損している期間の割合
- フォールバック時の動作（確定オッズ使用 + 警告ログ）

## 影響範囲

| ファイル | 変更内容 |
|---------|---------|
| `src/db/odds_extractor.py` | 新規: 共通オッズ抽出関数 |
| `src/backtest/engine.py` | オッズ取得・精算ロジック変更 |
| `scripts/run_paper_trading.py` | `_extract_pre_post_odds` → import に変更、bankroll_after バグ修正 |
| テスト | `tests/test_backtest*.py` にオッズ関連テスト追加 |

## 期待される効果

- バックテストのベット頻度がペーパートレードに近づく（0.6 → ~1.0-1.7 bets/race 程度）
- バックテストの ROI は低下する可能性があるが、より現実的な数字になる
- ルックアヘッドバイアスの除去により、実運用への信頼性が向上
