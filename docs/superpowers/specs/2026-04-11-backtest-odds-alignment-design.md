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

#### 精算オッズの渡し方

`Bet` データクラス（`src/domain/types.py`）に `final_odds: float` フィールドを追加。

```python
@dataclass
class Bet:
    race_id: str
    umaban: int
    bet_type: BetType
    odds: float           # 発走前オッズ（ベット判定に使用した値）
    final_odds: float     # 確定オッズ（精算に使用する値）
    ev_lower_corrected: float
    stake: float
```

- `Bet.odds`: 発走前オッズ（従来通り、EV判定に使用した値を記録）
- `Bet.final_odds`: 確定オッズ（精算用）

`_settle_bet()` は `bet.final_odds` を使用:
```python
return float(bet.stake * bet.final_odds)
```

`final_odds` の値は、`BacktestEngine._run_simulation()` 内で確定オッズの
`(race_id, umaban) -> fukuoddslow` マップを構築し、`select_bets()` の後に各 `Bet` に設定:

```python
# 確定オッズのマップ構築
final_odds_map = {
    (row["race_id"], row["umaban"]): row["fukuoddslow"]
    for _, row in final_odds_df.iterrows()
}

# ベット生成後
bets = race_predictor.select_bets(result_df, bankroll)
for bet in bets:
    bet = dataclasses.replace(bet, final_odds=final_odds_map.get(
        (bet.race_id, bet.umaban), bet.odds
    ))
```

#### `fukuoddslow_final` の生成方法

`FeatureEngine.build_all()` を呼ぶ前に、`pre_post_odds` に確定オッズを列として追加:

```python
# pre_post_odds に確定オッズを fukuoddslow_final として追加
pre_post_with_final = pre_post_odds.merge(
    final_odds_df[["race_id", "umaban", "fukuoddslow"]].rename(
        columns={"fukuoddslow": "fukuoddslow_final"}
    ),
    on=["race_id", "umaban"],
    how="left",
)
feat_df = feat_engine.build_all(race_df, entry_df, pre_post_with_final, odds_ts_df)
```

`FeatureEngine.build_all()` 自体は変更不要（`fukuoddslow_final` はマージされるが
EV計算には使われない。`final_odds_map` の構築に使用）。

#### `_generate_bets` レガシーメソッド

`_generate_bets()` (engine.py:429-464) も `Bet(odds=float(row["fukuoddslow"]))` を使用している。
このメソッドは `_run_simulation` から呼ばれなくなったレガシーだが、
互換性のため残している。同様に `final_odds` を設定するよう修正。

#### フォールバック

時系列データが存在しないレースでは、確定オッズをそのまま使用
（旧来の動作と同じ）。ログで警告を出力。
フォールバック時は `bet.odds == bet.final_odds` となる。

バックテスト結果には、発走前オッズでベットした件数とフォールバック件数を
メトリクスとして含め、結果の信頼性を評価可能にする。

### セクション3: その他の修正

#### 3a. `bankroll_after` バグ修正（ペーパートレード）

**ファイル:** `scripts/run_paper_trading.py`
**現状:** line 534 で `bankroll_after: bet.stake`（賭け金を誤設定）
**修正:** バンクロールを正しく追跡して `bankroll_after: round(bankroll, 2)` にする

#### 3b. BloodlineFeatures の扱い

今回は変更しない。学習時に使われていない列は `predict()` で無視されるため動作に影響なし。
将来的に血統特徴量をモデルに組み込む際にバックテストにも追加を検討。

#### 3c. 時系列オッズデータの可用性

**設計上の前提条件:**

- `data/odds/time_series/` は 2015-2024 年のデータを含む
- `data/odds/jodds_tanpuku/` は 2015-2026 年のデータを含む
- `load_odds_time_series_range()` は `time_series` → `jodds_tanpuku` へ自動フォールバック
- したがって、2025年テストでは `jodds_tanpuku` が使用される

**重要:** `_extract_pre_post_odds()` は `happyotime` 列を「MMDDHHmm」(8桁) 形式と想定。
`jodds_tanpuku` Parquet の `happyotime` が同じ形式であることを実装時に検証する。
形式が異なる場合は、正規化ロジックを追加。

実装前に以下を確認:
- `jodds_tanpuku` の `happyotime` 列の形式とサンプル値
- データが欠損している期間の割合
- フォールバック時の動作（確定オッズ使用 + 警告ログ）

#### 3d. スコープ外の対象

以下は同じルックアヘッド問題を抱えているが、**今回のスコープ外**とする:
- `_run_dry_run()` (run_paper_trading.py:1004-1153): 確定オッズスナップショットを使用
- `_run_diagnose()` (run_paper_trading.py:657-780): 確定オッズスナップショットを使用

これらは本格的なバックテストではなく、補助的な診断機能であるため。
将来のタスクで対応を検討。

## 影響範囲

| ファイル | 変更内容 |
|---------|---------|
| `src/db/odds_extractor.py` | 新規: 共通オッズ抽出関数 |
| `src/domain/types.py` | `Bet` に `final_odds: float` フィールド追加 |
| `src/backtest/engine.py` | オッズ取得・精算ロジック変更、`_generate_bets` レガシーメソッド対応 |
| `scripts/run_paper_trading.py` | `_extract_pre_post_odds` → import に変更、bankroll_after バグ修正 |
| テスト | `tests/test_backtest*.py` にオッズ関連テスト追加 |

**テストケース:**
1. 発走前オッズが `Bet.odds` に格納され、確定オッズが `Bet.final_odds` に格納されること
2. `_settle_bet` が `bet.final_odds`（`bet.odds` ではない）を使用すること
3. 時系列データがない場合のフォールバック動作が確定オッズを使用すること
4. `extract_pre_post_odds` が共有モジュールから正しくインポートされること
5. `jodds_tanpuku` の `happyotime` 形式が想定通りであること

## 期待される効果

- バックテストのベット頻度がペーパートレードに近づく（実装後に bets/race を計測・比較）
- バックテストの ROI は低下する可能性があるが、より現実的な数字になる
- ルックアヘッドバイアスの除去により、実運用への信頼性が向上
- バックテスト結果に「発走前オッズ使用件数 / フォールバック件数」のメトリクスを含め、結果の信頼性を評価可能にする
