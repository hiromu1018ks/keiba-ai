# バックテスト オッズ整合 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** バックテストのオッズを確定オッズから発走5分前オッズに変更し、ルックアヘッドバイアスを除去する。精算は引き続き確定オッズを使用。

**Architecture:** `_extract_pre_post_odds()` を共通モジュールに抽出し、BacktestEngine から呼び出す。Bet データクラスに `final_odds` を追加し、ベット判定には発走前オッズ、精算には確定オッズを使う分離を実現。

**Tech Stack:** Python 3.11, pandas, pytest

**Spec:** `docs/superpowers/specs/2026-04-11-backtest-odds-alignment-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/db/odds_extractor.py` | **Create** | `extract_pre_post_odds()` 共通関数 |
| `src/domain/models.py` | **Modify** (line ~153) | `Bet` に `final_odds` フィールド追加 |
| `src/backtest/engine.py` | **Modify** (lines 108-128, 286-287, 466-479, 429-464) | オッズ取得・精算ロジック変更 |
| `src/backtest/race_predictor.py` | **No change** | `select_bets()` は変更なし（渡されたオッズを使う） |
| `scripts/run_paper_trading.py` | **Modify** (lines 224-329, 534) | import 切り替え + bankroll_after バグ修正 |
| `tests/test_odds_extractor.py` | **Create** | 共通関数のテスト |
| `tests/test_backtest_engine.py` | **Modify** | `final_odds`・精算ロジックのテスト |

---

### Task 1: `extract_pre_post_odds` 共通モジュール抽出

**Files:**
- Create: `src/db/odds_extractor.py`
- Modify: `scripts/run_paper_trading.py:224-329`
- Create: `tests/test_odds_extractor.py`

- [ ] **Step 1: テストを書く**

`tests/test_odds_extractor.py` を作成:

```python
"""odds_extractor のテスト"""
from __future__ import annotations

import pandas as pd
import pytest
from datetime import datetime


class TestExtractPrePostOdds:
    """extract_pre_post_odds のテスト"""

    def test_basic_extraction(self) -> None:
        """発走5分前のオッズを正しく抽出する"""
        from db.odds_extractor import extract_pre_post_odds

        race_df = pd.DataFrame(
            {
                "race_id": ["20250401110101"],
                "hassotime": [930],  # 09:30 発走
            }
        )
        odds_ts_df = pd.DataFrame(
            {
                "race_id": ["20250401110101"] * 3,
                "umaban": [1, 1, 1],
                "year": [2025, 2025, 2025],
                "happyotime": ["04010920", "04010925", "04010930"],
                "tanodds": [5.0, 4.8, 4.5],
                "fukuoddslow": [1.3, 1.3, 1.2],
                "tanninki": [3, 3, 3],
            }
        )
        # cutoff = 09:30 - 5min = 09:25
        # valid entries: 09:20 (min_cutoff=08:25以内) and 09:25
        # latest = 09:25
        now = datetime(2025, 4, 1, 12, 0)
        result = extract_pre_post_odds(odds_ts_df, race_df, minutes_before=5, _now=now)

        assert len(result) == 1
        assert result.iloc[0]["fukuoddslow"] == 1.3  # 09:25時点の値
        assert result.iloc[0]["tanodds"] == 4.8

    def test_empty_odds_returns_empty(self) -> None:
        """空の時系列データは空DataFrameを返す"""
        from db.odds_extractor import extract_pre_post_odds

        result = extract_pre_post_odds(
            pd.DataFrame(),
            pd.DataFrame({"race_id": ["20250401110101"], "hassotime": [930]}),
        )
        assert result.empty
        assert "fukuoddslow" in result.columns

    def test_no_valid_entries_returns_empty(self) -> None:
        """有効なエントリがない場合は空DataFrameを返す"""
        from db.odds_extractor import extract_pre_post_odds

        race_df = pd.DataFrame({"race_id": ["20250401110101"], "hassotime": [930]})
        # cutoff = 09:25, このデータは09:25より後なので除外
        odds_ts_df = pd.DataFrame(
            {
                "race_id": ["20250401110101"],
                "umaban": [1],
                "year": [2025],
                "happyotime": ["04010930"],  # 09:30 は cutoff 09:25 より後
                "tanodds": [4.5],
                "fukuoddslow": [1.2],
                "tanninki": [3],
            }
        )
        now = datetime(2025, 4, 1, 12, 0)
        result = extract_pre_post_odds(odds_ts_df, race_df, minutes_before=5, _now=now)
        assert result.empty

    def test_multiple_horses_per_race(self) -> None:
        """1レース複数馬の最新エントリを取得"""
        from db.odds_extractor import extract_pre_post_odds

        race_df = pd.DataFrame({"race_id": ["20250401110101"], "hassotime": [930]})
        odds_ts_df = pd.DataFrame(
            {
                "race_id": ["20250401110101"] * 4,
                "umaban": [1, 1, 2, 2],
                "year": [2025, 2025, 2025, 2025],
                "happyotime": ["04010920", "04010925", "04010920", "04010925"],
                "tanodds": [5.0, 4.8, 10.0, 9.5],
                "fukuoddslow": [1.3, 1.3, 2.5, 2.4],
                "tanninki": [3, 3, 7, 7],
            }
        )
        now = datetime(2025, 4, 1, 12, 0)
        result = extract_pre_post_odds(odds_ts_df, race_df, minutes_before=5, _now=now)

        assert len(result) == 2
        horse1 = result[result["umaban"] == 1].iloc[0]
        horse2 = result[result["umaban"] == 2].iloc[0]
        assert horse1["fukuoddslow"] == 1.3  # 09:25時点
        assert horse2["fukuoddslow"] == 2.4  # 09:25時点
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_odds_extractor.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'db.odds_extractor')

- [ ] **Step 3: `src/db/odds_extractor.py` を作成**

`scripts/run_paper_trading.py:224-329` の `_extract_pre_post_odds` をコピーし、
プライベート関数 `_extract_pre_post_odds` → パブリック関数 `extract_pre_post_odds` にリネーム。
内部 import (`from datetime import datetime, timedelta`, `import pandas as pd`) は関数外に移動。

```python
"""発走前オッズ抽出ユーティリティ"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


def extract_pre_post_odds(
    odds_ts_df: pd.DataFrame,
    race_df: pd.DataFrame,
    minutes_before: int = 5,
    max_staleness_minutes: int = 60,
    *,
    _now: datetime | None = None,
) -> pd.DataFrame:
    """各レースの発走N分前時点のオッズスナップショットを抽出。

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
    _now : datetime, optional
        現在時刻のオーバーライド (テスト用)。未指定時は datetime.now()。

    Returns
    -------
    DataFrame
        build_all() と互換のスキーマ:
        race_id, umaban, tanodds, fukuoddslow, tanninki
    """
    # run_paper_trading.py:253-329 のロジックをそのまま移動
    # （内部 import は不要、モジュールレベルで import 済み）
    ...
```

実装は `run_paper_trading.py:253-329` のコードをそのまま使用。
変更点:
- `from datetime import datetime, timedelta` / `import pandas as pd` → モジュールレベルへ
- 関数名の先頭アンダースコアを削除

- [ ] **Step 4: テストを実行して成功を確認**

Run: `python -m pytest tests/test_odds_extractor.py -v`
Expected: 4 passed

- [ ] **Step 5: `run_paper_trading.py` の import を変更**

`scripts/run_paper_trading.py` の224-329行を削除し、ファイル先頭の import 群に追加:

```python
from db.odds_extractor import extract_pre_post_odds
```

`_extract_pre_post_odds` の呼び出し箇所（387行付近）を `extract_pre_post_odds` に変更。

- [ ] **Step 6: 既存テストで回帰がないことを確認**

Run: `python -m pytest tests/ -v`
Expected: All existing tests pass

- [ ] **Step 7: コミット**

```bash
git add src/db/odds_extractor.py tests/test_odds_extractor.py scripts/run_paper_trading.py
git commit -m "feat: extract_pre_post_odds を共通モジュールに抽出"
```

---

### Task 2: `Bet` データクラスに `final_odds` フィールド追加

**Files:**
- Modify: `src/domain/models.py:145-166`
- Modify: `tests/test_backtest_engine.py` (Bet構築箇所の更新)

- [ ] **Step 1: テストを書く**

`tests/test_backtest_engine.py` の `TestBacktestResult` に追加:

```python
def test_bet_final_odds_default(self) -> None:
    """Bet.final_odds のデフォルトは 0.0"""
    from domain.models import Bet, BetType

    bet = Bet(
        race_id="20250401110101",
        umaban=1,
        bet_type=BetType.PLACE,
        odds=1.3,
        ev_lower_corrected=1.5,
        stake=100.0,
    )
    assert bet.final_odds == 0.0
    assert bet.odds == 1.3


def test_bet_final_odds_set(self) -> None:
    """Bet.final_odds に値を設定できる"""
    from domain.models import Bet, BetType

    bet = Bet(
        race_id="20250401110101",
        umaban=1,
        bet_type=BetType.PLACE,
        odds=1.3,
        final_odds=1.5,
        ev_lower_corrected=1.5,
        stake=100.0,
    )
    assert bet.final_odds == 1.5
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_backtest_engine.py::TestBacktestResult::test_bet_final_odds_default -v`
Expected: FAIL (TypeError: __init__() got an unexpected keyword argument 'final_odds')

- [ ] **Step 3: `Bet` に `final_odds` を追加**

`src/domain/models.py` line 153 の後に追加:

```python
class Bet:
    """投票情報"""

    race_id: str
    umaban: int
    bet_type: BetType
    odds: float  # ベット判定に使用したオッズ（発走前 or 確定）
    final_odds: float = 0.0  # 精算用オッズ（確定オッズ）
    ev_lower_corrected: float  # EV下限値（補正済み）
    stake: float  # 投票金額
    result: Optional[float] = None  # 払戻金（確定後）
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `python -m pytest tests/test_backtest_engine.py::TestBacktestResult -v`
Expected: PASS

- [ ] **Step 5: 既存テストで回帰がないことを確認**

Run: `python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 6: コミット**

```bash
git add src/domain/models.py tests/test_backtest_engine.py
git commit -m "feat: Bet データクラスに final_odds フィールドを追加"
```

---

### Task 3: `BacktestEngine` のオッズ取得・精算ロジック変更

**Files:**
- Modify: `src/backtest/engine.py:108-128` (データロード部分)
- Modify: `src/backtest/engine.py:286-287` (settlement 部分)
- Modify: `src/backtest/engine.py:466-479` (`_settle_bet` メソッド)
- Modify: `src/backtest/engine.py:429-464` (`_generate_bets` レガシー)
- Modify: `tests/test_backtest_engine.py` (統合テスト追加)

- [ ] **Step 1: テストを書く**

`tests/test_backtest_engine.py` の `TestBacktestEngine` に追加:

```python
@patch("backtest.engine.load_odds_time_series_range")
@patch("backtest.engine.load_odds_snapshots")
@patch("backtest.engine.load_entries")
@patch("backtest.engine.load_races")
def test_pre_post_odds_used_for_bet_decision(
    self,
    mock_load_races: MagicMock,
    mock_load_entries: MagicMock,
    mock_load_odds: MagicMock,
    mock_load_odds_ts: MagicMock,
    mock_models: MagicMock,
) -> None:
    """ベット判定に発走前オッズが使われる（確定オッズではない）"""
    # レースデータ: 発走時刻付き
    mock_load_races.return_value = pd.DataFrame(
        {"race_id": ["20240101010101"], "race_date": pd.to_datetime("2024-01-01")}
    )
    mock_load_entries.return_value = pd.DataFrame(
        {
            "race_id": ["20240101010101"],
            "umaban": [1],
            "kettonum": [1234],
            "kakuteijyuni": [2],
            "odds": [5.0],
            "ninki": [3],
            "bataijyu": [480],
            "zogen_fugo": [0],
            "zogen_sa": [0],
            "kisyucode": [100],
            "chokyosicode": [200],
        }
    )
    # 確定オッズ: fukuoddslow = 1.1 (EV低い → ベットしないはず)
    mock_load_odds.return_value = pd.DataFrame(
        {
            "race_id": ["20240101010101"],
            "umaban": [1],
            "tanodds": [5.0],
            "fukuoddslow": [1.1],  # 確定オッズ: 低い
            "tanninki": [3],
        }
    )
    # 時系列オッズ: 発走5分前のオッズ = 2.0 (EV高い → ベットするはず)
    mock_load_odds_ts.return_value = pd.DataFrame(
        {
            "race_id": ["20240101010101"],
            "umaban": [1],
            "year": [2024],
            "happyotime": ["01010925"],  # 発走09:30の5分前
            "tanodds": [10.0],
            "fukuoddslow": [2.0],  # 発走前オッズ: 高い
            "tanninki": [3],
        }
    )
    # feat_df は発走前オッズ（2.0）を含む → select_bets がベットを生成
    feat_df = pd.DataFrame(
        {
            "race_id": ["20240101010101"],
            "umaban": [1],
            "surface": ["turf"],
            "kyori": [1200],
            "distance_bin": ["sprint"],
            "popularity_rank": [3],
            "ninki": [3],
            "ev_place": [1.5],  # 閾値以上
            "fukuoddslow": [2.0],  # 発走前オッズ
            "kakuteijyuni": [2],
            "kettonum": [1234],
            "odds": [5.0],
            "bataijyu": [480],
            "jyocd": [6],
            "racenum": [11],
            "grade_code": ["E"],
            "hondai": ["テスト"],
            "bamei": ["テスト馬"],
            "kisyuryakusyo": ["テスト騎手"],
            "track_condition_code": [1],
            "p_place_pred": [0.65],
            "e_return_place_pred": [1.80],
            "hassotime": [930],
            "fukuoddslow_final": [1.1],  # 確定オッズ（精算用）
        }
    )

    with (
        patch("features.feature_engine.FeatureEngine") as mock_fe_cls,
        patch("models.submodel_manager.SubModelManager") as mock_sm_cls,
        patch("features.horse_history_features.HorseHistoryFeatures") as mock_hist_cls,
        patch("features.interaction_features.compute_interaction_features", side_effect=lambda df: df),
        patch("features.jockey_context_features.JockeyContextFeatures") as mock_jc_cls,
        patch("features.trainer_context_features.TrainerContextFeatures") as mock_tc_cls,
        patch("features.jockey_trainer_combo.JockeyTrainerComboFeatures") as mock_jt_cls,
    ):
        mock_fe = MagicMock()
        mock_fe_cls.return_value = mock_fe
        mock_fe.build_all.return_value = feat_df
        mock_sm = MagicMock()
        mock_sm_cls.return_value = mock_sm
        mock_sm.add_distance_band_features.return_value = feat_df
        for cls in [mock_hist_cls, mock_jc_cls, mock_tc_cls, mock_jt_cls]:
            inst = MagicMock()
            cls.return_value = inst
            inst.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        submodel = MagicMock()
        mock_models.submodels["turf"] = submodel
        submodel.market.predict_and_calc_error.return_value = feat_df
        submodel.stage1.add_ability_probs.return_value = feat_df
        submodel.place_ability.predict.return_value = feat_df
        submodel.win.predict_ev.return_value = feat_df
        submodel.ev_corrector.correct_ev.return_value = feat_df
        submodel.place.predict_ev.return_value = feat_df
        submodel.confidence.predict_lower_bound.return_value = (
            feat_df,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )

        from backtest.engine import BacktestEngine

        mock_store = MagicMock()
        engine = BacktestEngine(models=mock_models, store=mock_store)
        result = engine.run("2024-01-01", "2024-12-31")

        # ベットが生成されている（発走前オッズ EV=1.5 >= 閾値）
        assert result.total_bets >= 1
        bet = result.bet_history[0]
        # 精算は確定オッズ (1.1) で計算 → 3着以内でも払戻 = 100 * 1.1 = 110
        assert bet["odds"] == 2.0  # bet.odds = 発走前オッズ


def test_settle_bet_uses_final_odds(self, mock_models: MagicMock) -> None:
    """_settle_bet が final_odds を使用する"""
    from backtest.engine import BacktestEngine
    from domain.models import Bet, BetType

    engine = BacktestEngine(models=mock_models)
    bet = Bet(
        race_id="20240101010101",
        umaban=1,
        bet_type=BetType.PLACE,
        odds=2.0,  # 発走前オッズ
        final_odds=1.1,  # 確定オッズ
        ev_lower_corrected=1.5,
        stake=100.0,
    )
    race_df = pd.DataFrame(
        {"umaban": [1], "kakuteijyuni": [2]}  # 2着 → 複勝的中
    )
    payout = engine._settle_bet(bet, race_df)
    # 精算は final_odds (1.1) で計算: 100 * 1.1 = 110.0
    assert payout == 110.0
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_backtest_engine.py::TestBacktestEngine::test_settle_bet_uses_final_odds -v`
Expected: FAIL (bet.stake * bet.odds = 200.0 ≠ 110.0)

- [ ] **Step 3: `_settle_bet` を変更**

`src/backtest/engine.py:466-479` の `_settle_bet` で `bet.odds` → `bet.final_odds` に変更:

```python
def _settle_bet(self, bet: Bet, race_df: pd.DataFrame) -> float:
    """ベットの結果を判定"""
    horse = race_df[race_df["umaban"] == bet.umaban]
    if horse.empty:
        return 0.0

    finish_pos = int(horse.iloc[0]["kakuteijyuni"])
    settle_odds = bet.final_odds if bet.final_odds > 0 else bet.odds

    if bet.bet_type == BetType.PLACE:
        if 1 <= finish_pos <= 3:
            return float(bet.stake * settle_odds)
    elif bet.bet_type == BetType.WIN:
        if finish_pos == 1:
            return float(bet.stake * settle_odds)
    # ... 以下同様に bet.odds → settle_odds
```

`settle_odds = bet.final_odds if bet.final_odds > 0 else bet.odds` により、
`final_odds` が設定されていない場合（旧来の動作）は `bet.odds` にフォールバック。

- [ ] **Step 4: `_settle_bet` テストを実行して成功を確認**

Run: `python -m pytest tests/test_backtest_engine.py::TestBacktestEngine::test_settle_bet_uses_final_odds -v`
Expected: PASS

- [ ] **Step 5: `BacktestEngine.run()` のオッズ取得を変更**

`src/backtest/engine.py:108-128` を変更:

```python
# 1. データロード
start = test_start.replace("-", "")
end = test_end.replace("-", "")
race_df = load_races(self.store, start, end)
entry_df = load_entries(self.store, start, end)
final_odds_df = load_odds_snapshots(self.store, start, end)  # 確定オッズ（精算用）

if race_df.empty:
    logger.warning(f"No races found in {test_start} ~ {test_end}")
    return BacktestResult(final_bankroll=self.initial_bankroll)

# 2. 特徴量生成
from features.feature_engine import FeatureEngine
from models.submodel_manager import SubModelManager
from db.odds_extractor import extract_pre_post_odds

feat_engine = FeatureEngine()
submodel_mgr = SubModelManager()
odds_ts_df = load_odds_time_series_range(self.store, start, end)

# 発走前オッズの抽出（フォールバック: 確定オッズ）
if not odds_ts_df.empty and "hassotime" in race_df.columns:
    pre_post_odds = extract_pre_post_odds(odds_ts_df, race_df, minutes_before=5)
    if pre_post_odds.empty:
        logger.warning("extract_pre_post_odds returned empty, falling back to final odds")
        pre_post_odds = final_odds_df
    n_pre_post = len(pre_post_odds)
    n_final = len(final_odds_df)
    logger.info("Pre-post odds: %d entries, final odds: %d entries", n_pre_post, n_final)
else:
    pre_post_odds = final_odds_df
    logger.warning("No time-series odds data, using final odds (look-ahead bias)")

# 確定オッズを fukuoddslow_final として発走前オッズにマージ
if not final_odds_df.empty:
    pre_post_odds = pre_post_odds.merge(
        final_odds_df[["race_id", "umaban", "fukuoddslow"]].rename(
            columns={"fukuoddslow": "fukuoddslow_final"}
        ),
        on=["race_id", "umaban"],
        how="left",
    )

feat_df = feat_engine.build_all(
    race_df, entry_df, pre_post_odds, odds_ts_df=odds_ts_df, store=self.store
)
feat_df = submodel_mgr.add_distance_band_features(feat_df)
```

レースループ内（line 258 付近、`select_bets()` の後）に `final_odds` 設定を追加:

```python
# Bet に確定オッズを設定
bets = self._race_predictor.select_bets(result_df, bankroll)
final_odds_map = {}  # fukuoddslow_final から取得
if "fukuoddslow_final" in result_df.columns:
    for _, r in result_df.iterrows():
        key = (r["race_id"], int(r["umaban"]))
        if pd.notna(r.get("fukuoddslow_final")):
            final_odds_map[key] = float(r["fukuoddslow_final"])

updated_bets = []
for bet in bets:
    fo = final_odds_map.get((bet.race_id, bet.umaban), bet.odds)
    updated_bets.append(replace(bet, final_odds=fo))
bets = updated_bets
```

ファイル先頭の import に `from dataclasses import replace` を追加
（既存の `from dataclasses import dataclass, field` に `replace` を追記）。

- [ ] **Step 6: `_generate_bets` レガシーメソッドも更新**

`src/backtest/engine.py:429-464` の `Bet()` 構築に `final_odds` を追加:

```python
bets.append(
    Bet(
        race_id=row["race_id"],
        umaban=int(row["umaban"]),
        bet_type=BetType.PLACE,
        odds=float(row["fukuoddslow"]),
        final_odds=float(row.get("fukuoddslow_final", row["fukuoddslow"])),
        ev_lower_corrected=float(row.get("ev_place", 0)),
        stake=stake,
    )
)
```

- [ ] **Step 7: 統合テストを実行して成功を確認**

Run: `python -m pytest tests/test_backtest_engine.py -v`
Expected: All pass

- [ ] **Step 8: 全テストで回帰がないことを確認**

Run: `python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 9: コミット**

```bash
git add src/backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat: バックテストのオッズを発走前オッズに変更 + 精算分離"
```

---

### Task 4: ペーパートレード `bankroll_after` バグ修正 + import 更新

**Files:**
- Modify: `scripts/run_paper_trading.py:504-541`

- [ ] **Step 1: テストは不要（スクリプト修正のみ）**

※ `run_paper_trading.py` はスクリプトファイルであり、テストなしで修正。
動作確認は手動で `--predict --date YYYYMMDD` を実行して検証。

- [ ] **Step 2: `bankroll_after` を修正**

`scripts/run_paper_trading.py` の510-541行付近で、`bankroll_after` の追跡を修正:

```python
# 変更前:
"bankroll_after": bet.stake,  # reconcileで更新

# 変更後:
"bankroll_after": round(bankroll - bet.stake, 2),
```

`bankroll -= bet.stake` の行（541行付近）は既にあるので、`bankroll_after` の値を
`bankroll - bet.stake` にするだけで正しく追跡される。

- [ ] **Step 3: `_extract_pre_post_odds` の呼び出しを確認**

既に Task 1 で `from db.odds_extractor import extract_pre_post_odds` に変更済み。
呼び出し箇所が `extract_pre_post_odds` になっていることを確認。

- [ ] **Step 4: リント・フォーマット**

Run: `ruff check scripts/run_paper_trading.py && ruff format --check scripts/run_paper_trading.py`
Expected: No errors

- [ ] **Step 5: コミット**

```bash
git add scripts/run_paper_trading.py
git commit -m "fix: ペーパートレード bankroll_after バグ修正"
```

---

### Task 5: フォールバック メトリクス追加

**Files:**
- Modify: `src/backtest/engine.py` (BacktestResult + ログ出力)
- Modify: `src/backtest/engine.py` (最終結果JSON出力)

- [ ] **Step 1: `BacktestResult` にフォールバック メトリクスを追加**

`src/backtest/engine.py` の `BacktestResult` dataclass に:

```python
@dataclass
class BacktestResult:
    """バックテスト結果"""
    # ... 既存フィールド ...
    n_pre_post_odds_bets: int = 0   # 発走前オッズでベットした件数
    n_fallback_odds_bets: int = 0   # フォールバック（確定オッズ）でベットした件数
```

- [ ] **Step 2: レースループ内で メトリクス をカウント**

`BacktestEngine.run()` のレースループ内、ベット処理の後に:

```python
# フォールバック判定（グローバルフラグベース）
# _used_pre_post_odds は Step 5 で設定されるフラグ
if not getattr(self, "_used_pre_post_odds", False):
    n_fallback_odds_bets += len(bets)
else:
    n_pre_post_odds_bets += len(bets)
```

`_used_pre_post_odds` フラグは、Step 5 のオッズ取得部分で設定:

```python
# 発走前オッズの抽出（フォールバック: 確定オッズ）
self._used_pre_post_odds = False
if not odds_ts_df.empty and "hassotime" in race_df.columns:
    pre_post_odds = extract_pre_post_odds(odds_ts_df, race_df, minutes_before=5)
    if not pre_post_odds.empty:
        self._used_pre_post_odds = True
    else:
        logger.warning("extract_pre_post_odds returned empty, falling back to final odds")
        pre_post_odds = final_odds_df
else:
    pre_post_odds = final_odds_df
    logger.warning("No time-series odds data, using final odds (look-ahead bias)")
```

`BacktestResult` 構築時に `n_pre_post_odds_bets` と `n_fallback_odds_bets` を渡す。

- [ ] **Step 3: `summary()` に メトリクス を表示**

`BacktestResult.summary()` に追記:

```python
if self.n_pre_post_odds_bets + self.n_fallback_odds_bets > 0:
    total = self.n_pre_post_odds_bets + self.n_fallback_odds_bets
    fallback_pct = self.n_fallback_odds_bets / total * 100
    lines.append(f"  Odds fallback: {self.n_fallback_odds_bets}/{total} ({fallback_pct:.1f}%)")
```

- [ ] **Step 4: テストで メトリクス を確認**

`tests/test_backtest_engine.py` の既存テスト（`test_engine_populates_enriched_fields`）で
`result.n_pre_post_odds_bets >= 1` をアサーションに追加。

Run: `python -m pytest tests/test_backtest_engine.py -v`
Expected: All pass

- [ ] **Step 5: 全テストで回帰がないことを確認**

Run: `python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 6: コミット**

```bash
git add src/backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat: バックテスト結果にフォールバック メトリクスを追加"
```

---

## 実装後の検証

全タスク完了後、以下を実行して動作確認:

```bash
# 1. 全テスト
python -m pytest tests/ -v

# 2. バックテスト実行（2025テスト、4年学習、ensemble、flat）
python scripts/run_backtest.py \
  --train-start 20210101 --train-end 20241231 \
  --test-start 20250101 --test-end 20251231 \
  --ensemble

# 3. 結果確認
# backtest_result.json の total_bets と bets/race を確認
# フォールバック メトリクスを確認
# ペーパートレードとのベット頻度を比較
```
