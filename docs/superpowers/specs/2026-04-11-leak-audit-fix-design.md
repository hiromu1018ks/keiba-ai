# データリーク修正 設計書

**日付**: 2026-04-11
**ステータス**: Draft
**対象**: バックテストとペーパートレードの乖離原因となっているデータリークの全件修正

---

## 1. 背景

バックテスト (2025テスト, ROI 214%) とペーパートレードの間に大幅な乖離が見られる。
根本原因は、パイプラインが EveryDB2 の POST_RACE（レース後確定）情報を特徴量として使用しているため。

### EveryDB2 DataKubun の概念

- DataKubun=1,2: 発走前（出走馬名表・出馬表）
- DataKubun=3-7: レース後（速報成績〜確定成績）

以下の SE/RA カラムは **POST_RACE** であり、発走前には知り得ない:

| カラム | 内容 | パイプライン使用 |
|--------|------|-----------------|
| `KakuteiJyuni` | 確定着順 | 学習ラベルのみ (OK) |
| `Odds` (entries) | 確定単勝オッズ | confirmed_odds に退避済み (OK) |
| `Ninki` | 確定人気 | フォールバックのみ (WARNING) |
| `KyakusituKubun` | 今回レース脚質 | running_style にマッピング (**LEAK**) |
| `HaronTimeL3/L4` | ハロンタイム | 過去走のみ参照 (OK) |
| `Jyuni1c-4c` | コーナー通過順位 | 過去走のみ参照 (OK) |
| `TenkoCD` | 天候 | AMBIGUOUS (直前発表) |
| `SibaBabaCD` | 馬場状態 | AMBIGUOUS (直前発表) |
| `SyussoTosu` | 出走頭数 | AMBIGUOUS (直前確定) |

---

## 2. リーク一覧と修正方針

### C1: JockeyTrainerComboFeatures の max_date リーク [CRITICAL]

**現状**: `jockey_trainer_combo.py:59`
```python
max_date = entry_df["race_date"].max()
hist = hist[hist["race_date"] < max_date]
```
entry_df がテスト期間全体を含むため、期間末の結果が期間最初のレースで参照される。

**修正**: `horse_history_features.py` の searchsorted パターンに統一。
行ごとに `target_race_date` を用いて過去データのみを参照する。

```python
# 修正後: 行ごとの race_date でフィルタ
def compute(self, entry_df: pd.DataFrame) -> pd.DataFrame:
    hist = self._load_history()
    # ... (groupby + transform で行ごとに race_date 以前のデータのみ集計)
```

**実装詳細**: entry_df の各行の `(kisyucode, chokyosicode, race_date)` ごとに、
`hist[hist["race_date"] < row["race_date"]]` を集計。
パフォーマンスのため、hist を `(kisyucode, chokyosicode)` でグループ化し、
各グループ内で `searchsorted(race_date)` でカットオフ。

---

### C2: kyakusitukubun → running_style マッピング [CRITICAL]

**現状**: `feature_engine.py:256-257`
```python
if "kyakusitukubun" in df.columns:
    df["running_style"] = df["kyakusitukubun"].map(_KYAKUSITU_MAP)
```
`kyakusitukubun` (SE No.73) は「今回レース脚質判定」= レース後判定。

**影響範囲** (Spec Review で指摘):
`running_style` は以下のコンポーネントで消費されている:

| ファイル | 使用箇所 | 修正方針 |
|---------|---------|---------|
| `feature_engine.py:256` | `kyakusitukubun` → `running_style` マッピング | **マッピングを削除** |
| `interaction_features.py` | `kyakusitukubun_cd` (過去脚質) のみ使用 | **変更不要** (既に PRE_RACE) |
| `wide_pair_builder.py:43,74` | `horses["running_style"]` を参照 | `kyakusitukubun_cd` (HorseHistoryFeatures 由来) に差し替え |
| `wide_two_stage_model.py:211` | `running_style_combo != 0` でフィルタ | `kyakusitukubun_cd_combo` ベースに変更 |
| `jvlink_fetcher.py:111` | `running_style=int(row["kyakusitukubun"])` | 0 (NaN/未判定) にフォールバック |
| `domain/models.py:119` | `Entry.running_style` フィールド | docstring に POST_RACE 警告を追加 |

**修正**: `feature_engine.py` での `running_style` マッピングを削除。
`wide_pair_builder.py` / `wide_two_stage_model.py` は `kyakusitukubun_cd`
(HorseHistoryFeatures で計算される過去走脚質 = PRE_RACE) を使用するよう変更。

---

### C3: favorite_win_rate の kakuteijyuni 依存 [CRITICAL]

**現状**: `training_pipeline.py:458-461`
```python
favorite_win_rate=(
    "kakuteijyuni",
    lambda x: (x == 1).mean() if len(x) > 0 else 0.0,
),
```

**修正**: `favorite_win_rate` を expanding window で計算。
`compute_hist_features()` と同様に、`expanding().shift(1)` で過去レースの
1番人気勝率を rolling 計算。初期値は 0.3 (一般論)。

```python
# 修正後:
# favorite_win_rate を compute_hist_features 内で expanding で計算
# 入力には kakuteijyuni を使わず、race_feat の topk_hit (past-only) を使用
```

---

### H1: compute_roi_ema() の kakuteijyuni 依存 [HIGH]

**現状**: `odds_dynamics_features.py:218`
```python
df["is_win"] = (df["kakuteijyuni"] == 1).astype(float)
df["roi"] = df["tanodds"] * df["is_win"]
```

**修正**: ROI EMA を**オッズ分布のみ**で計算可能な指標に置き換える:
- `favorite_implied_prob_ema`: 1番人気の implied probability (1/tanodds) の EMA
- `overround_ema`: overround の EMA
- `entropy_ema`: market_entropy の EMA

これらは全て `tanodds` (発走前) のみから計算可能。

**計算順序の注意** (Spec Review で指摘):
現在の `feature_engine.py` の呼び出し順序は:
1. `compute_odds_dynamics()` — overround/entropy がまだない
2. `compute_market_bias()` — overround/entropy を生成

新しい EMA 指標は overround/entropy に依存するため、
`compute_market_bias()` の**後**に計算する必要がある。
具体的には、`compute_roi_ema()` を `compute_market_bias()` の後に移動するか、
`tanodds` から直接 overround/entropy を計算するよう変更する。

---

### H2: compute_flb_slope() の kakuteijyuni 依存 [HIGH]

**現状**: `market_bias_features.py:80`
```python
win = (group["kakuteijyuni"] == 1).astype(float).values
slope = np.polyfit(sorted_log_odds, sorted_win, 1)[0]
```

**修正**: FLB slope を**オッズ分布の歪み度**に置き換える:
- `odds_skewness`: tanodds 分布の歪度 (人気薄のオッズがどれだけ裾を引いているか)
- `implied_prob_hhi`: implied probability の HHI (Herfindahl-Hirschman Index)

これらは tanodds のみから計算可能で、FLB の代理指標として機能する。

---

### H3: favorite_win_rate_rolling の kakuteijyuni 依存 [HIGH]

C3 の修正により自動的に解決（expanding window 版の favorite_win_rate を rolling に適用）。

---

### M1: 確定オッズフォールバック [MEDIUM]

**現状**: `engine.py:148`
```python
pre_post_odds = final_odds_df
logger.warning("No time-series odds data, using final odds (look-ahead bias)")
```

**修正**: フォールバック時は当該レースをスキップする（ベット対象外とする）。
ペーパートレードでも同様にフォールバック時はスキップし、バックテストと方針を統一。

---

### M2: ninki フォールバック [MEDIUM]

**現状**: `feature_engine.py:249-251`
```python
mask = (df["popularity_rank"] == 0) | df["popularity_rank"].isna()
df.loc[mask, "popularity_rank"] = df.loc[mask, "ninki"]
```

**修正**: フォールバック時に警告ログを追加し、統計を取れるようにする。
tanninki が欠損するレースでは当該馬の popularity_rank を NaN とする（フォールバックしない）。

---

### M3: feat_df の POST_RACE 列残存 [MEDIUM]

**現状**: `feat_df` に `kakuteijyuni`, `confirmed_odds` が残っている。

**修正**: `_settle_bet()` 用に別途保持し、`predict()` に渡す前に
POST_RACE 列を drop する。具体的には:

```python
# engine.py run() 内:
_POST_RACE_COLS = ["kakuteijyuni", "confirmed_odds"]

for race_id in race_ids:
    race_df_single = feat_df[feat_df["race_id"] == race_id].copy()
    # ...

    # predict 用に POST_RACE 列を除外
    predict_df = race_df_single.drop(
        columns=[c for c in _POST_RACE_COLS if c in race_df_single.columns],
        errors="ignore",
    )
    result_df = self._race_predictor.predict(predict_df, ...)

    # 精算・bet_history は元の race_df_single で実施
    _top3 = race_df_single.nsmallest(3, "kakuteijyuni")  # OK: 精算用
    bet_result = self._settle_bet(bet, race_df_single)  # OK: 精算用
```

---

### P1: ペーパートレードの確定オッズ使用 [PAPER TRADING]

**現状**: `paper_trading/predictor.py:69`, `scripts/run_paper_trading.py:574`
```python
odds_df = load_odds_snapshots(store, start, end)  # 確定オッズ
feat_engine.build_features(race, entries, odds_df, ...)  # 特徴量に確定オッズを使用
```

**修正**: バックテストと同じ `extract_pre_post_odds()` を追加:
```python
odds_ts_df = load_odds_time_series_range(store, start, end)
pre_post_odds = extract_pre_post_odds(odds_ts_df, race_df, minutes_before=5)
if pre_post_odds.empty:
    # M1 と統一: フォールバック時はスキップ (確定オッズは使用しない)
    logger.warning("No pre-race odds available, skipping race")
    continue
```

---

## 3. RegimeDetector の修正（H1-H3 の統合）

### 現状の問題

1. `FEATURE_COLS` に結果依存列 (`favorite_win_rate`, `flb_slope`, `roi_ema`) が含まれる
2. バックテストループで `detect()` が呼ばれない（CONSERVATIVE 固定）

### 修正方針

#### 3a. FEATURE_COLS の置き換え

| 旧 (結果依存) | 新 (発走前のみ) | 計算方法 |
|---|---|---|
| `favorite_win_rate` | `favorite_implied_prob_rolling` | 1/tanodds(1番人気) の rolling mean |
| `flb_slope` | `odds_skewness_rolling` | tanodds 分布の歪度 rolling mean |
| `favorite_roi_ema` | `overround_rolling` | overround の rolling mean |
| `mid_roi_ema` | `entropy_rolling` | market_entropy の rolling mean |
| `longshot_roi_ema` | `odds_volatility_mean` | (既存・発走前のみ) |
| `market_error_std` | (変更なし) | MarketModel 出力 (発走前) |
| `market_error_mean` | (変更なし) | MarketModel 出力 (発走前) |
| `field_size_mean` | (変更なし) | PRE_RACE |

#### 3b. バックテストループでの detect() 呼び出し

`engine.py` のレースループ内で、過去 N レースのローリング統計を蓄積し、
`regime_detector.detect(recent_stats)` を呼び出してレジームを更新する。

**蓄積する統計スキーマ** (RegimeDetector.FEATURE_COLS に対応):

| 列名 | 型 | 計算元 | タイミング |
|------|----|--------|-----------|
| `market_error_std` | float | MarketModel の signed_log_error_win の std | レース予測後 (PRE_RACE のみ) |
| `market_error_mean` | float | MarketModel の signed_log_error_win の mean | レース予測後 |
| `overround_rolling` | float | overround (tanodds 由来) | 特徴量計算後 |
| `entropy_rolling` | float | market_entropy (tanodds 由来) | 特徴量計算後 |
| `odds_skewness_rolling` | float | tanodds 分布の歪度 | 特徴量計算後 |
| `favorite_implied_prob_rolling` | float | 1/tanodds(1番人気) | 特徴量計算後 |
| `odds_volatility_mean` | float | compute_rolling_volatility | 特徴量計算後 |
| `field_size_mean` | float | field_size | 特徴量計算後 |

```python
# engine.py run() 内:
recent_stats_list: list[dict] = []

for race_id in race_ids:
    race_df_single = feat_df[feat_df["race_id"] == race_id].copy()
    # ... 特徴量計理・predict() 呼び出し ...

    # レジーム判定 (直近200レースの統計を使用)
    recent_stats_df = pd.DataFrame(recent_stats_list[-200:])
    if len(recent_stats_df) >= self.models.regime_detector.cfg.min_samples:
        regime = self.models.regime_detector.detect(recent_stats_df)

    # ... ベット判定・精算 ...

    # 統計を蓄積 (発走前情報のみ)
    row = result_df.iloc[0] if not result_df.empty else {}
    recent_stats_list.append({
        "market_error_std": float(result_df["signed_log_error_win"].std()) if "signed_log_error_win" in result_df.columns else 0.2,
        "market_error_mean": float(result_df["signed_log_error_win"].mean()) if "signed_log_error_win" in result_df.columns else 0.0,
        "overround_rolling": float(row.get("overround", 0.20)),
        "entropy_rolling": float(row.get("market_entropy", 2.0)),
        "odds_skewness_rolling": _calc_odds_skewness(result_df),
        "favorite_implied_prob_rolling": _calc_favorite_implied_prob(result_df),
        "odds_volatility_mean": float(result_df.get("odds_volatility", 0.0).mean()) if "odds_volatility" in result_df.columns else 0.1,
        "field_size_mean": float(row.get("field_size", 14.0)),
    })
```

**ヘルパー関数** (`engine.py` 内に追加):
```python
def _calc_odds_skewness(race_df: pd.DataFrame) -> float:
    """tanodds 分布の歪度 (レース単位)"""
    odds = race_df["odds"].dropna()  # odds 列は tanodds で上書き済み
    return float(odds.skew()) if len(odds) >= 3 else 0.0

def _calc_favorite_implied_prob(race_df: pd.DataFrame) -> float:
    """1番人気の implied probability"""
    if "popularity_rank" not in race_df.columns:
        return 0.3
    fav = race_df[race_df["popularity_rank"] == 1]
    if fav.empty or "odds" not in fav.columns:
        return 0.3
    odds_val = fav["odds"].iloc[0]
    return float(1.0 / odds_val) if pd.notna(odds_val) and odds_val > 0 else 0.3
```

#### 3c. RegimeDetector.train() の修正

教師ラベル生成も結果非依存に変更:
- 現在: `favorite_win_rate` × `(1 - overround)` → market_efficiency
- 修正後: `overround` × `entropy` × `favorite_implied_prob` → market_condition_score

閾値ベースの3状態分類 (AGGRESSIVE/CONSERVATIVE/COLLAPSED) は維持。

---

## 4. 変更ファイル一覧

| ファイル | 変更内容 | 影響度 |
|---------|---------|-------|
| `src/features/jockey_trainer_combo.py` | searchsorted ベースの行ごとフィルタ | 大 |
| `src/features/feature_engine.py` | kyakusitukubun マッピング削除、ninki フォールバック修正、EMA 計算順序修正 | 大 |
| `src/features/odds_dynamics_features.py` | compute_roi_ema をオッズのみ指標に変更 | 大 |
| `src/features/market_bias_features.py` | compute_flb_slope をオッズ歪度に変更 | 大 |
| `src/pipelines/training_pipeline.py` | favorite_win_rate を expanding で再計算、RegimeDetector 特徴量修正 | 大 |
| `src/backtest/engine.py` | POST_RACE 列 drop、RegimeDetector.detect() 組み込み、フォールバック時スキップ | 大 |
| `src/models/regime_detector.py` | FEATURE_COLS 置き換え、train() の教師ラベル修正 | 大 |
| `src/models/wide_pair_builder.py` | `running_style` → `kyakusitukubun_cd` に差し替え | 中 |
| `src/models/wide_two_stage_model.py` | `running_style_combo` フィルタを修正 | 中 |
| `src/ingestion/jvlink_fetcher.py` | `running_style` を 0 にフォールバック | 小 |
| `src/domain/models.py` | `Entry.running_style` に POST_RACE 警告 docstring | 小 |
| `src/paper_trading/predictor.py` | extract_pre_post_odds() 追加、フォールバック時スキップ | 小 |
| `scripts/run_paper_trading.py` | extract_pre_post_odds() 追加、フォールバック時スキップ | 小 |

---

## 5. テスト方針

### 既存テストの修正

既存の mock ベーステストは DB 不要で実行可能。
以下のテストケースを追加:

1. **JockeyTrainerComboFeatures**: `compute()` が行ごとの race_date 以前のデータのみ使用することを確認
2. **FeatureEngine**: `kyakusitukubun` が `running_style` にマッピングされないことを確認
3. **RegimeDetector**: 全 FEATURE_COLS が PRE_RACE カラムのみで構成されていることを確認
4. **BacktestEngine**: predict() に渡す DataFrame に POST_RACE 列が含まれないことを確認
5. **WidePairBuilder**: `running_style` 列が存在しない場合でも正常動作することを確認
6. **compute_roi_ema**: `kakuteijyuni` を使用しない代替指標が正しく計算されることを確認
7. **compute_flb_slope**: `kakuteijyuni` を使用しない代替指標が正しく計算されることを確認

### 既存テストの回帰テスト

以下の既存テストファイルが修正の影響を受けるため、更新が必要:

- `test_jockey_trainer_combo.py` — C1修正後の searchsorted 動作確認
- `test_wide_pair_builder.py` — `running_style` 列不在時の挙動テスト
- `test_regime_detector.py` — 新 FEATURE_COLS での train()/detect() テスト
- `test_market_bias_features.py` — `compute_flb_slope()` 代替指標テスト
- `test_odds_dynamics_features.py` — `compute_roi_ema()` 代替指標テスト

### バックテストでの検証

修正後に以下のバックテストを実行し、結果を比較:

```
# 修正前 (ベースライン)
python scripts/run_backtest.py \
  --train-start 20210101 --train-end 20241231 \
  --test-start 20250101 --test-end 20251231

# 修正後
# 同一条件で実行 → ROI が低下しても real な数字であることを確認
```

### 検証の評価基準

- **有意ベット数**: ベット数が100以下の場合は統計的信頼性が低い
- **月次ROIの一貫性**: 修正後は月ごとのバラツキが減少するはず (月次黒字率の改善)
- **ドローダウン**: リーク修正後はDDが増加する可能性があるが、よりリアルな値

---

## 6. リスク評価

| リスク | 内容 | 軽減策 |
|--------|------|--------|
| ROI の大幅低下 | リークに依存した成績が消失 | これが本来の真の成績。低下は正常 |
| RegimeDetector 精度低下 | 特徴量の変更でレジーム分類が変化 | 新特徴量 (overround, entropy) は市場状態の直接的指標 |
| 過去走データの品質 | HorseHistoryFeatures が正しく searchsorted を使っているか | 既存テストで検証済み |
| ペーパートレードの差分 | extract_pre_post_odds の追加で挙動変化 | バックテストと同一パイプラインに統一 (本来あるべき姿) |
