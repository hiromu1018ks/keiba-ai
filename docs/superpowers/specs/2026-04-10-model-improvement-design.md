# MLモデル改善デザイン (A群: 即効性のある改善)

日付: 2026-04-10
ステータス: レビュー承認済み

## 背景

現在のバックテスト結果 (4年学習, 2023-2025テスト):

| 指標 | 値 |
|------|------|
| 全体ROI | 123.2% |
| 総利益 | +¥436,700 |
| 2023年 | 112.0% |
| 2024年 | 127.8% |
| 2025年 | 130.3% |

ROI 123%は良好だが、以下の改善余地を特定:

1. **MarketModelのデータリーク** — ランダム分割で未来データ混入
2. **体重変化の表現力不足** — 生値のみで馬個体のコンテキストなし
3. **休養期間特徴量の欠落** — 前走からの日数情報が未使用
4. **バックテストと本番のベット選択不整合** — 検証信頼性の問題

## A群: 実装項目

### A1: MarketModel 時系列分割

**ファイル:** `src/models/market_model.py:58`

**問題:** `np.random.RandomState(42).permutation(n)` でランダム分割している。
時系列データにランダム分割は未来データの混入 (データリーク) を引き起こす。
MarketModelのCVスコアが過大評価されている可能性が高い。

**修正:** 現在の単一80/20分割アーキテクチャに合わせ、**時間ベースの単一分割**を採用する。
`TimeSeriesSplit`は現在のコード構造 (LightGBMの単一train/valid) と互換性がないため使わない。

```python
# Before (line 58)
perm = np.random.RandomState(42).permutation(n)

# After — 時間順の単一分割 (最初80%=学習, 最後20%=検証)
split = int(n * 0.8)
train_idx = np.arange(split)
valid_idx = np.arange(split, n)
```

**前提条件 (必須):** `MarketModel.train()` に渡すDataFrameを `race_date` でソートする。
`TrainingPipelineV5._train_submodel()` 内で `df.sort_values("race_date")` を実行してから
`market.train()` を呼び出す。

**期待効果:** MarketModel精度は下がる可能性があるが、これは「真の精度」。他モデルへのEV補正入力の信頼性が向上する。

---

### A2: 体重変化特徴量の高度化

**ファイル:** `src/features/horse_history_features.py`, `src/features/feature_engine.py`

**現状:** ドメインモデルに `zogen_sa`/`zogen_fugo` は存在するが、**現在はML特徴量として使用されていない**。
実際に使用されている体重関連特徴量は:
- `weight_diff_from_mean` (intra_race_features.py): レース内の平均体重からの差 — レース相対
- `weight_absolute` (horse_history_features.py): 馬の絶対体重

これらは**レースコンテキスト**の情報であって、**馬個体の履歴コンテキスト** (その馬の過去の体重分布に対する相対値) は欠けている。

**exa調査で判明したベストプラクティス:**

日本の競馬AI研究・実践では、体重変化は正規化とカテゴリ化の2段階で扱うのが主流:
- 同じ+10kgでも、普段±5kgの馬 (=異常) と±20kgの馬 (=普通) では意味が全く異なる
- +4~+12kgの「ゴールデンゾーン」は好調サイン (勝率約8%)
- ±14kg以上の大幅変動は異常 (勝率約3%)

**追加特徴量:**

1. `weight_zscore`: 馬個体の体重平均/標準偏差に対する正規化値
   ```
   weight_zscore = (today_weight - horse_weight_mean) / horse_weight_stddev
   ```
   HorseHistoryFeatures で過去の馬体重から平均・標準偏差を計算。

2. `weight_change_zone`: カテゴリ化 (4段階)
   **閾値は絶対kg値 (`zogen_sa`) に基づく。** z-scoreではない。
   理由: ゴールデンゾーン等の閾値はexa調査で「絶対kg」ベースで実証されている。
   `zogen_sa` 列はentries_histに含まれるが、現在 `cols_horse` に含まれていないため追加が必要。
   - `golden`: +4 ~ +12kg (好調サイン)
   - `stable`: -4 ~ +4kg (コンディション維持)
   - `caution`: -14 ~ -4kg, +12 ~ +14kg (注意)
   - `danger`: ±14kg以上 (大幅変動、異常)

   LightGBMは数値のままでも分割可能だが、カテゴリ化で閾値を明示的に与えると安定性が増す。

**実装方針:**
- `HorseHistoryFeatures.cols_horse` に `bataijyu` を追加して過去体重を取得可能にする
- `HorseHistoryFeatures`: 過去出走から馬個体の体重平均・標準偏差を計算
- `FeatureEngine._map_basic_features()`: 正規化・カテゴリ化を適用
- `weight_zscore` と既存の `weight_diff_from_mean` (レース相対) は**直交する情報**。両方保持する

---

### A3: 休養期間特徴量 (days_since_last_race)

**ファイル:** `src/features/horse_history_features.py`

**追加特徴量:**

1. `days_since_last_race`: 前走からの経過日数 (整数)
2. `rest_category`: カテゴリ化
   - `consecutive`: 7日以内 (連闘、疲労リスク)
   - `short`: 8~30日 (通常ローテーション)
   - `medium`: 31~90日 (中休み)
   - `long`: 91~180日 (長期休養)
   - `return`: 181日以上 (復帰戦)

**実装方針:**
- `HorseHistoryFeatures` で前走の `race_date` から経過日数を計算
- 初出走の場合 (n_past = 0 の分岐): `days_since_last_race = NaN`, `rest_category = NaN` を明示的に設定
- LightGBMがネイティブにNaN処理可能

---

### A4: バックテストと本番のベットロジック統一 (切替可能)

**ファイル:** `src/backtest/race_predictor.py`, `src/backtest/engine.py`, `src/betting/orchestrator.py`

**現状の不整合:**

| 項目 | バックテスト (RacePredictor) | 本番 (BettingOrchestrator) |
|------|------|------|
| 投資額 | 常に100円固定 | Kelly基準で可変 |
| EV判定 | `ev_place` (line 123) | `ev_lower_corrected` (line 199) |
| DD制御 | なし | DD Controller (line 204) |

**設計方針:** 設定で切り替え可能にする。結果次第で元に戻せるように。

**実装:**
- `BacktestEngine` に `betting_mode` パラメータを追加
- CLI引数のみで制御 (settings.yamlは使わない — 現在BacktestEngineは設定ファイルを読まない)
- `RacePredictor.select_bets()` にモード分岐を追加:
  - `flat` (デフォルト): 既存の100円固定 + ev_place (変更なし)
  - `kelly`: `ev_lower_corrected` + Fractional Kelly + DD Controller

**kellyモードの依存関係注入:**
- `RacePredictor.__init__()` に `StakeCalculator` と `DrawdownController` をオプション引数で追加
- `kelly` モード時のみこれらを使用:
  ```python
  def __init__(self, models, *, stake_calculator=None, dd_controller=None):
      self.stake_calc = stake_calculator
      self.dd_ctrl = dd_controller
  ```
- `BacktestEngine` の `kelly` モード初期化でインスタンス化して注入:
  ```python
  if betting_mode == "kelly":
      from betting.stake_calculator import StakeCalculator
      from betting.drawdown_controller import DrawdownController
      self.predictor = RacePredictor(
          models,
          stake_calculator=StakeCalculator(fraction=0.5),
          dd_controller=DrawdownController(peak_bankroll=self.initial_bankroll),
      )
  ```

**kellyモードのEV列:** `predict()` が既に出力している `EV_lower_place` を使用。
`ev_lower_corrected` は `BettingOrchestrator` 内部での命名だが、`RacePredictor.predict()` では
`EV_lower_place` として出力済み (line 95、大文字EVに注意)。

**`run_backtest.py` CLI:**
```bash
# 従来 (デフォルト)
python scripts/run_backtest.py --train-start ... --test-start ...

# Kelly版バックテスト
python scripts/run_backtest.py --train-start ... --test-start ... --betting-mode kelly
```

---

## B群: ロードマップ (高影響・中コスト)

A群完了後に別スペックで設計:

| # | 改善 | 影響 | コスト |
|---|------|------|--------|
| B1 | スタックド・アンサンブル (LightGBM + XGBoost + CatBoost) | 高 | 中 |
| B2 | Optuna ハイパーパラメータチューニング | 中〜高 | 中 |
| B3 | 過去3走→5走拡張 + フォームサイクル特徴量 | 中 | 中 |
| B4 | 騎手-調教師コンビ特徴量 | 中 | 低〜中 |

参考: Nguyen et al. (2024) のスタックアンサンブル研究。レベル1: 3モデル独立学習、レベル2: Ridge メタラーナーで統合。

## C群: ロードマップ (中影響・高コスト)

B群完了後に別スペックで設計:

| # | 改善 | 影響 | コスト |
|---|------|------|--------|
| C1 | ペース/シナリオモデリング (先行・差し・追込) | 高 | 高 |
| C2 | レース内ポートフォリオ最適化 (複数馬の相関考慮) | 中 | 高 |
| C3 | 枠順・コース特性特徴量 | 中 | 中 |

## 影響範囲まとめ

### A群の変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `src/models/market_model.py` | 時間ベース単一分割 (race_dateソート前提) |
| `src/pipelines/training_pipeline.py` | MarketModel学習前にrace_dateソート追加 |
| `src/features/horse_history_features.py` | cols_horseにbataijyu追加, 体重統計・休養期間の計算 |
| `src/features/feature_engine.py` | weight_zscore, weight_change_zone, rest_category のマッピング |
| `src/backtest/race_predictor.py` | betting_mode分岐 (flat/kelly), StakeCalculator/DDController注入 |
| `src/backtest/engine.py` | betting_mode パラメータ伝播 |
| `scripts/run_backtest.py` | --betting-mode CLI引数追加 |

### 新特徴量 → モデル FEATURE_COLS マッピング

新特徴量を追加するだけではモデルに使われない。以下のFEATURE_COLSに追加が必要:

| 新特徴量 | 追加先モデル |
|----------|------------|
| `weight_zscore` | `Stage1AbilityModel.FEATURE_COLS`, `PlaceAbilityModel` フィーチャー |
| `weight_change_zone` | `Stage1AbilityModel.FEATURE_COLS` (カテゴリはエンコード必要) |
| `days_since_last_race` | `Stage1AbilityModel.FEATURE_COLS`, `PlaceAbilityModel` フィーチャー |
| `rest_category` | `Stage1AbilityModel.FEATURE_COLS` (カテゴリはエンコード必要) |

MarketModel には追加しない (MarketModelは市場指標ベースであり、馬個体の特徴量は入力しない)。

### 学習への影響

A1のMarketModel修正により、再学習後のバックテスト結果は変動する:
- MarketModel精度は下がる可能性 (リーク排除の代償)
- 体重・休養特徴量の追加で精度向上が期待できる
- 全体としてROIがどう変化するかは再学習+バックテストで確認

### テスト方針

CLAUDE.mdに従いDB不要・mock使用でテストを作成:

1. **A1テスト**: 時間ベース分割が過去インデックスのみを学習に使用することを確認
2. **A2テスト**: weight_zscore, weight_change_zone が期待される型と範囲で生成されることを確認
3. **A3テスト**: days_since_last_race, rest_category の計算が正しいこと (初出走=NaN含む)
4. **A4テスト (flat)**: 従来の100円固定動作と完全一致すること (回帰テスト)
5. **A4テスト (kelly)**: kellyモードがStakeCalculatorとDDControllerを使用することを確認
