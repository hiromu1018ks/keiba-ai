# MLモデル改善デザイン (A群: 即効性のある改善)

日付: 2026-04-10
ステータス: 承認済み

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

**修正:**
```python
# Before (line 58)
indices = np.random.RandomState(42).permutation(n)

# After
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)
```

**期待効果:** MarketModel精度は下がる可能性があるが、これは「真の精度」。他モデルへのEV補正入力の信頼性が向上する。

**注意:** 学習データが時系列順にソートされていることが前提。`TrainingPipelineV5` でソート確認が必要。

---

### A2: 体重変化特徴量の高度化

**ファイル:** `src/features/horse_history_features.py`, `src/features/feature_engine.py`

**現状:** `zogen_sa` (体重変化値) と `zogen_fugo` (増減符号) をそのまま特徴量として使用。

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
   - `golden`: +4 ~ +12kg (好調サイン)
   - `stable`: -4 ~ +4kg (コンディション維持)
   - `caution`: -14 ~ -4kg, +12 ~ +14kg (注意)
   - `danger`: ±14kg以上 (大幅変動、異常)

   LightGBMは数値のままでも分割可能だが、カテゴリ化で閾値を明示的に与えると安定性が増す。

**実装方針:**
- `HorseHistoryFeatures`: 過去出走から馬個体の体重平均・標準偏差を計算
- `FeatureEngine._map_basic_features()`: 正規化・カテゴリ化を適用

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
- 初出走の場合はNaN (LightGBMがネイティブ処理)

---

### A4: バックテストと本番のベットロジック統一 (切替可能)

**ファイル:** `src/backtest/race_predictor.py`, `src/backtest/engine.py`

**現状の不整合:**

| 項目 | バックテスト (RacePredictor) | 本番 (BettingOrchestrator) |
|------|------|------|
| 投資額 | 常に100円固定 | Kelly基準で可変 |
| EV判定 | `ev_place` | `ev_lower_corrected` |
| DD制御 | なし | DD Controller |

**設計方針:** 設定で切り替え可能にする。結果次第で元に戻せるように。

**実装:**
- `BacktestEngine` に `betting_mode` パラメータを追加
- `settings.yaml` または CLI引数で指定:
  ```yaml
  betting_mode: "flat"   # 従来の100円固定 (現状維持)
  betting_mode: "kelly"  # 本番と同じFractional Kelly (0.5x)
  ```
- `RacePredictor.select_bets()` にモード分岐を追加:
  - `flat`: 既存の100円固定 + ev_place (変更なし)
  - `kelly`: ev_lower_corrected + Fractional Kelly + DD Controller

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
| `src/models/market_model.py` | TimeSeriesSplit導入 |
| `src/features/horse_history_features.py` | 体重統計・休養期間の計算追加 |
| `src/features/feature_engine.py` | weight_zscore, weight_change_zone, rest_category の追加 |
| `src/backtest/race_predictor.py` | betting_mode分岐 (flat/kelly) |
| `src/backtest/engine.py` | betting_mode パラメータ伝播 |
| `scripts/run_backtest.py` | --betting-mode CLI引数追加 |

### 学習への影響

A1のMarketModel修正により、再学習後のバックテスト結果は変動する:
- MarketModel精度は下がる可能性 (リーク排除の代償)
- 体重・休養特徴量の追加で精度向上が期待できる
- 全体としてROIがどう変化するかは再学習+バックテストで確認
