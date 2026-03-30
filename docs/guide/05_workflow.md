# 開発ワークフロー — ROI改善のための実践ガイド

現在のホールドアウト結果: **ROI 63.8%** (2766ベット、36.2%の損失)。目標は ROI > 100%。

---

## 基本ルール

1. **ノートブック11は再実行禁止** (データバグ修正時のみ例外)
2. **改善は時系列交差検証で検証** → 最終確認のみホールドアウト
3. **ソース編集後は必ず `python -m pytest tests/ -v` を実行**

---

## ワークフロー概要

```
ソース編集 → pytest → 分析ノートブック(01-10)で検証 → 効果確認
```

ノートブック11は最後の最後 (データバグ修正時) だけ。

---

## A. 特徴量を追加する場合

### 何を追加すべきか

現在 Stage1 (AbilityModel) は **8特徴量のみ**。馬の能力を全く捉えられていない。
以下は EveryDB2 の `n_uma_race` / `n_uma` テーブルに存在するデータ。

| 追加候補 | 元テーブル.カラム | 期待効果 |
|----------|-------------------|----------|
| 騎手勝率 | `n_uma_race.kisyucode` + 過去レース集計 | 騎手スキルの定量評価 |
| 調教師勝率 | `n_uma_race.chokyosicode` + 過去レース集計 | 調教スキルの定量評価 |
| 馬の過走着順 | `n_uma_race.kakuteijyuni` 直近N走 | フォーム評価 |
| 距離適性 | 過走の距離×着順から算出 | 距離変更時の予測改善 |
| 血統情報 | `n_uma` テーブル | 種牡馬の距離/芝ダート適性 |
| 負担重量 | `n_uma_race.bataijyu` (現weight_diffのみ) | 絶対値の活用 |
| ハロンタイム | `n_uma_race.harontimel3` | 末脚能力の定量評価 |
| レース間隔 | 前走からの日数 | 休み明け/連戦の影響 |

### 手順

#### 1. ETLにカラムを追加

**ファイル**: `src/db/etl.py`

例: `etl_entries()` に `harontime_l3` を追加する場合:

```python
# etl_entries() の df 変換部分に追加
df["harontime_l3"] = df["harontimel3"].apply(_to_float)
```

**確認**: `python -m pytest tests/test_etl.py -v`

#### 2. 特徴量生成モジュールを追加 or 拡張

**新規ファイル**: `src/features/horse_history_features.py`

```python
class HorseHistoryFeatures:
    FEATURE_COLS = ["jockey_win_rate", "trainer_win_rate", "last_3_avg_pos", ...]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # n_uma_race の過去データから騎手勝率等を計算
        ...
```

**既存モジュール**: `src/features/intra_race_features.py` に追加してもよい

#### 3. FeatureEngine に登録

**ファイル**: `src/features/feature_engine.py` の `build_all()` に呼び出しを追加

#### 4. モデルの FEATURE_COLS に追加

**ファイル**:
- `src/models/stage1_ability_model.py` → `FEATURE_COLS` に新カラム名を追加
- `src/models/two_stage_return_model.py` → `FEATURE_COLS` に追加 (必要に応じて)
- `src/models/market_model.py` → 同上

**確認**: `python -m pytest tests/ -v` (importエラーやカラム欠落がないか)

#### 5. 効果確認

**ノートブック**: `01b_feature_engineering.ipynb` を実行して:
- 追加した特徴量の統計量 (mean, std, null率) を確認
- 目的変数 (finish_pos) との相関を確認

**ノートブック**: `01_eda.ipynb` でも基礎確認可能

#### 6. 最終確認 (データバグレベルの変更時のみ)

**ノートブック**: `11_holdout_final_evaluation.ipynb` を再実行
- **許可条件**: ETLに新カラムを追加した = データパイプラインの変更
- **禁止**: モデルパラメータをいじって「精度どうかな？」と確認する目的

---

## B. モデルパラメータを調整する場合

**ホールドアウトを使わずに検証する方法が必要** (現状では交差検証ノートブックが未作成)。

### 必要な作業: 時系列交差検証ノートブックの作成

**新規ノートブック**: `12_time_series_cv.ipynb`

```python
# Expanding Window CV
folds = [
    ("20150101", "20181231", "20190101", "20191231"),  # train → val
    ("20150101", "20191231", "20200101", "20201231"),
    ("20150101", "20201231", "20210101", "20211231"),
]

for train_start, train_end, val_start, val_end in folds:
    pipeline = TrainingPipelineV5(engine)
    models = pipeline.run(train_start, train_end)
    # バックテスト on val期間
    engine_bt = BacktestEngine(models, initial_bankroll=100000)
    result = engine_bt.run(val_start, val_end)
    # ROI, MaxDD を記録
```

### パラメータ調整の対象

**ファイルと変更箇所**:

| モデル | ファイル | パラメータ | 現在値 | 調整範囲 |
|--------|----------|-----------|--------|----------|
| Stage1 | `stage1_ability_model.py` L63-71 | `num_leaves` | 31 | 15-63 |
| Stage1 | 同上 | `learning_rate` | 0.03 | 0.01-0.1 |
| Stage1 | 同上 | `num_boost_round` | 500 | 200-1000 |
| Stage1 | 同上 | `feature_fraction` | 0.7 | 0.5-0.9 |
| Win Hit | `two_stage_return_model.py` | `hit_leaves` | 31 | 15-63 |
| Win Return | 同上 | `return_leaves` | 15 | 7-31 |
| Place Return | 同上 | `return_leaves` | 25 | 15-63 |
| EV P補正 | `ev_correction_model.py` | `num_leaves` | 15 | 7-31 |
| EV E補正 | 同上 | `num_leaves` | 15 | 7-31 |

※ TwoStageConfig は `src/domain/models.py` の `TwoStageConfig` dataclass でデフォルト値を変更可能

### 手順

1. **`12_time_series_cv.ipynb` を作成** (上記のスケルトンをベースに)
2. **パラメータを変更** (該当ファイルの該当箇所)
3. **ノートブック12を実行** → 各foldのROI, MaxDDを比較
4. **全foldで安定してROI改善なら採用**
5. **最終確認はしない** (ホールドアウトは温存)

---

## C. ベッティング戦略を調整する場合

### 閾値の調整

**ファイル**: `src/models/regime_detector.py` の `REGIME_STRATEGIES`

```python
REGIME_STRATEGIES = {
    "AGGRESSIVE":   {"ev_threshold": 1.10, "max_bets_per_race": 3, ...},
    "CONSERVATIVE": {"ev_threshold": 1.30, "max_bets_per_race": 2, ...},
    "COLLAPSED":    {"ev_threshold": 1.50, "max_bets_per_race": 1, ...},
}
```

- `ev_threshold` を上げる → ベット数減少・精度向上・回収率変動
- `max_bets_per_race` を下げる → リスク低下

### ベット金額の調整

**ファイル**: `src/backtest/engine.py` の `_generate_bets()`

現在は固定100円。Kelly基準などの動的 sizing を追加する場合はここを変更。

### 手順

1. **閾値を変更**
2. **`python -m pytest tests/ -v`**
3. **ノートブック12 (時系列CV) で検証**
4. **ノートブック08** (`08_dd_rolling_roi_simulation.ipynb`) でDD/ROI推移をシミュレーション

---

## D. p_ability_place の改善

現在 `p_ability_place = clip(p_ability_win * 3.0, 0, 1)` は粗い近似。
複勝的中率は「3着以内に入る確率」で、単勝の3倍ではない (特に大穴・1番人気でズレる)。

### 手順

**ファイル**: `src/models/stage1_ability_model.py`

`add_ability_probs()` 内の L112 を変更:

```python
# 現在 (粗い近似)
df["p_ability_place"] = np.clip(df["p_ability_win"] * 3.0, 0, 1)

# 改善案1: 3着以内のランクスコア合計
# Ranker の出力で rank <= 3 のスコアを合計

# 改善案2: 別モデル (binary objective, label=finish_pos<=3)
# lambdarank ではなく binary classification で place model を独立訓練
```

**確認**: `python -m pytest tests/ -v` → ノートブック04 (`04_twostage_win_place_ab_test.ipynb`) で A/B 比較

---

## ノートブック使い分け

### 実行順序 (特徴量追加時の例)

```
1. ソース編集 (etl.py, features/, models/)
2. python -m pytest tests/ -v           ← エラーがないか確認
3. 01_eda.ipynb                          ← データの基礎確認
4. 01b_feature_engineering.ipynb         ← 追加特徴量の統計量確認
5. 12_time_series_cv.ipynb (新規作成)    ← ROI改善効果をCVで確認
```

### 各ノートブックの使い方

| Notebook | いつ使う | 何を見る |
|----------|---------|----------|
| `01_eda.ipynb` | DB内容を確認したい時 | カラムの分布・欠損・外れ値 |
| `01b_feature_engineering.ipynb` | 特徴量を追加/変更した後 | 特徴量の統計量・目的変数との相関 |
| `02_odds_dynamics.ipynb` | オッズ時系列を分析したい時 | Late Money効果・オッズ推移パターン |
| `03_market_model_diff_analysis.ipynb` | 市場モデルを改善したい時 | log_errorの分布・予測精度 |
| `04_twostage_win_place_ab_test.ipynb` | Stage1/Stage2を改善したい時 | Win vs Place の予測精度比較 |
| `05_wide_risk_adjusted_score.ipynb` | ワイド戦略を改善したい時 | リスク調整スコアの妥当性 |
| `06_race_quality_independence.ipynb` | レース品質スクリーナーを調整したい時 | スクリーナーの独立性検証 |
| `07_submodel_2split_vs_7split.ipynb` | サブモデル分割を変更したい時 | 2分割 vs 細分化の比較 |
| `08_dd_rolling_roi_simulation.ipynb` | 閾値・資金管理を調整したい時 | DD推移・ROI推移シミュレーション |
| `09_ev_correction_analysis.ipynb` | EV補正モデルを改善したい時 | P補正/E補正の効果分析 |
| `10_log_error_normalization.ipynb` | ログ誤差の分布を確認したい時 | 正規化の妥当性検証 |
| `11_holdout_final_evaluation.ipynb` | **データバグ修正時のみ** | 最終ホールドアウト結果 |
| `12_time_series_cv.ipynb` (**要作成**) | パラメータ変更・モデル改善時 | CV各foldのROI/DD |

---

## 現在の結果 (2026-03-27)

| 項目 | 値 |
|------|-----|
| Total bets | 2,766 |
| ROI | 63.8% |
| Max DD | 100.0% |
| Final bankroll | ¥0 |
| Train期間 | 2015-2021 |
| Test期間 | 2022-2024 (ホールドアウト) |

## 合格基準

| 基準 | 閾値 |
|------|------|
| 複勝回収率 | >= 100% |
| 全体回収率 | >= 101% |
| 最大ドローダウン | <= 16% |
| 月次100%超 | >= 22/36ヶ月 |

---

## よく使うコマンド

```bash
# テスト実行 (ソース編集後は必ず実行)
python -m pytest tests/ -v

# リント
ruff check src/ tests/

# DB内オッズ確認 (オッズが1.1〜999.9の範囲であること)
PGPASSWORD=aa8940aa "C:/Program Files/PostgreSQL/18/bin/psql.exe" \
  -h localhost -U postgres -d everydb2 \
  -c "SELECT AVG(tan_odds), MIN(tan_odds), MAX(tan_odds) FROM odds_history.odds_snapshots;"
```

---

## ETL再実行 (スキーマ全削除→再構築)

データ変換バグを修正した場合のみ実行:

```bash
PGPASSWORD=aa8940aa python -c "
from db.connection import DatabaseConnection
from db.etl import run_full_etl
from sqlalchemy import text

db = DatabaseConnection()
engine = db.get_engine()

with engine.begin() as conn:
    for schema in ['raw', 'odds_history', 'feature', 'prediction', 'betting']:
        conn.execute(text(f'DROP SCHEMA IF EXISTS {schema} CASCADE'))

run_full_etl(engine, '20150101', '20241231')
"
```
