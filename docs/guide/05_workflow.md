# 開発ワークフロー

## 全体フロー

```
ETL → 特徴量生成 → モデル学習 → バックテスト → 実運用
 ①       ②            ③           ④          ⑤
```

各フェーズで使うコマンド・ノートブックを以下にまとめる。

---

## ① ETL: データロード

EveryDB2 からプロジェクトスキーマへデータをコピーする。

```bash
# Python から実行
PGPASSWORD=aa8940aa python -c "
from db.connection import DatabaseConnection
from db.etl import run_full_etl

db = DatabaseConnection()
engine = db.get_engine()

# 既存スキーマを削除して再構築する場合
from sqlalchemy import text
with engine.begin() as conn:
    for schema in ['raw', 'odds_history', 'feature', 'prediction', 'betting']:
        conn.execute(text(f'DROP SCHEMA IF EXISTS {schema} CASCADE'))

# ETL実行 (2015〜2024)
run_full_etl(engine, '20150101', '20241231')
"
```

**注意**: EveryDB2 のオッズはゼロ埋め整数形式 (`"0014"` = 1.4倍)。`_to_odds()` が自動的に ÷10 (tan/fuku) または ÷100 (wide) するので、手動変換は不要。

---

## ② 特徴量生成 → ③ モデル学習

ノートブック `11_holdout_final_evaluation.ipynb` がパイプラインを一括実行する。

```
TrainingPipelineV5.run()
  ├── データロード (DB from SQL)
  ├── FeatureEngine.build_all()      ← 特徴量生成
  ├── SubModelManager.add_distance_band_features()
  ├── surface別にループ (turf / dirt):
  │     ├── MarketModel.train()       ← 市場モデル
  │     ├── AbilityModel.train()      ← Stage1 能力モデル
  │     ├── WinTwoStageModel          ← 単勝2段階
  │     ├── EVCorrectionModel         ← EV補正
  │     ├── PlaceTwoStageModel        ← 複勝2段階
  │     ├── WideTwoStageModel         ← ワイド2段階
  │     └── RobustConfidenceEstimator ← 信頼区間
  ├── RaceQualityScreener.train()     ← レース品質スクリーナー
  └── RegimeDetector.train()          ← レジーム検出
```

---

## ④ バックテスト

`11_holdout_final_evaluation.ipynb` の後半で自動実行される。

```python
engine = BacktestEngine(models, initial_bankroll=100000)
result = engine.run("20220101", "20241231")
```

### バックテストのパラメータ

| パラメータ | 現在値 | 説明 |
|-----------|--------|------|
| ev_threshold | 1.20 (CONSERVATIVE) | ev_place がこれ以上の馬のみベット |
| max_bets_per_race | 2 | 1レースあたり最大ベット数 |
| stake | 100円 (固定) | 1ベットあたりの金額 |

### 合格基準

| 基準 | 閾値 |
|------|------|
| 複勝回収率 | >= 100% |
| 全体回収率 | >= 101% |
| 最大ドローダウン | <= 16% |
| 月次100%超 | >= 22/36ヶ月 |

### 現在の結果 (2026-03-27 時点)

| 項目 | 値 |
|------|-----|
| Total bets | 2,766 |
| ROI | 63.8% |
| Max DD | 100.0% |
| Final bankroll | ¥0 |

**→ モデル改善が必要 (ROI > 100% が目標)**

---

## ノートブック一覧

### 調査・分析 (何度でも実行OK)

| Notebook | 用途 | DB必要 | 実行頻度 |
|----------|------|--------|---------|
| `00_setup.ipynb` | 環境確認・DB接続テスト | Yes | 初回のみ |
| `01_eda.ipynb` | 基礎集計・分布確認 | Yes | 随時 |
| `01b_feature_engineering.ipynb` | 特徴量の統計量・欠損確認 | Yes | 随時 |
| `02_odds_dynamics.ipynb` | オッズ推移・Late Money分析 | Yes | 随時 |

### 実験ノート (何度でも実行OK)

| Notebook | 用途 | DB必要 |
|----------|------|--------|
| `03_market_model_diff_analysis.ipynb` | 市場モデル差分分析 | No (mock) |
| `04_twostage_win_place_ab_test.ipynb` | 2段階モデルABテスト | No (mock) |
| `05_wide_risk_adjusted_score.ipynb` | ワイドリスク調整スコア | No (mock) |
| `06_race_quality_independence.ipynb` | レース品質独立検証 | No (mock) |
| `07_submodel_2split_vs_7split.ipynb` | サブモデル分割比較 | No (mock) |
| `08_dd_rolling_roi_simulation.ipynb` | ドローダウン・ROI推移シミュレーション | No (mock) |
| `09_ev_correction_analysis.ipynb` | EV補正モデル分析 | No (mock) |
| `10_log_error_normalization.ipynb` | ログ誤差正規化検証 | No (mock) |

### ホールドアウト最終評価 (⚠️ 1回のみ)

| Notebook | 用途 | 注意 |
|----------|------|------|
| `11_holdout_final_evaluation.ipynb` | モデル学習 + バックテスト | **データバグ修正時以外は再実行禁止** |

#### 11の再実行が許可されるケース

- ETLのデータ変換バグを修正した (オッズ×10問題など)
- 特徴量の計算ロジックに誤りがあった
- **結果を見てモデルをチューニングする目的では再実行しないこと**

#### 11の再実行が禁止されるケース

- ROIが低いからパラメータを調整して再実行 → **NG** (ホールドアウトデータでチューニング = リーク)
- 新しい特徴量を追加して精度を確認 → **NG** (まず交差検証で検証すべき)

---

## 次のステップ (モデル改善)

現在 ROI 63.8% = **36.2%の損失/ベット**。ROI > 100% を目指す必要がある。

改善アプローチ (ホールドアウトを汚さない方法):

1. **時系列交差検証 (Expanding Window CV)**
   - 訓練: 2015-2018, 検証: 2019 → 訓練: 2015-2019, 検証: 2020 → ...
   - 検証セットでパラメータチューニング
   - 最終確認のみホールドアウト (2022-2024)

2. **特徴量の見直し**
   - 現在の FEATURE_COLS に含まれない情報の追加検討
   - 馬の過去成績、騎手・調教師の統計など

3. **モデルアーキテクチャ**
   - LightGBM パラメータチューニング (num_leaves, learning_rate)
   - モデルアンサンブル

---

## よく使うコマンド

```bash
# テスト実行
python -m pytest tests/ -v

# リント
ruff check src/ tests/

# フォーマット確認
ruff format --check src/ tests/

# 型チェック
mypy src/

# DB内オッズ確認 (オッズが1.1〜999.9の範囲であること)
PGPASSWORD=aa8940aa psql -h localhost -U postgres -d everydb2 \
  -c "SELECT AVG(tan_odds), MIN(tan_odds), MAX(tan_odds) FROM odds_history.odds_snapshots;"
```
