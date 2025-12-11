# 🐴 競馬AI予測システム (Keiba AI)

競馬レースの着順を予測するための機械学習プロジェクトです。  
LightGBMを使用した予測モデルと、期待値(EV)ベースの買い目戦略により、データに基づいた馬券購入をサポートします。

---

## 📁 ディレクトリ構造

```
keiba-ai/
├── common/               # 共通モジュール・共有データ
│   ├── data/             # 共有データファイル
│   └── src/              # 共通ソースコード
│
├── data/                 # データディレクトリ（gitignore対象）
│   ├── raw_data/         # 生データ（netkeibaからスクレイピング）
│   ├── processed/        # 処理済みデータ
│   └── scraped_html/     # スクレイピングしたHTMLファイル
│
├── docs/                 # ドキュメント
│   ├── workflow.md       # 運用ワークフロー詳細
│   ├── daily-prediction.md # 当日予測の手順
│   ├── SCRAPING_GUIDE.md # スクレイピングガイド
│   └── migration_guide.md # 別PCへの移行手順
│
├── experiments/          # 実験用コード
│   └── v2_feature_expansion/ # 特徴量拡張実験
│
├── logs/                 # ログファイル出力先
│
├── models/               # 学習済みモデル（gitignore対象）
│   ├── lgbm_calibrated.pkl  # キャリブレーション済みLightGBMモデル
│   ├── feature_engineer.pkl # 特徴量エンジニア（エンコーダー等）
│   ├── model_features.json  # モデルが使用する特徴量リスト
│   └── best_params.json     # 最適ハイパーパラメータ
│
├── output/               # 出力ファイル
│   ├── predictions_*.html   # 予測結果HTML
│   └── predictions_*.csv    # 予測結果CSV
│
├── reports/              # レポート出力
│
├── src/                  # メインソースコード
│   ├── data/             # データ処理モジュール
│   ├── features/         # 特徴量エンジニアリング
│   ├── model/            # モデル学習・推論
│   ├── strategy/         # 賭け戦略・シミュレーション
│   ├── automation/       # IPAT連携（自動購入）
│   ├── analysis/         # 分析ツール
│   └── utils/            # ユーティリティ
│
├── .gitignore            # Git除外設定
└── requirements.txt      # Python依存パッケージ
```

---

## ⚙️ セットアップ

### 必要環境

- Python 3.9 以上
- Git

### インストール手順

```bash
# 1. リポジトリをクローン
git clone <repository-url>
cd keiba-ai

# 2. 仮想環境を作成して有効化
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# 3. 依存パッケージをインストール
pip install -r requirements.txt

# 4. Playwright（ブラウザ自動化）のセットアップ（スクレイピング用）
playwright install chromium
```

---

## 🚀 使い方

### 1. モデル学習（Training）

過去のレースデータを使用してモデルを学習します。

```bash
# 特徴量の生成とモデルの学習
python src/model/train.py
```

**出力ファイル:**
- `models/lgbm_calibrated.pkl` - 学習済みモデル
- `models/feature_engineer.pkl` - 特徴量エンジニア
- `models/model_features.json` - 使用特徴量リスト

---

### 2. シミュレーション（Backtesting）

学習済みモデルで過去データの予測精度・回収率を検証します。

```bash
# Walk-Forward検証でシミュレーション用データを生成
python -m src.strategy.simulate

# 様々な買い戦略での収支シミュレーション
python -m src.strategy.scenario_simulator
```

**出力ファイル:**
- `simulation_predictions.csv` - 予測結果
- `simulation_scenarios_result.csv` - 戦略別収支

**主な戦略:**
| 戦略名 | 条件 | 説明 |
|--------|------|------|
| Standard | EV > 1.2 | 期待値1.2以上で購入 |
| Conservative | EV > 1.5 | 堅実な期待値重視 |
| Longshot | オッズ20倍以上 | 穴狙い戦略 |

---

### 3. 当日予測（Daily Prediction）

レース当日の出馬表を取得し、推奨馬を予測します。

```bash
# 本日のレース予測
python -m src.predict_today

# 特定日を指定する場合
python -m src.predict_today --date 20251207
```

**推奨馬の記号:**
| 記号 | 条件 | 意味 |
|------|------|------|
| ◎ | EV > 1.5 | Strong Buy（強い買い推奨） |
| ○ | EV > 1.2 | Buy（買い推奨） |
| △ | EV > 1.0 | Watch（注目） |

**出力ファイル:**
- `output/predictions_YYYYMMDD.html` - HTMLレポート
- `output/predictions_YYYYMMDD.csv` - CSVデータ

---

### 4. 結果照合（Evaluation）

レース終了後に予測結果と実際の結果を照合します。

```bash
# 本日分の結果照合
python -m src.evaluate_today

# 特定日を指定する場合
python -m src.evaluate_today --date 20251207
```

**注意:** まだ結果が出ていないレースはSKIPされます。全レース終了後に実行してください。

---

## 🏗️ 主要モジュール

### `src/data/` - データ処理

| ファイル | 説明 |
|----------|------|
| `scraper_playwright.py` | netkeibaからのデータスクレイピング（Playwright使用） |
| `scraper.py` | 基本スクレイピング処理 |
| `scrape_horses.py` | 競走馬データの収集 |
| `parser.py` | レースデータのパース処理 |
| `parser_shutsuba.py` | 出馬表データのパース処理 |
| `loader.py` | データ読み込みユーティリティ |

### `src/features/` - 特徴量エンジニアリング

| ファイル | 説明 |
|----------|------|
| `engineer.py` | 特徴量エンジニア（メイン処理） |
| `history.py` | 過去成績ベースの特徴量生成 |
| `pedigree.py` | 血統情報ベースの特徴量生成 |
| `connections.py` | 騎手・調教師関連の特徴量生成 |

### `src/model/` - モデル

| ファイル | 説明 |
|----------|------|
| `train.py` | LightGBMモデルの学習（Optuna最適化対応） |
| `ensemble.py` | アンサンブルモデル処理 |

### `src/strategy/` - 戦略・シミュレーション

| ファイル | 説明 |
|----------|------|
| `simulate.py` | Walk-Forwardシミュレーション |
| `scenario_simulator.py` | 多様な戦略でのシナリオ検証 |
| `betting.py` | 賭け戦略ロジック（EV計算） |
| `optimization.py` | 戦略パラメータ最適化 |

---

## 📊 予測の仕組み

1. **データ収集**: netkeibaから出馬表・過去レース結果をスクレイピング
2. **特徴量生成**: 過去成績、血統、騎手成績などから150+の特徴量を作成
3. **予測**: LightGBMで各馬の勝利確率を予測
4. **期待値計算**: 予測確率 × オッズ で期待値(EV)を算出
5. **推奨**: EV > 1.0 の馬を買い推奨（値が高いほど強い推奨）

---

## 📝 詳細ドキュメント

詳しい運用手順は `docs/` フォルダ内のドキュメントを参照してください：

- [運用ワークフロー詳細](docs/workflow.md)
- [当日予測の手順](docs/daily-prediction.md)
- [スクレイピングガイド](docs/SCRAPING_GUIDE.md)
- [別PCへの移行手順](docs/migration_guide.md)

---

## ⚠️ 注意事項

- **投資のリスク**: 本システムは参考情報であり、馬券購入による損失は自己責任となります
- **データ利用**: netkeibaのデータ利用は個人利用の範囲に留めてください
- **モデル精度**: 過去データに基づく予測のため、将来の結果を保証するものではありません

---

## 📜 ライセンス

This project is for personal use only.
