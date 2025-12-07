# 競馬AIプロジェクト 移行ガイド

このドキュメントでは、現在の競馬AIプロジェクトを別のPCに移行し、作業を再開するための手順を説明します。

## 1. 必要な環境

移行先のPCに以下のソフトウェアがインストールされていることを確認してください。

*   **Python 3.10 以上** (推奨: 3.11 または 3.12)
*   **Git** (コード管理用)
*   **Google Chrome** (Playwrightによるスクレイピングで必要になる場合があります)

## 2. ファイルの移行

プロジェクトフォルダ（`keiba-ai`）を丸ごと新しいPCにコピーするのが最も簡単です。
特に以下のフォルダ・ファイルが重要です。

*   `src/`: ソースコード一式
*   `data/`: 過去のレースデータ (`results.csv`) やスクレイピング結果
    *   **重要**: `data/common/raw_data/results.csv` がないと学習ができません。
*   `models/`: 学習済みモデル (`lgbm_calibrated.pkl`) と特徴量エンジニア (`feature_engineer.pkl`)
    *   これを移行すれば、すぐに予測が可能です（再学習不要）。
*   `pyproject.toml` / `requirements.txt`: ライブラリ依存関係

※ `.venv` フォルダは移行しないでください（OSや環境ごとに作り直す必要があります）。

## 3. セットアップ手順 (新しいPCでの操作)

ターミナル（コマンドプロンプト/PowerShell）を開き、移行したプロジェクトフォルダに移動して実行します。

### 3-1. 仮想環境の作成と有効化

```bash
# 仮想環境(.venv)の作成
python -m venv .venv

# 有効化 (Mac/Linux)
source .venv/bin/activate

# 有効化 (Windows)
.venv\Scripts\activate
```

### 3-2. ライブラリのインストール

```bash
# pip自体のアップグレード
pip install --upgrade pip

# 依存ライブラリのインストール
pip install pandas numpy scikit-learn lightgbm optuna playwright beautifulsoup4 lxml tqdm ipykernel matplotlib seaborn
# または requirements.txt があれば
# pip install -r requirements.txt
```

### 3-3. Playwrightのブラウザインストール

スクレイピングに必要なブラウザドライバをインストールします。

```bash
playwright install
```

## 4. 動作確認

環境構築ができたら、以下のコマンドで動作を確認します。

### 4-1. 予測の実行

本日の予測が動くか確認します。モデルファイル (`models/`) が正しく移行されていれば、数分で完了します。

```bash
python -m src.predict_today
```

エラーが出なければ、`output/` フォルダに予測結果（HTML/CSV）が生成されます。

### 4-2. （オプション）モデルの再学習

もしモデルファイルが読み込めない、あるいはゼロから作り直したい場合は、再学習を実行します（時間がかかります）。

```bash
python -m src.model.train
```

## 5. よくあるトラブル

*   **ModuleNotFoundError**: `pip install xxx` で不足しているライブラリを入れてください。
*   **Pickle Load Error**: Pythonのバージョンが大きく異なると、`models/` 内のpklファイルが読み込めないことがあります。その場合は「4-2. モデルの再学習」を行ってください。
*   **Playwright Error**: `playwright install` を忘れていないか確認してください。

以上で移行作業は完了です。
