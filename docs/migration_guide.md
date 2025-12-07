# 競馬AIプロジェクト 移行ガイド (完全版)

このドキュメントでは、本プロジェクトを別のPC環境へ完全移行するための手順を詳細に解説します。
データサイズが大きいため（約10GB）、Gitだけでなく手動でのデータ移行が必要です。

## 1. 移行元（現在のPC）での作業

まず、Git管理外の重要なデータ（学習データ、モデルなど）をまとめて圧縮します。

### 1-1. データサイズの確認
現在、以下のフォルダが特に容量を食っています。
*   `data/` : **約10GB** (スクレイピングしたHTMLや過去のレース結果)
*   `models/`: **約120MB** (学習済みモデル、特徴量エンジニア)

### 1-2. データの一括圧縮 (アーカイブ作成)
以下のコマンドを実行して、必要なデータを1つのファイル (`keiba_data_backup.tar.gz`) にまとめます。
※ 完了まで数分かかる場合があります。

```bash
# プロジェクトルートディレクトリで実行
tar -czvf keiba_data_backup.tar.gz data models output requirements.txt
```

*   `data`: 必須（これがないと再スクレイピングに数日かかります）
*   `models`: 必須（これがないと再学習に数十分かかります）
*   `output`: 任意（過去の予測結果などを残したい場合）
*   `requirements.txt`: 必須（ライブラリ環境の再現用）

### 1-3. データの取り出し
作成された `keiba_data_backup.tar.gz` (約10GB) と、Git管理されているソースコード一式を新しいPCへ移動します。
ファイルサイズが大きいため、以下のいずれかの方法を推奨します。

*   **USBメモリ / 外付けHDD**: 最も確実で高速です。
*   **クラウドストレージ (Google Drive/Dropbox等)**: ネットワーク環境が良い場合。アップロード/ダウンロードに時間がかかります。
*   **AirDrop**: Mac同士かつ近くにある場合。

## 2. 移行先（新しいPC）での作業

### 2-1. ソースコードの配置
Git経由、またはフォルダコピーでソースコードを展開します。

```bash
# Gitを使う場合
git clone <リポジトリURL> keiba-ai
cd keiba-ai
```

### 2-2. データの展開 (解凍)
移行してきたバックアップファイル (`keiba_data_backup.tar.gz`) を、`keiba-ai` フォルダの直下に置きます。
その後、以下のコマンドで解凍します。

```bash
# 解凍 (既存の同名フォルダがある場合は上書きされます)
tar -xzvf keiba_data_backup.tar.gz
```

これで `data/`, `models/` などのフォルダが元通り配置されます。

### 2-3. 環境構築
Python環境を再現します。

1.  **Pythonのインストール**: 3.10以上をインストールしてください。
2.  **仮想環境の作成**:

    ```bash
    # 仮想環境の作成
    python -m venv .venv
    
    # 有効化 (Mac/Linux)
    source .venv/bin/activate
    # 有効化 (Windows)
    .venv\Scripts\activate
    ```

3.  **ライブラリのインストール**:

    ```bash
    # pipのアップグレード
    pip install --upgrade pip
    
    # バックアップに含まれていた requirements.txt からインストール
    pip install -r requirements.txt
    ```

4.  **Playwrightのセットアップ**:

    ```bash
    playwright install
    ```

## 3. 動作確認

すべてのフォルダとライブラリが揃っていれば、即座に実行可能です。

### 3-1. 予測のテスト
エラーなく動作し、推奨馬が表示されれば成功です。

```bash
python -m src.predict_today
```

### 3-2. シミュレーションのテスト
データ欠損がないか確認するため、短時間のシミュレーションを試すのも良いでしょう。

```bash
# 2024年だけシミュレーションして確認
python -c "from src.strategy.simulate import Backtester; Backtester().run_walk_forward(start_year=2024)"
```

以上で移行作業は完了です。ご不明な点があればお問い合わせください。
