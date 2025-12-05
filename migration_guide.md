# PC移行手順ガイド

このプロジェクトを別のPC（Mac推奨）に移行するための手順書です。
Gitを使ってコードを管理し、重たいデータファイルは圧縮して直接転送する方法を採用します。

## 1. 準備するもの

*   **移行元PC（現在のPC）**
*   **移行先PC（新しいPC）**
*   **データ転送手段**（以下のいずれか）
    *   AirDrop（Mac同士なら最速・推奨）
    *   USBメモリ / 外付けHDD
    *   Google Drive / Dropbox などのクラウドストレージ

---

## 2. 旧PC（移行元）での作業

### 2-1. コードの同期（Git/GitHub）

まだGitHub等にプッシュしていない変更がある場合は、コミットしてプッシュしておきます。

```bash
# 変更状況を確認
git status

# 変更があればコミットしてプッシュ
git add .
git commit -m "PC移行前のバックアップ"
git push origin main
```

> **Note:** もしGitHubを使っていない場合は、プロジェクトフォルダ全体を圧縮して送る方法になりますが、`data`フォルダが巨大な場合はコードとデータを分けたほうが扱いやすいです。今回は「コードはGit、データは手動」の前提で進めます。

### 2-2. データの圧縮

Gitで管理されていない（`.gitignore` に含まれている）重要なデータを圧縮してまとめます。
主に `data` フォルダと `models` フォルダ、そして設定ファイル `.env` が対象です。

ターミナルでプロジェクトのルートディレクトリ（`keiba-ai`）に移動し、以下のコマンドを実行してください。

```bash
# 重要なデータをまとめて圧縮 (keiba-data.tar.gz というファイルができます)
# dataフォルダ、modelsフォルダ、.envファイル（あれば）を対象にします
tar -czvf keiba-data.tar.gz data models .env
```

※ `tar: .env: Cannot stat: No such file or directory` というエラーが出ても、`.env`ファイルがないだけなので気にせず進めてください。

### 2-3. データの送信

作成された `keiba-data.tar.gz` を、AirDropやUSBメモリを使って**新PC**へ送ります。

---

## 3. 新PC（移行先）での作業

### 3-1. 環境構築

まず、PythonとGitがインストールされているか確認してください。

```bash
python3 --version
git --version
```

まだの場合はインストールしてください（Homebrewを使うのが便利です）。

### 3-2. コードのダウンロード

適当な作業フォルダ（`Documents`など）で、GitHubからコードをダウンロード（クローン）します。

```bash
cd ~/Documents
git clone <あなたのリポジトリのURL> keiba-ai
cd keiba-ai
```

### 3-3. データの配置と解凍

先ほど旧PCから送った `keiba-data.tar.gz` を、この `keiba-ai` フォルダの中に置きます。
その後、以下のコマンドで解凍します。

```bash
# データを解凍
tar -xzvf keiba-data.tar.gz
```

これで `data/` フォルダなどが元の場所に復元されます。
解凍が終わったら、圧縮ファイルは削除して構いません。

```bash
rm keiba-data.tar.gz
```

### 3-4. ライブラリのインストール

新しいPC用に、Pythonの仮想環境を作ってライブラリをインストールします。

```bash
# 仮想環境の作成 (.venv という名前で作ります)
python3 -m venv .venv

# 仮想環境の有効化
source .venv/bin/activate

# ライブラリのインストール
pip install -r requirements.txt
```

---

## 4. 動作確認

移行が正しく行われたか確認します。

### 4-1. データの確認

データファイルが存在するか確認します。

```bash
ls -l data/html/horse/ | head
```

ファイルが表示されればOKです。

### 4-2. プログラムの実行確認

試しにスクリプトを実行してみます。

```bash
# パスを通すために PYTHONPATH=. をつけて実行
PYTHONPATH=. python src/data/scrape_horses.py
```

エラーなく動作（または「既に存在するためスキップ」等のログが出る）すれば移行完了です！
