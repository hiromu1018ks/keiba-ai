# EveryDB2 マニュアル 12章データフォーマット スクレイピング設計

**作成日:** 2026-03-29
**ステータス:** Approved

## 目的

EveryDB2マニュアル（https://everydb.iwinz.net/edb2_manual/）の12章「データフォーマット」の全ページをPlaywrightでスクレイピングし、体系化されたマークダウンファイルとして `docs/everydb2/` に保存する。

## 背景

- 現在 `docs/everydb2.txt` に7テーブル分のHTMLコピペがあるが、フォーマットが不揃いで検索性が低い
- `docs/everydb2-data-reference.md` はML利用フィールドのマッピングに特化しており、全フィールド定義ではない
- 12章には59のデータフォーマット + コード表(12-3-99)があり、全てを体系化したい

## アプローチ

**Python + Playwright によるスクレイピングスクリプト（アプローチA）**

再現性と正確性を重視。スクリプトとして実装することで、マニュアル更新時の再実行・差分検出が可能。

## スクレイピングスクリプト

### ファイル: `scripts/scrape_everydb2_manual.py`

### 依存関係

- Python 3.11 + playwright
- `pyproject.toml` に `[project.optional-dependencies]` の `scraping = ["playwright>=1.40"]` を追加
- 初回セットアップ: `pip install -e ".[scraping]" && playwright install chromium`
- standalone Playwright（MCP Playwrightではない）

### ページ一覧の取得

`12-3-00-FormatList.html`（データフォーマット一覧）から全リンクを動的に取得。フォールバックとしてハードコード済みリストも保持。

### 前提条件

- EveryDB2マニュアルサイトは認証不要（公開サイト）
- 各ページは直接URLアクセス可能（iframeナビは不要）

### 各ページの処理フロー

1. `https://everydb.iwinz.net/edb2_manual/12-3-XX-NAME.html` にアクセス
2. `<h3>` 要素からページタイトルを抽出（例: `12-3-03. レース詳細（RACE）`）
3. DOMクエリ（`document.querySelectorAll('table tr')`）でテーブル行を取得
   - アクセシビリティスナップショットではなく、直接DOMパースを使用（プレーンHTMLテーブル向け）
4. 8列（No, キー, 項目, フィールド名, 型, サイズ, 初期値, 説明文）をパース
5. RecordSpec は行1の説明文から抽出（例: `RA をセット` → RecordSpec = `RA`）
6. 説明文中のコード表参照を検出:
   - 実際のパターン: `<コード表 NNNN.NAME>参照`（例: `<コード表 2001.競馬場コード>参照`）
   - 正規表現: `コード表\s*(\d{4})\.[^<>]+`
   - 検出時は `[コード表NNNN NAME](codes/CNNNN.md)` に変換
7. マークダウンファイルとして出力
8. 各リクエスト間にデフォルト1秒のディレイ（`--delay` で調整可能）

### コード表（12-3-99-CODE.html）の処理

**構造の注意:** このページは単一テーブルではなく、**20個以上の独立した`<table>`要素**で構成されている。各コードテーブルの列構造はコードごとに異なる:

- ヘッダー行3行（タイトル行、バイト数/値/内容行、サブカラム名行）
- データ行の列数は3〜8個と可変
- 例: コード2001(JyoCD)は場名・場略名など多数列、コード2003(GradeCD)は3列

**パース戦略:**
- 各`<table>`要素のタイトル行からコード番号（例: `2001`）と名称（例: `競馬場コード`）を抽出
- データ行は汎用的に「値と内容」の2列ペアとして抽出
- 列数が多いテーブルは、コード値 + 名称 + 説明 の標準フォーマットに正規化
- 各コード表を `docs/everydb2/codes/CNNNN.md` に個別保存
- コード表のインデックスファイル `docs/everydb2/codes/INDEX.md` も生成

## 出力ファイル構成

```
docs/everydb2/
├── INDEX.md                    # 全テーブルのインデックス（No, ページ名, RecordSpec, フィールド数）
├── 12-3-01-TOKU_RACE.md       # 特別レース
├── 12-3-02-TOKU.md            # 特別登録馬
├── 12-3-03-RACE.md            # レース詳細
├── 12-3-04-UMA_RACE.md        # 馬毎レース情報
├── ...                        # (全59テーブル分)
├── 12-3-59-WOOD_CHIP.md       # ウッドチップ調教
└── codes/                     # コード表
    ├── INDEX.md               # コード表のインデックス
    ├── C2001.md               # 競馬場コード
    ├── C2003.md               # グレードコード
    ├── ...                    # (全コード表)
    └── C9999.md
```

## 各MDファイルのフォーマット

```markdown
# 12-3-03. レース詳細（RACE）

**RecordSpec:** RA
**テーブル名:** n_race

| No | キー | 項目 | フィールド名 | 型 | サイズ | 初期値 | 説明文 |
|---:|:---:|---|---|---|---:|---|---|
| 1 | | レコード種別ID | RecordSpec | varchar | 2 | - | RA をセット |
| 2 | | データ区分 | DataKubun | varchar | 1 | 0 | 1:出走馬名表(木曜) ... |
| ... | | | | | | | |

## コード参照
- `TrackCD` → [コード表2015 トラックコード](codes/C2015.md)
- `SyubetuCD` → [コード表2005 レース種別コード](codes/C2005.md)
```

## INDEX.md フォーマット

```markdown
# EveryDB2 データフォーマット一覧

12章データフォーマットの全テーブル定義。

| No | ページ | テーブル名 | RecordSpec | フィールド数 | 親/子 | 通常/速報 |
|---:|---|---|---|---:|:---:|:---:|
| 01 | [特別レース](12-3-01-TOKU_RACE.md) | - | - | - | | |
| 03 | [レース詳細](12-3-03-RACE.md) | x_RACE | RA | (自動) | 親 | 通常 |
| 04 | [馬毎レース情報](12-3-04-UMA_RACE.md) | x_UMA_RACE | UM | (自動) | 親 | 通常 |
| ... |
```

**注:** テーブル名はインデックスページの `x_` プレフィックス付きの名前をそのまま記録。RecordSpecとフィールド数はスクレイピング時に自動取得。親/子、通常/速報のメタデータもインデックスページから取得。

## エラー処理

- **ページアクセス失敗:** リトライ3回 → スキップしてログに記録、最後にサマリー表示
- **テーブル構造の不一致:** 期待する8列でない行は警告ログ出力、可能な限りパース継続
- **コード表リンクの解決失敗:** `[コード表XXXX](codes/CXXXX.md) ⚠️リンク先未確認` とマーク

## バリデーション

- **行数チェック:** 各ページの行数を自動カウントしINDEX.mdに記録。再実行時に前回と比較 → 差分があれば警告
- **欠損ページ検出:** `12-3-00-FormatList.html` から動的に取得したリストに対して、ハードコードの01〜59と照合
- **フィールド名の重複チェック:** 同一ページ内でフィールド名が重複していないか確認

## 実行オプション

```bash
# 全ページスクレイピング
python scripts/scrape_everydb2_manual.py

# 特定ページのみ
python scripts/scrape_everydb2_manual.py --pages 03 04 05

# コード表のみ
python scripts/scrape_everydb2_manual.py --codes-only

# ドライラン（アクセスのみ、ファイル出力なし）
# 各ページのテーブル名、行数、検出されたコード参照をコンソール出力
python scripts/scrape_everydb2_manual.py --dry-run

# リクエスト間隔の調整（デフォルト1秒）
python scripts/scrape_everydb2_manual.py --delay 2
```

## コーディング規約

- `scripts/` の既存スクリプトに合わせて `argparse` を使用
- ruff: target py311, line-length=100
- mypy: `disallow_untyped_defs = true`（全関数に型アノテーション必須）
- テスト不要（スクレイピングスクリプトは手動実行前提）

## 既存ファイルの扱い

- `docs/everydb2.txt` — 保持（歴史的参照用）
- `docs/everydb2-data-reference.md` — 保持（ML利用フィールドのマッピングとして独立した価値）

## 対象ページ一覧（12章 12-3節）

全59ページ + コード表1ページ:

| No | URL末尾 | 名称 |
|---:|---|---|
| 01 | 12-3-01-TOKU_RACE | 特別レース |
| 02 | 12-3-02-TOKU | 特別登録馬 |
| 03 | 12-3-03-RACE | レース詳細 |
| 04 | 12-3-04-UMA_RACE | 馬毎レース情報 |
| 05 | 12-3-05-HARAI | 払戻 |
| 06 | 12-3-06-HYOSU | 票数 |
| 07 | 12-3-07-HYOSU_TANPUKU | 票数_単複 |
| 08 | 12-3-08-HYOSU_WAKU | 票数_枠連 |
| 09 | 12-3-09-HYOSU_UMARENWIDE | 票数_馬連_ワイド |
| 10 | 12-3-10-HYOSU_UMATAN | 票数_馬単 |
| 11 | 12-3-11-HYOSU_SANREN | 票数_3連複 |
| 12 | 12-3-12-HYOSU2 | 票数2 |
| 13 | 12-3-13-HYOSU_SANRENTAN | 票数_3連単 |
| 14 | 12-3-14-ODDS_TANPUKUWAKU_HEAD | オッズ_単複枠_ヘッダ |
| 15 | 12-3-15-ODDS_TANPUKU | オッズ_単複 |
| 16 | 12-3-16-ODDS_WAKU | オッズ_枠連 |
| 17 | 12-3-17-ODDS_UMAREN_HEAD | オッズ_馬連_ヘッダ |
| 18 | 12-3-18-ODDS_UMAREN | オッズ_馬連 |
| 19 | 12-3-19-ODDS_WIDE_HEAD | オッズ_ワイド_ヘッダ |
| 20 | 12-3-20-ODDS_WIDE | オッズ_ワイド |
| 21 | 12-3-21-ODDS_UMATAN_HEAD | オッズ_馬単_ヘッダ |
| 22 | 12-3-22-ODDS_UMATAN | オッズ_馬単 |
| 23 | 12-3-23-ODDS_SANREN_HEAD | オッズ_3連複_ヘッダ |
| 24 | 12-3-24-ODDS_SANREN | オッズ_3連複 |
| 25 | 12-3-25-ODDS_SANRENTAN_HEAD | オッズ_3連単_ヘッダ |
| 26 | 12-3-26-ODDS_SANRENTAN | オッズ_3連単 |
| 27 | 12-3-27-UMA | 競走馬マスタ |
| 28 | 12-3-28-KISYU | 騎手マスタ |
| 29 | 12-3-29-KISYU_SEISEKI | 騎手マスタ_成績 |
| 30 | 12-3-30-CHOKYO | 調教師マスタ |
| 31 | 12-3-31-CHOKYO_SEISEKI | 調教師マスタ_成績 |
| 32 | 12-3-32-SEISAN | 生産者マスタ |
| 33 | 12-3-33-BANUSI | 馬主マスタ |
| 34 | 12-3-34-HANSYOKU | 繁殖馬マスタ |
| 35 | 12-3-35-SANKU | 産駒マスタ |
| 36 | 12-3-36-RECORD | レコードマスタ |
| 37 | 12-3-37-HANRO | 坂路調教 |
| 38 | 12-3-38-BATAIJYU | 馬体重 |
| 39 | 12-3-39-TENKO_BABA | 天候馬場状態 |
| 40 | 12-3-40-TORIKESI_JYOGAI | 出走取消・競走除外 |
| 41 | 12-3-41-KISYU_CHANGE | 騎手変更 |
| 42 | 12-3-42-HASSOU_JIKOKU_CHANGE | 発走時刻変更 |
| 43 | 12-3-43-COURSE_CHANGE | コース変更 |
| 44 | 12-3-44-MINING | データマイニング予想 |
| 45 | 12-3-45-SCHEDULE | 開催スケジュール |
| 46 | 12-3-46-JODDS_TANPUKUWAKU_HEAD | 時系列オッズ_単複枠_ヘッダ |
| 47 | 12-3-47-JODDS_TANPUKU | 時系列オッズ_単複 |
| 48 | 12-3-48-JODDS_WAKU | 時系列オッズ_枠連 |
| 49 | 12-3-49-JODDS_UMAREN_HEAD | 時系列オッズ_馬連_ヘッダ |
| 50 | 12-3-50-JODDS_UMAREN | 時系列オッズ_馬連 |
| 51 | 12-3-51-SALE | 競走馬市場取引価格 |
| 52 | 12-3-52-BAMEIORIGIN | 馬名の意味由来 |
| 53 | 12-3-53-KEITO | 系統情報 |
| 54 | 12-3-54-COURSE | コース情報 |
| 55 | 12-3-55-TAISENGATA_MINING | 対戦型データマイニング予想 |
| 56 | 12-3-56-JYUSYOSIKI_HEAD | 重勝式_ヘッダ |
| 57 | 12-3-57-JYUSYOSIKI | 重勝式 |
| 58 | 12-3-58-JOGAIBA | 競走馬除外情報 |
| 59 | 12-3-59-WOOD_CHIP | ウッドチップ調教 |
| 99 | 12-3-99-CODE | コード表 |
