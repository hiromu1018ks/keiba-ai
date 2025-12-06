---
description: 当日のレース予測を実行するワークフロー
---

# 当日予測ワークフロー

このワークフローでは、本日開催されるJRAレースの予測を実行し、期待値に基づく推奨買い目を表示します。

## 前提条件

- Python 3.9以上
- 仮想環境 `.venv` がセットアップ済み
- Playwrightがインストール済み (`playwright install chromium`)

---

## 実行手順

### 1. 仮想環境の有効化

```bash
cd /Users/hiromu/Documents/keiba-ai
source .venv/bin/activate
```

### 2. 当日予測の実行

// turbo
```bash
PYTHONPATH=. python src/predict_today.py
```

**オプション**: 特定の日付を指定する場合:
```bash
PYTHONPATH=. python src/predict_today.py --date 20251207
```

---

## 出力の見方

```
--- Race 202509050101 ---
No.  Horse                Prob     Odds   EV     Rec
------------------------------------------------------------
9    エクスプロウド              0.0722   42.4   3.06   ◎ (Strong Buy)
14   クリノセーラーマン            0.0543   32.2   1.75   ◎ (Strong Buy)
```

| 列 | 説明 |
|---|---|
| `No.` | 馬番 |
| `Horse` | 馬名 |
| `Prob` | AIによる勝利確率 |
| `Odds` | 単勝オッズ（リアルタイム取得） |
| `EV` | 期待値 (Prob × Odds) |
| `Rec` | 推奨 (◎=Strong Buy EV>1.5, ○=Buy EV>1.2, △ EV>1.0) |

---

## フィルタ条件

以下の条件に合致しないレースは自動的にスキップされます:

- **馬体重未発表**: 50%以上の馬で馬体重データがない場合（レース1時間前に発表）
- **オッズ100倍超**: 極端な穴馬はフィルタ対象
- **確率5%未満**: 低確率予測は無視

---

## トラブルシューティング

### 「No races found」と表示される
- 指定日にレースがない、またはNetkeiba側の問題
- ネットワーク接続を確認

### オッズが0.0と表示される
- オッズ発売前のレース（通常レース30分前から発売）
- スクレイパーの待機時間不足（通常は自動で15秒待機）

### タイムアウトエラー
- ネットワークが不安定
- `scraper_playwright.py` の `timeout` を増加

---

## ファイル構成

```
keiba-ai/
├── src/
│   ├── predict_today.py      # メイン予測スクリプト
│   ├── data/
│   │   ├── scraper_playwright.py  # Playwrightスクレイパー
│   │   ├── parser_shutsuba.py     # HTMLパーサー
│   │   └── loader.py              # データローダー
│   ├── features/                  # 特徴量生成
│   └── model/                     # モデル関連
├── models/
│   └── lgbm_model.pkl        # 学習済みモデル
└── data/
    ├── processed/
    │   └── merged.csv        # 履歴データ
    └── html/shutsuba/        # スクレイピング済みHTML
```
