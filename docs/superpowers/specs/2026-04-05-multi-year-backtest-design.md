# マルチ年度バックテスト + HTMLレポート 設計書

**日付:** 2026-04-05
**ステータス:** Approved

## 概要

現在のバックテストは単一年度のみテスト可能。2023/2024/2025年をそれぞれ前3年で学習してバックテストを行うスクリプトを作成する。コンソールに年度別サマリーを表示し、HTMLファイルで視覚的に詳細な結果を確認できるようにする。

## 要件

1. **マルチ年度テスト**: 2023, 2024, 2025年の各テスト年度について、前3年で学習してバックテスト
2. **コンソール出力**: 年度別サマリー + 全体比較表をターミナルに表示
3. **HTMLレポート**: 1ファイル・タブ切り替え形式で視覚的にわかりやすいレポート
4. **詳細ベット情報**: 開催日、競馬場、レース名、馬名、騎手名、着順、予測値と実際の比較、レース結果サマリー

## アプローチ

**アプローチA: 新スクリプト + 既存エンジン再利用**

- 新しいスクリプト `scripts/run_multi_year_backtest.py` が年度ループを管理
- 既存の `BacktestEngine` と `TrainingPipelineV5` はそのまま再利用
- `BacktestEngine` の `bet_history.append()` 部分のみ拡張してリッチなフィールドを追加
- 新しい `MultiYearReportGenerator` クラスを `src/backtest/report.py` に追加

選定理由: bet_history フィールドの拡張は1箇所の変更で済み、既存テストへの影響も最小。

## ファイル構成

### 新規ファイル
- `scripts/run_multi_year_backtest.py` — メインスクリプト（年度ループ + コンソール出力 + レポート生成）
- `src/backtest/templates/multi_year_report.html` — マルチ年度用HTMLテンプレート

### 変更ファイル
- `src/backtest/engine.py` — `bet_history.append()` にフィールド追加
- `src/backtest/report.py` — `MultiYearReportGenerator` クラス追加

## bet_history 拡張フィールド

既存フィールドに加えて以下を追加:

| フィールド | 型 | 説明 | 取得元 |
|---|---|---|---|
| `race_date` | str | 開催日 (YYYY-MM-DD) | race_id から派生 |
| `jyocd` | str | 競馬場コード (01-10) | feat_df の `jyocd` 列 |
| `racenum` | int | レース番号 | feat_df の `racenum` 列 |
| `grade_code` | str | グレードコード (A/B/C/D/E/_) | feat_df の `grade_code` 列 (FeatureEngine派生) |
| `race_name` | str | レース名 | feat_df の `hondai` 列 |
| `bamei` | str | 馬名 | feat_df の `bamei` 列 |
| `kisyu` | str | 騎手名 | feat_df の `kisyuryakusyo` 列 |
| `kakuteijyuni` | int | 確定着順 | feat_df の `kakuteijyuni` 列 |
| `track_condition_code` | int | 馬場状態コード (1=良,2=稍重,3=重,4=不良) | feat_df の `sibababacd`/`dirtbabacd` から計算済み列 |
| `p_place_pred` | float | 複勝確率予測 | result_df (RacePredictor.predict() の出力) |
| `e_return_place_pred` | float | 期待払戻予測 | result_df (RacePredictor.predict() の出力) |
| `top3_finishers` | list[dict] | 上位3着 (下記スキーマ参照) | feat_df を kakuteijyuni でソートして抽出 |

### `top3_finishers` スキーマ

```python
# 各要素の dict 構造:
{
    "umaban": int,          # 馬番
    "bamei": str,           # 馬名
    "kisyuryakusyo": str,   # 騎手名（略称）
    "kakuteijyuni": int,    # 確定着順
}
```

- `feat_df` を `kakuteijyuni` で昇順ソートし、上位3件を抽出
- 出走取消・除外・失格（`kakuteijyuni` が 0 または NaN）の馬は除外
- 3頭に満たない場合は取得できた分のみ格納（空リストもあり得る）

## スクリプト処理フロー

```
for test_year in [2023, 2024, 2025]:
    train_start = f"{test_year - 3}-01-01"
    train_end = f"{test_year - 1}-12-31"
    test_start = f"{test_year}-01-01"
    test_end = f"{test_year}-12-31"

    1. TrainingPipelineV5.run(train_start, train_end)  → models
    2. BacktestEngine(models, store).run(test_start, test_end)  → BacktestResult
    3. 結果を all_results[test_year] に格納
    4. コンソールに年度サマリー表示

5. 全体サマリー表示
6. MultiYearReportGenerator.generate(all_results)  → HTML
```

## コンソール出力

年度ごとに:
- 学習完了時間
- テスト完了時間
- ベット数 / 投資額 / 払戻額 / ROI / 利益 / 最大DD

全体サマリー:
- 総ベット数 / 総投資額 / 総払戻額 / 総利益 / 合計ROI
- 最良年度 / 最悪年度
- レポート出力パス

## HTMLレポート構造

1ファイル、タブ切り替え形式 (JavaScript + CSS display切替)。

### タブ構成

1. **全体サマリー**
   - 年度比較KPIカード (各年のROI + 合計ROI)
   - 年度比較表 (学習期間/ベット数/投資額/払戻額/利益/ROI/的中率/最大DD)
   - 年度別ROI比較棒グラフ (Chart.js)
   - 年度別累積資金推移線グラフ (Chart.js)

2. **2023/2024/2025年 (各1タブ)**
   - 既存 `report.html` と同等の構成:
     - KPIカード (ROI / 的中率 / 利益 / 最大DD / 最終資金)
     - 資金推移チャート + ドローダウン
     - 月次ダッシュボード (ROI棒グラフ + ベット数 + 月次表)
     - 条件分析 (路面×距離帯 / 人気帯 / EV帯)
     - 年度内ベット明細テーブル

3. **ベット明細 (全年度統合)**
   - 年度フィルター (全年度 / 2023 / 2024 / 2025)
   - DataTables でソート・検索・ページング
   - 列: 日付 | 競馬場 | R | レース名 | 馬名 | 騎手 | 人気 | EV | オッズ | ベット額 | 着順 | 払戻 | 利益
   - レース行に上位3着情報を展開可能領域として表示

### 競馬場コード変換 (HTMLテンプレート内)

```javascript
const KEIBAJO_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
    '05': '東京', '06': '中山', '07': '中京', '08': '京都',
    '09': '阪神', '10': '小倉'
};
```

### グレード変換

```javascript
const GRADE_MAP = {
    'A': 'G1', 'B': 'G2', 'C': 'G3', 'D': '重賞',
    'E': '特別', 'F': 'J·G1', 'G': 'J·G2', 'H': 'J·G3', '_': '一般'
};
```

### 馬場状態変換

```javascript
const TRACK_CONDITION_MAP = {
    1: '良', 2: '稍重', 3: '重', 4: '不良'
};
```

## BacktestEngine の変更詳細

`src/backtest/engine.py` の `run()` メソッド内:

1. レースループの開始前に、レース情報（競馬場コード、レース番号等）を `feat_df` から取得
2. `bet_history.append()` 内で追加フィールドを設定
3. `top3_finishers` はレース単位で1回計算し、同じレースの全ベットに付与
4. 追加フィールドの取得に失敗した場合はフォールバック値（空文字/0）を使用
5. 各フィールドの取得は `dict.get()` で安全にアクセス（列が存在しない場合の KeyError を防止）

## MultiYearReportGenerator

`src/backtest/report.py` に追加するクラス:

```python
class MultiYearReportGenerator:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._single_year_gen = BacktestReportGenerator(output_dir)  # 委譲

    def generate(
        self,
        results: dict[int, BacktestResult],  # {year: result}
        metadata: dict[int, dict[str, str]],  # {year: {"train_start", "train_end", ...}}
    ) -> Path:
        """マルチ年度HTMLレポートを生成"""
```

**委譲パターン（delegation）を採用。** 理由:
- `BacktestReportGenerator.generate()` と `MultiYearReportGenerator.generate()` はシグネチャが異なるため、継承は不適切
- ヘルパーメソッド（`_compute_monthly_stats`, `_compute_condition_stats`, `_compute_bankroll_series`）は `BacktestReportGenerator` インスタンス経由で再利用
- mypy の `disallow_untyped_defs` でも委譲の方がクリーン

### 年度メタデータの受け渡し

`MultiYearReportGenerator.generate()` には `BacktestResult` だけでなく、学習期間等のメタデータも渡す必要がある。
`BacktestResult` dataclass 自体は変更せず、メソッド引数 `metadata` で別途渡す:

```python
metadata = {
    2023: {"train_start": "2020-01-01", "train_end": "2022-12-31",
           "test_start": "2023-01-01", "test_end": "2023-12-31",
           "train_seconds": 1123, "test_seconds": 689},
    ...
}
```

## コマンドライン引数

```
python scripts/run_multi_year_backtest.py [--years 2023 2024 2025]
```

- `--years`: テスト年度（デフォルト: 2023 2024 2025）
- 学習期間は自動計算（各テスト年の前3年）

## テスト方針

- 新しいスクリプトのテストは不要（スクリプトは薄いラッパー）
- `BacktestEngine` の既存テストは `bet_history` のキーが増える可能性があるため、**実装後に `pytest` を実行して確認**が必要
  - 特に `bet_history[0].keys()` のような検証をしているテストがあれば更新
- `MultiYearReportGenerator` のユニットテストを追加（コンソール出力テストは省略）

## エラーハンドリング

- 学習失敗時: その年度をスキップし、コンソールにエラーを表示して次年度に進む
- テスト失敗時: 同様にスキップ
- 全年度失敗時: エラーメッセージを表示して終了（HTML生成はスキップ）
- 学習のローリングウィンドウ設計: test_year=2024 の学習データ (2021-2023) には test_year=2023 のテストデータが含まれる。これは意図的な設計（過去の予測結果を含むデータで学習するローリングウィンドウ）

## 出力先

- HTMLレポート: `data/backtest/multi_year_report.html`
- JSON結果: `data/backtest/multi_year_result.json`
- bet_history: `data/backtest/multi_year_bet_history.json`
