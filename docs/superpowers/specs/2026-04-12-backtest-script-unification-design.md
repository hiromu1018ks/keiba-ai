# バックテストスクリプト統合 & モデル分離 設計書

日付: 2026-04-12

## 目的

1. バックテスト実行による本番モデル (`data/models/`) の上書きを防止する
2. `run_backtest.py` (単一年度) と `run_multi_year_backtest.py` (複数年度) を統合する
3. レポート生成機能を統一的に扱う

## 現状の問題

- `TrainingPipelineV5.run()` → `_save_models_local()` が `data/models/` を全削除→再保存するため、バックテスト実行で本番モデルが上書きされる
- `run_multi_year_backtest.py` は `--betting-mode`, `--ensemble` オプションがない
- `run_multi_year_backtest.py` は 5年学習固定で柔軟性がない
- レポート生成がスクリプト間で非対称（片方は `--report` フラグ、片方は常に生成）

## 設計

### 1. モデル保存の分離

**変更ファイル:** `src/pipelines/training_pipeline.py`

```python
class TrainingPipelineV5:
    def __init__(self, store: ParquetStore, model_dir: Path | None = None) -> None:
        self.store = store
        self.model_dir = model_dir or Path("data/models")
```

- `_save_models_local()` 内の `Path("data/models")` を `self.model_dir` に置き換え
- MLflow への記録は変更なし（常に実行）
- `run_train.py` (本番) は引数なし → デフォルト `data/models/`
- `run_backtest.py` は `model_dir=Path("data/models-backtest")` を渡す

### 2. スクリプト統合

**削除:** `scripts/run_multi_year_backtest.py`
**変更:** `scripts/run_backtest.py`

#### CLI インターフェース

```
# モード1: 単一年度 (従来互換)
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231

# モード2: マルチ年度 (新機能)
python scripts/run_backtest.py \
  --years 2023 2024 2025 \
  --train-window 4

# 共通オプション
  --betting-mode flat|kelly   (デフォルト: flat)
  --ensemble                  (アンサンブル有効化)
  --report                    (HTMLレポート + JSON 生成)
```

#### 引数体系

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--train-start` / `--train-end` | 必須(モード1) | 学習期間 (YYYYMMDD) |
| `--test-start` / `--test-end` | 必須(モード1) | テスト期間 (YYYYMMDD) |
| `--years` | なし | マルチ年度指定 (モード2) |
| `--train-window` | 4 | マルチ年度の学習年数 |
| `--betting-mode` | flat | flat / kelly |
| `--ensemble` | false | アンサンブル有効化 |
| `--report` | false | HTMLレポート + JSON 生成 |

#### 排他ロジック

- `--years` 指定あり → マルチ年度モード（`--train-start/end`, `--test-start/end` は無視）
- `--years` 指定なし → 単一年度モード（`--train-start/end`, `--test-start/end` 必須）

#### 内部フロー

```
main()
  ├── モード判定 (--years の有無)
  ├── 単一年度モード:
  │     ├── TrainingPipelineV5(store, model_dir=Path("data/models-backtest"))
  │     ├── BacktestEngine(models, store, betting_mode)
  │     ├── engine.run(test_start, test_end)
  │     └── 結果表示 + --report で BacktestReportGenerator 呼び出し
  └── マルチ年度モード:
        ├── for year in years:
        │     ├── train_start = (year - train_window)-01-01
        │     ├── train_end = (year-1)-12-31
        │     ├── TrainingPipelineV5(store, model_dir=Path("data/models-backtest"))
        │     ├── BacktestEngine(models, store, betting_mode)
        │     └── engine.run(test_start, test_end)
        ├── 全体サマリー表示
        └── --report で MultiYearReportGenerator 呼び出し
```

### 3. レポート生成の統合

`BacktestReportGenerator` と `MultiYearReportGenerator` はそのまま残す。
内部で `MultiYearReportGenerator` が `BacktestReportGenerator` を再利用している構造は健全。

スクリプト側で使い分け:

```python
if args.report:
    if is_multi_year:
        gen = MultiYearReportGenerator(output_dir=Path("data/backtest"))
        gen.generate(all_results, all_metadata)
    else:
        gen = BacktestReportGenerator(output_dir=Path("data/backtest"))
        gen.generate(result, result.bet_history, ...)
```

### 4. 出力先

| モード | --report あり | --report なし |
|---|---|---|
| 単一年度 | `data/backtest/backtest_result.json` + HTML | `backtest_result.json` (ルート) |
| マルチ年度 | `data/backtest/multi_year_result.json` + HTML + bet_history | コンソール出力のみ |

### 5. .gitignore 更新

```
data/models-backtest/
```

## 影響範囲

| コンポーネント | 変更 | 影響 |
|---|---|---|
| `TrainingPipelineV5.__init__` | `model_dir` パラメータ追加 | `run_train.py` は変更不要（デフォルト値） |
| `_save_models_local()` | `Path("data/models")` → `self.model_dir` | 保存先の変更のみ |
| `ModelLoader` | 変更なし | 本番は `data/models/` を参照し続ける |
| `run_train.py` | 変更不要 | デフォルト `data/models/` を使用 |
| `run_backtest.py` | 大幅変更 | マルチ年度モード + model_dir 分離 |
| `run_multi_year_backtest.py` | 削除 | 機能は `run_backtest.py` に統合 |
| `.gitignore` | `data/models-backtest/` 追加 | バックテストモデルをGit管理外に |

## テスト計画

- 既存テストへの影響は最小（mock 使用のため）
- `TrainingPipelineV5` の `model_dir` パラメータのテストを追加
- モード判定ロジック（`--years` vs `--train-start/end`）のテストを追加
- バックテスト実行時に `data/models/` が上書きされないことの確認
