# 特徴量 Parquet 出力設計

日付: 2026-04-12

## 目的

バックテストとペーパートレードの成績乖離の原因調査のため、両パイプラインで
計算した特徴量（生特徴量 + 予測値 + ベット判定）を horse-level parquet として
出力し、比較分析を可能にする。

## 現状

- BacktestEngine: 特徴量をメモリ上で計算して破棄。JSON bet_history と CSV 診断ログのみ出力。
- ペーパートレード: 15列の bet-level parquet を出力。特徴量は `_pre.parquet` として
  RaceWatcher 用に一時保存のみ。
- `DiagnosticLogger`: horse-level CSV を出力するが 7 列 (p_place_pred, ev_place 等) のみ。
- 特徴量の列レベルでの比較手段が存在しない。

## アプローチ: 拡張 DiagnosticLogger

既存の `DiagnosticLogger` (`src/backtest/diagnostic_logger.py`) に
特徴量収集機能を追加する。既存の CSV 出力と API は一切変更しない。

### 理由

- DiagnosticLogger は既に horse-level でデータを収集している → 自然な拡張点
- `log_horse()` 呼び出し箇所で `result_df` が利用可能 → 全特徴量にアクセス可能
- 既存の出力を壊さない → 後方互換
- 変更範囲が最小

## 設計

### 1. DiagnosticLogger の拡張

`feature_records: list[dict[str, Any]]` を追加し、
`log_horse_features()` メソッドで収集、`save()` で parquet 出力する。

```python
class DiagnosticLogger:
    def __init__(self) -> None:
        self.race_records: list[RaceDiagnostic] = []
        self.horse_records: list[HorseDiagnostic] = []
        self.feature_records: list[dict[str, Any]] = []  # NEW

    # 既存の log_horse() は変更なし
    def log_horse(self, ...): ...

    # NEW: result_df の1行を収集
    def log_horse_features(self, row: dict[str, Any]) -> None:
        self.feature_records.append(row)

    def save(self, outdir: Path, prefix: str = "diag") -> None:
        # 既存の CSV 出力は変更なし
        ...

        # NEW: 全特徴量 parquet 出力
        if self.feature_records:
            path = outdir / f"{prefix}_horse_features.parquet"
            pd.DataFrame(self.feature_records).to_parquet(path, index=False)
            logger.info(
                "Feature diagnostics saved: %d records -> %s",
                len(self.feature_records),
                path,
            )
```

### 2. 呼び出し側の変更

以下の 3 箇所に `log_horse_features()` を追加する。
全て既存の `log_horse()` 呼び出しループ内で、`result_df` の `iterrows()` が
利用可能な箇所。

#### 2-1. BacktestEngine (`src/backtest/engine.py`)

`run()` 内のレースループ。2箇所（should_bet=False / should_bet=True）の
両方の `log_horse()` ループに追加。

```python
for _, hr in result_df.iterrows():
    diag_logger.log_horse(...)            # 既存 (変更なし)
    diag_logger.log_horse_features(       # NEW
        {k: v for k, v in hr.items()
         if not isinstance(v, (list, dict))}
    )
```

出力先: `data/backtest/{diag_prefix}_horse_features.parquet`

#### 2-2. ペーパートレード predict モード (`scripts/run_paper_trading.py`)

`_run_predict()` 内のレースループ。同じパターン。

出力先: `data/paper_trading/diag_{YYYYMMDD}_horse_features.parquet`

#### 2-3. ペーパートレード diagnose モード (`scripts/run_paper_trading.py`)

`_run_diagnose()` 内のレースループ。同じパターン。

出力先: `data/paper_trading/diag_parquet_{start}_{end}_horse_features.parquet`

### 3. ネスト型列の除外

`top3_finishers` (list[dict]) 等、parquet にシリアライズできない型は
dict 内包で除外:

```python
{k: v for k, v in hr.items() if not isinstance(v, (list, dict))}
```

NumPy 型 (np.int64, np.float64 等) は pandas が parquet 書き出し時に
自動変換するため対応不要。

### 4. 出力スキーマ

各 `*_horse_features.parquet` は **1行 = 1頭** で以下の列群を含む:

| 列群 | 例 | 取得元 |
|------|-----|--------|
| ID列 | `race_id`, `umaban`, `race_date` | feat_df |
| レース条件 | `surface`, `distance_bin`, `track_condition_code`, `grade_code`, `field_size` | Group A |
| オッズ | `fukuoddslow`, `odds_drop_rate_*`, `odds_velocity`, `overround` | Group J, I |
| 能力特徴量 | `blood_surface_wr`, `hist_place_rate_*`, `weight_zscore` 等 | Group D-G |
| 騎手/調教師 | `jockey_wr_*`, `trainer_wr_*`, `jt_combo_wr` | Group N, O |
| インタラクション | `kyakusitu_x_distance`, `weight_x_distance` | Group H |
| 予測値 | `p_place_pred`, `e_return_place_pred`, `ev_place`, `ev_win` | RacePredictor |
| ベット判定 | `is_bet` | DiagnosticLogger |

推定列数: ~100列

### 5. ファイル命名規則

| ソース | ファイルパス |
|--------|-------------|
| バックテスト | `data/backtest/bt_{year}_horse_features.parquet` |
| PT predict | `data/paper_trading/diag_{YYYYMMDD}_horse_features.parquet` |
| PT diagnose | `data/paper_trading/diag_parquet_{start}_{end}_horse_features.parquet` |

### 6. Phase 2: 比較分析（将来実装）

特徴量 parquet が出揃った後の分析ステップ。今回のスコープ外。

1. **分布比較スクリプト** (`scripts/compare_features.py`)
   - 両 parquet を読み込み、列ごとに mean/std/quantile を比較
   - 大きく乖離している特徴量を自動検出 (z-score > 2)

2. **同一レース差分比較**
   - `race_id` で JOIN して列値の差分を確認
   - 特にオッズ系特徴量（発走前オッズのタイミング差等）に注目

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `src/backtest/diagnostic_logger.py` | `feature_records`, `log_horse_features()`, parquet 出力を追加 |
| `src/backtest/engine.py` | `log_horse_features()` 呼び出しを 2箇所に追加 |
| `scripts/run_paper_trading.py` | `log_horse_features()` 呼び出しを predict/diagnose に追加 |

## テスト方針

- 既存テストは mock 使用のため DB 不要。変更なしでパスするはず。
- `DiagnosticLogger` のテストに `log_horse_features()` のケースを追加。
- parquet 出力の列数・型を検証するテストを追加。
