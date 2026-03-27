# 設定リファレンス

本ドキュメントは `config/` ディレクトリ下の全設定ファイルについて、項目・型・デフォルト値・説明を網羅する。

---

## config/settings.yaml

システム全体の基本設定ファイル。データベース接続、パス、ロギング、特徴量エンジン、遅め買い戦略、サブモデル分割を定義する。

### database

PostgreSQL (EveryDB2) への接続設定。

| 項目 | 型 | デフォルト値 | 説明 |
|------|----|-------------|------|
| `database.host` | string | `"localhost"` | PostgreSQL ホスト名 |
| `database.port` | integer | `5432` | PostgreSQL ポート番号 |
| `database.dbname` | string | `"everydb2"` | データベース名 (EveryDB2) |
| `database.user` | string | `"postgres"` | 接続ユーザー名 |
| `database.password` | string | `""` | 接続パスワード。環境変数 `PGPASSWORD` で上書き可能 |

### paths

データ・モデル・MLflow のディレクトリパス設定。

| 項目 | 型 | デフォルト値 | 説明 |
|------|----|-------------|------|
| `paths.data_dir` | string | `"data"` | データファイル格納ディレクトリ (プロジェクトルート相対) |
| `paths.model_dir` | string | `"models"` | 学習済みモデル格納ディレクトリ (プロジェクトルート相対) |
| `paths.mlflow_tracking_uri` | string | `"file:///mlruns"` | MLflow トラッキングURI。ローカルファイル or リモートサーバURI |

### logging

アプリケーションログの設定。

| 項目 | 型 | デフォルト値 | 説明 |
|------|----|-------------|------|
| `logging.level` | string | `"INFO"` | ログレベル。`DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `logging.format` | string | `"%(asctime)s [%(levelname)s] %(name)s: %(message)s"` | ログ出力フォーマット文字列 |

### feature_engine

特徴量エンジンの動作設定。

| 項目 | 型 | デフォルト値 | 説明 |
|------|----|-------------|------|
| `feature_engine.exclude_steeple` | boolean | `true` | 障害レース (TrackCD 51-59) を学習・予測対象から除外するか |

### late_money

遅め買い (late money) 戦略の閾値設定。レース出走前のオッズ変動に基づく判断基準を定義する。

| 項目 | 型 | デフォルト値 | 説明 |
|------|----|-------------|------|
| `late_money.cancel_threshold` | float | `0.25` | 単勝オッズがこの割合以上急落した場合、買い対象からキャンセル (25%以上下落でキャンセル) |
| `late_money.add_rise_threshold` | float | `0.30` | 単勝オッズがこの割合以上急騰した場合、追加買い候補とする (30%以上上昇で追加) |
| `late_money.cancel_time_minutes` | integer | `3` | キャンセル判定を実行する発走前の分数 (t-3分) |
| `late_money.log_time_minutes` | integer | `2` | ログ記録を実行する発走前の分数 (t-2分) |

### submodel

サブモデル分割設定。芝・ダート別に距離バンドを定義し、各サブモデルの学習・予測単位とする。

#### submodel.surfaces

| 項目 | 型 | デフォルト値 | 説明 |
|------|----|-------------|------|
| `submodel.surfaces` | list[string] | `["turf", "dirt"]` | サブモデル分割対象の馬場種別。芝 (`turf`) とダート (`dirt`) の2分割 |

#### submodel.distance_bands

各馬場種別における距離バンド (距離区分) の定義。

| 項目 | 型 | デフォルト値 | 説明 |
|------|----|-------------|------|
| `submodel.distance_bands.turf.sprint` | list[integer] | `[0, 1400]` | 芝・短距離の範囲 (m) |
| `submodel.distance_bands.turf.mile` | list[integer] | `[1401, 1700]` | 芝・マイルの範囲 (m) |
| `submodel.distance_bands.turf.intermediate` | list[integer] | `[1701, 2100]` | 芝・中距離の範囲 (m) |
| `submodel.distance_bands.turf.long` | list[integer] | `[2101, 9999]` | 芝・長距離の範囲 (m) |
| `submodel.distance_bands.dirt.sprint` | list[integer] | `[0, 1400]` | ダート・短距離の範囲 (m) |
| `submodel.distance_bands.dirt.mile` | list[integer] | `[1401, 1700]` | ダート・マイルの範囲 (m) |
| `submodel.distance_bands.dirt.intermediate` | list[integer] | `[1701, 9999]` | ダート・中距離の範囲 (m)。ダートは芝と異なり long バンドを持たない |

> **備考:** ダートの `intermediate` は上限 `9999` までをカバーするため、ダートには `long` バンドが存在しない。芝は4バンド (sprint/mile/intermediate/long)、ダートは3バンド (sprint/mile/intermediate) となる。

---

## config/backtest_config.yaml

バックテスト・ホールドアウト検証の設定ファイル。Walk-Forward 検証、合格基準、EV補正評価、バリデーション制約を定義する。

### walk_forward

Walk-Forward 交差検証の期間設定。

| 項目 | 型 | デフォルト値 | 説明 |
|------|----|-------------|------|
| `walk_forward.train_years` | integer | `4` | 学習期間の年数 |
| `walk_forward.test_years` | integer | `1` | テスト期間の年数 |
| `walk_forward.step_years` | integer | `1` | ウィンドウをスライドさせる年数 |

> **動作:** 例えば `train_years=4`, `test_years=1`, `step_years=1` の場合、2018-2021学習/2022テスト → 2019-2022学習/2023テスト → 2020-2023学習/2024テスト のようにウィンドウが進行する。

### holdout

ホールドアウト検証 (最終評価用) の期間設定。

| 項目 | 型 | デフォルト値 | 説明 |
|------|----|-------------|------|
| `holdout.start` | string | `"2022-01-01"` | ホールドアウト期間の開始日 (ISO 8601) |
| `holdout.end` | string | `"2024-12-31"` | ホールドアウト期間の終了日 (ISO 8601) |

### pass_criteria

バックテストの合格判定基準。全ての条件を満たした場合に合格とする。

| 項目 | 型 | デフォルト値 | 説明 |
|------|----|-------------|------|
| `pass_criteria.place_roi` | float | `1.00` | 複勝回収率の下限 (100%以上で合格) |
| `pass_criteria.wide_roi` | float | `1.03` | ワイド回収率の下限 (103%以上で合格) |
| `pass_criteria.overall_roi` | float | `1.01` | 全体回収率の下限 (101%以上で合格) |
| `pass_criteria.max_drawdown` | float | `0.16` | 最大ドローダウンの上限 (16%以下で合格) |
| `pass_criteria.min_profitable_months` | integer | `22` | 月次で回収率100%を超える月の最小数 (ホールドアウト期間36ヶ月中22ヶ月以上で合格) |

### ev_correction

EV (Expected Value) 補正の効果測定基準。

| 項目 | 型 | デフォルト値 | 説明 |
|------|----|-------------|------|
| `ev_correction.mae_improvement` | float | `0.10` | EV補正後のMAE (平均絶対誤差) 改善率の下限 (10%以上改善で合格) |
| `ev_correction.mid_range_improvement` | float | `0.15` | 中穴ゾーン (オッズ中間帯) での改善率の下限 (15%以上改善で合格) |

### validation

モデル検証の制約条件。

| 項目 | 型 | デフォルト値 | 説明 |
|------|----|-------------|------|
| `validation.min_submodel_samples` | integer | `20000` | サブモデル学習に必要な最低サンプル数。これ未満のサブモデルは学習・予測から除外される |
| `validation.p_e_correlation_max` | float | `0.30` | P補正モデルとE補正モデルの出力相関の上限 (0.30以下)。上限を超える場合はモデル設計の見直しが必要 |

---

## 環境変数

本システムでは以下の環境変数を利用する。

| 環境変数 | 用途 | 説明 |
|----------|------|------|
| `PGPASSWORD` | データベース認証 | `settings.yaml` の `database.password` を上書きする。CI/CD 環境や本番環境でパスワードをハードコードしないために使用する |

**使用例:**

```bash
# 環境変数でパスワードを設定
export PGPASSWORD="your_password"

# .env ファイルで設定 (python-dotenv 使用時)
PGPASSWORD=your_password
```

---

## MLflow 設定

`paths.mlflow_tracking_uri` で MLflow のトラッキングサーバURIを指定する。

### URI 形式

| 形式 | 例 | 説明 |
|------|----|------|
| ローカルファイル | `"file:///mlruns"` | プロジェクトルート直下の `mlruns/` ディレクトリに保存。開発・個人利用向け |
| リモートサーバ | `"http://localhost:5000"` | MLflow Tracking Server に接続。チーム共有向け |
| Databricks | `"databricks://..."` | Databricks 上の MLflow に接続 |

### パス解決

`mlflow_tracking_uri` の `file://` プロトコルでは、プロジェクトルート (`config/settings.yaml` の3階層上位) を基準にパスが解決される。

```python
# config/settings.yaml 内のパス解決ロジック
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# file:///mlruns → {PROJECT_ROOT}/mlruns
```

---

> **次のドキュメント:** [開発ガイド](04_contributing.md) | **前のドキュメント:** [コード構造](02_code_structure.md) | **ドキュメント一覧:** [README](../../README.md)
