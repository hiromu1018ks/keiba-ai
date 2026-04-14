# PIT 血統特徴量: ルックアヘッドバイアス修正 (2026-04-13)

## 概要

血統特徴量 (`blood_total_wr`, `blood_surface_wr`, `blood_distance_wr`, `blood_prize_log`) のデータソースを
`horses.parquet` (ETL時点の累積値) から `horse_career_stats.parquet` (各レース時点の事前累積値) に変更し、
バックテストのルックアヘッドバイアスを排除した。

## ルックアヘッドの定量的証拠

| 指標 | 値 |
|------|-----|
| bwr_pit (正当) AUC | 0.555 |
| bwr_etl (ルックアヘッド) AUC | 0.684 |
| ルックアヘッド利得 | **+0.129 AUC** |
| bwr_gap (純粋な未来情報) の place 相関 | r=0.380 |
| デビュー馬での bwr_etl と place 相関 | r=0.478 |
| BT テスト馬の 86% が 20戦未満 | 設計書想定 (100戦) の不成立 |

## 実装内容

### 新規ファイル

| ファイル | 説明 |
|--------|------|
| `src/features/horse_career_stats.py` | PIT 累積キャリア統計の事前計算モジュール |
| `scripts/precompute_career_stats.py` | ETL 後の事前計算スクリプト |
| `tests/test_horse_career_stats.py` | 6件のユニットテスト |
| `tests/test_bloodline_features_pit.py` | 5件の PIT 固有テスト |

### 変更ファイル

| ファイル | 変更内容 |
|--------|----------|
| `src/features/bloodline_features.py` | 全面書き換え: horses.parquet → career_stats |
| `src/db/readers.py` | `load_career_stats()` 追加 + jodcs_tanpuku 優先 |
| `tests/test_bloodline_features.py` | 全面書き換え: PIT 版モックに変更 |
| `tests/test_readers.py` | `load_career_stats` テスト + odds 優先度テスト更新 |
| `docs/superpowers/specs/2026-03-29-feature-engineering-design.md` | PIT 時点制約情報を記載 |

### バグ修正 (実行中に発見)

| ファイル | 修正内容 |
|--------|----------|
| `src/features/horse_career_stats.py` | `_classify_surface()` で NaN trackcd が `pd.NA` を返し、`np.where` で TypeError |
| `src/db/odds_extractor.py` | time_series データが `ninki` 列名を使用 → `tanninki`/`ninki` 両対応 |
| `src/db/readers.py` | `jodcs_tanpuku` (新ETL) を time_series (旧ETL) より優先 |

### コミット一覧 (7コミット)

```
4655d84 fix: メインの未コミット変更を同期 (odds_extractor ninki対応 + readers jodcs_tanpuku優先)
9587de9 fix: _classify_surface で NaN trackcd を "other" 扱いに修正
9a51b02 docs: 設計書に PIT 血統特徴量の時点制約情報を追加
89177c3 refactor: _smoothed_wr デッドコードとテストを削除
32ad0c1 feat: BloodlineFeatures を point-in-time キャリア統計に移行
042692d feat: キャリア統計の事前計算スクリプトと load_career_stats リーダーを追加
841d355 feat: point-in-time キャリア統計の事前計算モジュールを追加
```

## バックテスト結果

### 条件

- 学習期間: 2021-01-01 ~ 2024-12-31 (4年学習)
- テスト期間: 2025-01-01 ~ 2025-12-31
- モード: `--ensemble`, flat (固定¥100), JRA のみ

### 修正前 (ルックアヘッドあり)

| 指標 | 値 |
|------|-----|
| ROI | **216.6%** |
| 最大DD | 0.8% |
| 利益 | +¥240,030 |
| ベット数 | 2,401 |

### 修正後 (PIT)

| 指標 | 値 |
|------|-----|
| ROI | **143.7%** |
| 最大DD | 1.8% |
| 利益 | +¥4,940 |
| ベット数 | 113 |

### 変化

| 指標 | 修正前 | 修正後 | 変化 |
|------|--------|--------|------|
| ROI | 216.6% | 143.7% | **-72.9%** |
| 最大DD | 0.8% | 1.8% | +1.0% |
| 利益 | +¥240,030 | +¥4,940 | -¥235,090 |
| BT-PT乖離 | +97.2% | +24.3% | **-72.9pt** |

### 解釈

- ROI の大幅低下は **正常な動作** — AUC +0.129 のルックアヘッド利得が排除された
- 修正前の 216.6% ROI は「未来情報を使って過剰に良い予測ができていた」状態
- BT-PT乖離が +97.2% → +24.3% に縮小、ルックアヘッド排除が成功
- PT ROI (119.4%) と BT ROI (143.7%) の差は +24.3pt。BT の方が高いが、これは
  PT の期間 (4月) が BT 全年 (1-12月) よりも条件が良かった可能性がある

## 注意点

1. **feature definition の変更:** `ba1`(芝直線) → 芝全般 の近似。学習時と推論時で同じ近似値を使うため BT/PT 間の整合性は保たれるが、モデルの再学習が必要。
2. **事前計算が必要:** ETL 実行後に `python scripts/precompute_career_stats.py` を実行して `data/raw/horse_career_stats.parquet` を更新する必要がある。
3. **データ未生成時の動作:** `horse_career_stats.parquet` が存在しない場合、全馬の血統特徴量が NaN になる。
