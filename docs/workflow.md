# 競馬予測AI 運用ワークフロー

本プロジェクトにおけるモデル学習、シミュレーション、およびレース当日の運用手順をまとめました。

> **別PCへの移行手順**: 開発環境を別のPCに移す場合は [移行ガイド (migration_guide.md)](migration_guide.md) を参照してください。

## 1. 学習フェーズ (Training)
過去のレースデータを収集し、予測モデルを構築するフェーズです。

### 1-1. データ収集
netkeibaから過去のレース結果や競走馬データを収集します。

```bash
# 競走馬データの収集（時間がかかります）
python src/data/scrape_horses.py
```
> ※ レース結果データ (`results.csv`) は `data/common/raw_data/` に配置されている前提です。

### 1-2. 特徴量生成・モデル学習
収集したデータから特徴量を作成し、LightGBMモデルを学習させます。予測確率はCalibration（補正）されます。

```bash
# 特徴量の生成とモデルの学習
python src/model/train.py
```
- 学習済みモデルは `models/lgbm_calibrated.pkl` に保存されます。
- 特徴量エンジニアは `models/feature_engineer.pkl` に保存されます。

---

## 2. シミュレーションフェーズ (Simulation)
学習済みモデルを用いて、過去データに対する予測精度や回収率を検証します。

### 2-1. バックテストと予測値生成
時系列に沿った学習・予測（Walk-Forward Validation）を行い、シミュレーション用データを生成します。

```bash
# シミュレーションの実行（予測データの生成）
python src/strategy/simulate.py
```
- 実行後、予測結果が `simulation_predictions.csv` に保存されます。

### 2-2. シナリオ別シミュレーション
生成された予測データを用いて、様々な買い方（戦略）での収支を高速にシミュレーションします。
賭け金、期待値(EV)閾値、オッズ条件などを変更して比較可能です。

```bash
# 多様な戦略でのシミュレーション実行
python -m src.strategy.scenario_simulator
```
- 結果はコンソールに表示されるほか、`simulation_scenarios_result.csv` に保存されます。
- **主な戦略例**:
    - **Standard**: EV > 1.2 で購入
    - **Conservative**: EV > 1.5 で購入
    - **Longshot**: オッズ20倍以上の穴狙い

---

## 3. 当日運用フェーズ (Daily Operation)
レース当日に出馬表を取得し、予測と結果確認を行います。

### 3-1. 当日予測の実行
開催される全レースの出馬表を取得し、推奨馬を判定します。

```bash
# 当日の予測を実行（日付指定も可能: --date 20251207）
python -m src.predict_today
```
- **出力**:
    - ターミナルに推奨馬一覧を表示
    - HTMLレポート: `output/predictions_YYYYMMDD.html`
    - CSVデータ: `output/predictions_YYYYMMDD.csv`
- **推奨基準**:
    - **◎ (Strong Buy)**: 期待値(EV) > 1.5
    - **○ (Buy)**: 期待値(EV) > 1.2
    - **△ (Watch)**: 期待値(EV) > 1.0

### 3-2. 結果照合・評価
レース終了後（または途中経過）、予測結果と実際の結果を照合して収支計算を行います。

```bash
# 結果の照合（当日分）
python -m src.evaluate_today
```
- `predictions_YYYYMMDD.csv` にある「推奨馬（Buy/Strong Buy）」のみを集計対象とします。
- まだ結果が出ていないレースは **SKIP** として集計から除外されます。
- 夕方以降、全レース終了後に実行することで、最終的な「的中率」「回収率」が確認できます。
