# Milestone v1.1 — ROI Advanced Model Summary

**Generated:** 2026-05-04
**Purpose:** Team onboarding and project review

---

## 1. Project Overview

競馬AI予測システム v5.5 の単勝モデル改善プロジェクト。既存の LightGBM 2段階モデル (P(hit) x E(odds|hit)) をベースに、アンサンブル・オッズ活用・特徴量改良の3本柱で単勝ベッティングのバックテスト ROI を 100% 超えに引き上げる。

**Core Value:** 単勝モデルのバックテスト ROI を 100% 超えにすること。

**技術スタック:** Python 3.11, LightGBM, XGBoost, CatBoost, scikit-learn, pandas, pyarrow, Optuna, MLflow, PostgreSQL (EveryDB2/JRA-VAN DataLab)

**データパイプライン:** EveryDB2 外部テーブル -> PostgreSQL (ETL入力) -> Parquet ファイル群 -> DataRepository -> ML パイプライン

**v1.1 マイルストーン状況:** 全3フェーズ (Phase 5-7) 完了。バックテスト検証は保留中。

## 2. Architecture & Technical Decisions

- **Decision:** 3モデル GBM スタッキング (LightGBM + XGBoost + CatBoost) + Ridge メタラーナー
  - **Why:** 単一モデルより予測精度向上が期待。3モデルの多様性を Optuna 探索空間分離で強制
  - **Phase:** Phase 7 (Ensemble Enhancement)

- **Decision:** Optuna 探索空間分離 (LGB 浅い木 / XGB 中深さ / CAT 深い木)
  - **Why:** 各モデルが異なる表現空間を学習し、多様性を確保するため
  - **Phase:** Phase 7 (Ensemble Enhancement)

- **Decision:** オッズ乖離を3信号 (odds_to_ability_ratio / deviation_rank / deviation_zscore) で表現
  - **Why:** 絶対値・順序・標準化の3直交信号で LightGBM が非線形閾値を自動学習
  - **Phase:** Phase 6 (Odds Deviation EV)

- **Decision:** Conformal 予測区間の2段階信頼水準 (80%/90%) + conformal_confidence_score
  - **Why:** 80%で高信頼判定、90%で最低基準判定。プロのベッティング運用に倣う
  - **Phase:** Phase 6 (Odds Deviation EV)

- **Decision:** 指数減衰重み付け EMA (halflife=3) でハロンタイム平均を置き換え
  - **Why:** 直近成績に高い重みを付与。金融時系列解析の標準的な選択
  - **Phase:** Phase 5 (Foundation Features)

- **Decision:** ペースフィグアを3サブ特徴量に分割 (corner_stability / closing_power / position_consistency)
  - **Why:** LightGBM が非線形組み合わせを自動学習。単一スコアより情報量が多い
  - **Phase:** Phase 5 (Foundation Features)

- **Decision:** オッズ変動の2次微分 (3点差分) で steam move を検出
  - **Why:** オッズ低下加速 = スマートマネー流入の定量的シグナル
  - **Phase:** Phase 5 (Foundation Features)

- **Decision:** confidence を pair_scores として統合 (4次元コンビネーションではなく)
  - **Why:** 組み合わせ爆発によるサンプル枯渇を回避
  - **Phase:** Phase 6 (Odds Deviation EV)

- **Decision:** K-fold OOF 内で 80/20 分割 + early stopping (stopping_rounds=100)
  - **Why:** K-fold 内の各 fold でも過学習を防止し、OOF 予測品質を向上
  - **Phase:** Phase 7 (Ensemble Enhancement)

## 3. Phases Delivered

| Phase | Name | Status | One-Liner |
|-------|------|--------|-----------|
| 1 | Feature Analysis & Enhancement | Complete | SHAP/gain 特徴量分析 + ノイズ除外 + 単勝特化新特徴量6つ |
| 2 | Win Benter Combination & Calibration | Complete | WinBenterGate + OOF 予測 + Beta/Isotonic キャリブレーション比較 |
| 3 | Selection Gate, Confidence & Betting | Complete | WinSelectionGate + Conformal 信頼性 + JRA 控除率 edge_threshold |
| 4 | Walk-Forward Validation | Complete | WFValidationResult + 過学習検出 + run_wf_validation.py |
| 5 | Foundation Features | Complete | EMA ハロンタイム + クラス調整フォーメトリック + z-score トラジェクトリ + ペース3サブ特徴量 + オッズ変動2特徴量 |
| 6 | Odds Deviation EV | Complete | 乖離特徴量3信号 + Conformal EV 区間 + WinSelectionGate 統合 |
| 7 | Ensemble Enhancement | Complete | Optuna 個別 HP 最適化 + early stopping + feature subset + 多様性検証 |

## 4. Requirements Coverage

### v1.0 Requirements (Shipped)

- [x] ETL pipeline: PostgreSQL (EveryDB2) -> Parquet
- [x] Feature engine: 14モジュール・100+列
- [x] 2-stage model: Win/Place/Wide 各 P(hit) x E(odds|hit)
- [x] Surface submodel: 芝/ダート独立学習
- [x] Backtest engine + OOF prediction + Betting system + MLflow

### v1.1 Requirements

- [x] **TSER-01**: 過去走 EMA 重み付け (halflife=3)
- [x] **TSER-02**: クラス調整済みフォーメトリック
- [x] **TSER-03**: z-score 改善トラジェクトリ
- [x] **PACE-01**: ペースフィグア 3サブ特徴量
- [x] **PACE-02**: 実績ベース ペース適性
- [x] **ODTS-01**: オッズ変動 2次微分 (steam move 検出)
- [x] **ODTS-02**: オッズ変動方向一貫性
- [x] **ODDS-01**: p_market/p_ability 乖離特徴量 (deviation_rank, deviation_zscore)
- [x] **ODDS-02**: BenterGate -> WinSelectionGate パイプライン整合性検証
- [x] **ODDS-03**: Conformal EV 区間 + conformal_confidence_score
- [x] **ENS-01**: Optuna 探索空間分離でモデル間多様性確保
- [x] **ENS-02**: バリデーションベース early stopping (全モデル)
- [x] **ENS-03**: feature_fraction/colsample_bytree/rsm で特徴量サブセット分割

**Coverage:** 13/13 requirements complete (100%)

### Deferred to v2

- ENS-04: Stage1 AbilityModel Ranker スタッキング (複雑度高)
- ODDS-04: Kelly 最適賭け金 (ベッティング戦略は v1.2 以降)
- ODTS-03/04: Late money intensity / Volume-weighted odds movement (データ可用性未検証)
- PACE-03/04: 投影コーナーポジション / リアルタイムペースシミュレーション

## 5. Key Decisions Log

| ID | Decision | Phase | Rationale |
|----|----------|-------|-----------|
| D-01 | harontimel5_avg を EMA (halflife=3) に置き換え | 5 | 直近成績を強調、全過去走を活用 |
| D-02 | class_adj_formetric = sum(norm_finish * class_level) / sum(class_level) | 5 | 高クラス好走の価値を反映 |
| D-03 | ペースフィグアを3サブ特徴量に分割 | 5 | LightGBM が非線形組み合わせを自動学習 |
| D-04 | actual_pace_fit は脚質ベース条件分岐 | 5 | 宣言脚質と実際の走法乖離を補完 |
| D-05 | odds_acceleration 3点差分 (vel_late - vel_early) | 5 | 金融時系列の2次微分標準 |
| D-06 | odds_direction_consistency EMA 重み付け (halflife=n/4) | 5 | 直近変動ほど高く評価 |
| D-07 | 乖離3信号: ratio + rank + z-score | 6 | 絶対値・順序・標準化の直交性が高い |
| D-08 | conformal_confidence_score = EV_lower_80 * (1 - normalized_width_90) | 6 | EV 下限と区間幅のバランス |
| D-09 | predict_lower_bound() を predict_interval() の wrapper にリファクタ | 6 | 二重保守防止 |
| D-10 | confidence を pair_scores として統合 | 6 | 組み合わせ爆発回避 |
| D-11 | Optuna 探索空間分離 (LGB 浅い木 / XGB 中深さ / CAT 深い木) | 7 | 表現空間の差別化で多様性確保 |
| D-12 | K-fold OOF 内 80/20 分割 + early stopping | 7 | 各 fold での過学習防止 |
| D-13 | 多様性検証: OOF 相関 + importance Spearman 相関の二重チェック | 7 | 真の多様性を評価 |

## 6. Tech Debt & Deferred Items

### Deferred Items (from v1.0 close)

| Category | Item | Status |
|----------|------|--------|
| Validation | WF 検証スクリプトの実際の実行 (PostgreSQL 環境必要、~4時間) | Pending |
| UAT | Human UAT 2項目 (04-HUMAN-UAT.md) | Pending |

### Deferred Items (new in v1.1)

- 実際のバックテストで ROI > 100% を検証 (PostgreSQL 環境必要)
- バックテスト feature importance で新特徴量の実際の寄与を確認
- Optuna チューニングによる学習時間増加 (推定 2-3 倍) の許容確認
- ENS-04 (Stage1 Ranker スタッキング) は ROI 目標未達の場合に検討

### Verification Gaps

- Phase 5: ROADMAP SC-1 の「feature importance 上位位置」はバックテスト実行で確認が必要
- Phase 6: ROADMAP SC-1 の「backtest feature importance 上位」も同様
- 全フェーズ: 実データでの学習・推論が未実行 (mock テストのみ完了)

### Known Concerns

- Base model prediction correlation unknown — 実データで相関を測定する必要あり
- Odds snapshot granularity unverified — sub-10-minute snapshots の存在確認が必要
- テスト実行時間が長い (StackedEnsemble テスト: 3モデル x 3fold の学習を含む)

## 7. Getting Started

- **Run the project:**
  ```bash
  # 環境変数
  export PGPASSWORD=<password>

  # ETL: PostgreSQL -> Parquet
  python scripts/run_etl.py --start 20140101 --end 20231231

  # 学習
  python scripts/run_train.py --start 20200101 --end 20231231

  # バックテスト
  python scripts/run_backtest.py --years 2024 --train-window 4
  ```

- **Key directories:**
  - `src/features/` — 特徴量エンジニアリング (14モジュール)
  - `src/models/` — ML モデル群 (StackedEnsemble, WinTwoStageModel, 等)
  - `src/pipelines/` — 学習・推論パイプライン
  - `src/backtest/` — バックテストエンジン
  - `src/db/` — データアクセス層 (ParquetStore -> DataRepository)

- **Tests:**
  ```bash
  python -m pytest tests/ -v          # 全テスト (DB不要、mock 使用)
  python -m pytest tests/ --cov=src   # カバレッジ付き
  ruff check src/ tests/              # リント
  mypy src/                           # 型チェック
  ```

- **Where to look first:**
  - `src/pipelines/training_pipeline.py` — 学習パイプラインのエントリポイント
  - `src/models/stacked_ensemble.py` — 3モデルスタッキング + Optuna チューニング
  - `src/models/two_stage_return_model.py` — 2段階モデル (P x E) の実装
  - `src/backtest/race_predictor.py` — 推論チェーン (全ステージ呼び出し順序)
  - `src/features/odds_deviation_features.py` — オッズ乖離特徴量
  - `src/models/robust_confidence_estimator.py` — Conformal 予測区間

---

## Stats

- **Timeline:** 2026-05-02 -> 2026-05-04 (v1.0+v1.1 combined)
- **v1.1 Timeline:** 2026-05-03 (1 day, Phase 4-7)
- **Phases:** 7 / 7 complete
- **Plans:** 11 total (2+2+2+1+2+1+1)
- **Commits (since 2026-05-03):** 67
- **Files changed:** 68 (+11,603 / -1,370)
- **Contributors:** hiromu1018ks
- **Test suite:** 1,095+ tests, all passing

---

*v1.1 ROI Advanced Model — Milestone complete, pending backtest verification*
