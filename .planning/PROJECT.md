# keiba-ai — Win Model Improvement

## What This Is

競馬AI予測システム v5.5 — 統計的 horse racing prediction system (単勝/複勝/ワイド)。
LightGBM + XGBoost + CatBoost 3モデルスタッキング、Optuna個別HP最適化、Isotonic EVキャリブレーション + オッズバンド別補正、CQR Conformal EV区間を搭載。
POST_RACE情報漏洩完全排除(3層CI検出)、100+特徴量Tier分類監査基盤、EveryDB2未活用データからの22新特徴量(mining/血統/BMS/record/相対比較)、12ドメイン知識交互作用項、OOF安全ターゲットエンコーディング(血統/騎手/調教師)を追加。
Race-level集約特徴量6 + 市場クロス整合性特徴量5 (Harville公式) で市場独立性を獲得。IC評価フレームワーク (B差分/C直交/E Incremental/Per-race) + Gain-per-Depth診断システム搭載。
3段階ベットフィルター(動的EV_lower/OddsBand/COLLAPSED skip)、レジーム別Kellyサイジング、DD%のみ3段階制御、Optuna TPE 16次元パラメータ最適化 + multi-seed安定性検証、SHA256特徴量凍結manifestを搭載。
MLflow で実験管理。PostgreSQL (EveryDB2/JRA-VAN DataLab) をデータソースとする。

## Core Value

単勝モデルのバックテストROIを100%超えにすること。全ての改善はこの指標で判断する。

## Requirements

### Validated

- ✓ ETL pipeline: PostgreSQL (EveryDB2) → Parquet 抽出 — existing
- ✓ Feature engine: 14モジュール・100+列の特徴量生成 — existing
- ✓ 2-stage model: Win/Place/Wide 各P(hit)×E(odds|hit) — existing
- ✓ Surface submodel: 芝/ダート独立学習 — existing
- ✓ Backtest engine: 歴史シミュレーション + バンクロール追跡 — existing
- ✓ OOF prediction: 学習リーク防止 — existing
- ✓ Betting system: Strategy/Orchestrator/StakeCalculator/DDController — existing
- ✓ MLflow experiment tracking — existing
- ✓ Paper trading: ライブ推論パイプライン — existing
- ✓ 既存特徴量の単勝に対する有効性を分析・特定 — v1.0
- ✓ 単勝特化の特徴量を設計・実装(レース展開、勝ち癖等) — v1.0
- ✓ モデル構造の最適化(キャリブレーション、アンサンブル、EV補正) — v1.0
- ✓ 単勝ベッティング戦略の最適化(Kelly基準、レジーム適応) — v1.0
- ✓ 単勝モデルの品質検証(時系列交差検証、過学習チェック) — v1.0
- ✓ EMA重み付け時系列特徴量(TSER-01~03)とペースフィグア(PACE-01~02) — v1.1
- ✓ オッズ変動特徴量(acceleration + direction_consistency) — v1.1
- ✓ オッズ乖離EV特徴量(deviation_rank/zscore) + Conformal EV区間 — v1.1
- ✓ 3モデルスタッキング多様性強制(Optuna + early stopping + feature subset) — v1.1
- ✓ 単勝バックテスト決済・候補選択・ベット生成の修正 — v1.2
- ✓ 単勝ベット履歴・ROI診断・オッズバンド分析レポート — v1.2
- ✓ バックテストパイプライン高速化(ベクトル化・groupby辞書・特徴量キャッシュ) — v1.2
- ✓ EV_lower フィルター + OddsBandFilter + COLLAPSED regime skip — v1.3
- ✓ レジーム別Kelly分数 + EV比例乗算器 — v1.3
- ✓ DD%のみ3段階制御 + ParameterFreezeProtocol + Optuna TPE最適化 — v1.3
- ✓ WinSelectionGateをアンサンブルOOF予測で再学習・閾値適正化 + ドリフト診断 — v1.4
- ✓ EV_lower閾値をアンサンブルOOF分布に動的適合(25th percentile) — v1.4
- ✓ EV診断(ECE/Brier/Reliability/時系列ドリフト) — v1.4
- ✓ OddsBandFilterをアンサンブルベースtraining_bet_historyで再構築 + ルックアヘッドバイアス修正 — v1.4
- ✓ Optuna 16次元パラメータ最適化 + 4fold化 + multi-seed安定性検証 — v1.4
- ✓ PFP SHA256改ざん検知二重検証 + 自動検証レポート生成 — v1.4
- ✓ Isotonic EVキャリブレーション + オッズバンド別補正層 — v1.5
- ✓ EVCorrectionModel統合 + パイプライン適用 — v1.5
- ✓ バックテスト高速化5段階(キャリブレーションBT条件付きスキップ、Categorical包括適用等) — v1.5
- ✓ 高オッズ的中18新特徴量(クラストラジェクトリ/フォーム改善率/環境変化適性) — v1.5
- ✓ CQR Conformal EV予測区間(80%/90%信頼区間 + 動的フィルタリング) — v1.5
- ✓ 統合バックテスト検証 — v1.5
- ✓ POST_RACE情報漏洩完全排除 + 3層CI漏洩検出テスト — v1.6
- ✓ 100+特徴量Tier分類 + ノイズ特徴量プルーニング基盤 + コードハッシュキャッシュ無効化 — v1.6
- ✓ EveryDB2未活用データから22新特徴量(mining/血統/BMS/record/相対比較)抽出 — v1.6
- ✓ ドメイン知識交互作用項12個 + OOF安全ターゲットエンコーディング3特徴量 — v1.6
- ✓ マルチ年度BT (ROI 85.7%, +1.3pp改善) + 12モデルSHA256特徴量凍結manifest — v1.6
- ✓ ETL拡張: 三連複/馬連/三連単オッズParquet抽出 + DataRepository DI + カバレッジ検証 — v1.7
- ✓ Race-level集約特徴量6 + 既存特徴量2昇格 (implied_prob_hhi, odds_skewness) — v1.7
- ✓ 市場クロス整合性特徴量5 (Harville理論オッズ) + ワイドオッズmerge統合 — v1.7
- ✓ IC評価フレームワーク (B差分/C直交/E Incremental/Per-race + 方向一致性) — v1.7
- ✓ Gain per Depth診断 (179特徴量カテゴリマップ + MDR/FAD + 可視化CLI) — v1.7
- ✓ BT 2024 ROI 97.8% (v1.6: 85.7%, +12.1pp改善) + Manifest v1.7凍結 — v1.7

### Active

(None — next milestone to be defined via `/gsd:new-milestone`)

## Current Milestone: Planning Next Milestone

**Shipped:** v1.7 Market-Independent Edge Discovery (2026-05-19)
**Total phases completed:** 34 across 8 milestones
**BT ROI progress:** 84.4% (v1.5) → 85.7% (v1.6) → 97.8% (v1.7)

### Out of Scope

| Feature | Reason |
|---------|--------|
| 複勝/ワイドモデルの変更 | 単勝に集中するため |
| 新データ源の導入 | 既存EveryDB2データで十分 |
| 実馬券購入機能 | ペーパートレードまで |
| Web UI | CLIベースで十分 |
| リアルタイムオッズ収集の改善 | 既存機能をそのまま使用 |
| LSTM/Transformer時系列モデリング | 過去5-15走では過学習リスク高 |
| 複雑メタラーナー(GBM/NN) | 特徴量3個ではRidgeが最適 |
| sklearn StackingClassifier | ネイティブブースティングAPIとPIT安全フォールドに非対応 |
| 外部Kellyライブラリ導入 | 既存StakeCalculatorで十分、JRA固有制約はカスタム実装が必要 |
| モデル再学習 | 既存3モデルスタッキングをそのまま使用 |
| オッズ特徴量の除去 | Echo Chamber脱却 = 追加アプローチ。除去はC直交ICを悪化させる (実証済み) |
| Stern/Heneryモデル | Harvilleで90%+のシグナルを捕捉。複雑モデルは利益逓減 |

## Current State

**Shipped:** v1.7 Market-Independent Edge Discovery (2026-05-19)
**Phases:** 34 total (v1.0-v1.7)
**LOC:** ~24,100 (src/)
**Tests:** 1,540+ passed
**BT ROI:** 97.8% (v1.7), up from 85.7% (v1.6)
**C-orthogonal IC:** 0.2753 (market-independent predictive power confirmed)
**Next:** Planning next milestone

## Context

### 現状 (v1.7完了)

- 8マイルストーン34フェーズ完了 (v1.0〜v1.7)
- BT ROI 85.7% → 97.8% (+12.1pp改善)
- 11新特徴量 (6 rl_* + 5 MCF) + 2既存特徴量昇格
- C直交IC 0.2753で市場独立予測力を確認
- ダートROI 107.4%、Aggressive regime 116.7%で黒字セグメント確認
- 高オッズ帯(10.0+) ROI 179.9%で高オッズ改善確認
- Turf conservative regimeは赤字 — 改善余地あり
- GPD診断でStage1モデルがfundamental-dominatedであることを確認 (Echo Chamber脱却)

### 残存課題

- ROI 100%目標未達 (97.8%、あと2.2pp)
- Turf conservative regime unprofitable — 最大の改善余地
- training_pipeline.pyの_build_race_level_features()にrl_*列処理追加必要
- GPD診断はplace modelがなくても動作するように修正が必要
- WF検証未実行 — 過学習の有無未確認
- Human UAT 5項目がPostgreSQL環境依存で未実行
- test_training_pipeline.py 3件既知失敗(RecordFeatures.compute mock問題)

### 技術背景

- 3モデルGBMスタッキング: LightGBM + XGBoost + CatBoost (Optuna個別HP最適化)
- Isotonic EVキャリブレーション + オッズバンド別補正層
- CQR Conformal EV区間: 80%/90% 2レベルalpha信頼区間
- Race-level特徴量: entropy, dispersion, top3_gap, top1_odds, favorite_rank_gap, n_horses
- Market-cross特徴量: Harville理論オッズ比率、単勝×ワイド×三連複クロス整合性
- IC評価: B差分/C直交/E Incremental/Per-race + 方向一致性自動チェック
- GPD診断: FEATURE_CATEGORY_MAP (179 features), MDR/FAD メトリクス
- Parquetベースのデータパイプライン(PostgreSQLはETL専用)
- RegimeDetector: 3状態(aggressive/conservative/collapsed) + override_params外部注入
- 3段階ベットフィルター: COLLAPSED skip → 動的EV_lower → OddsBandFilter
- Kelly基準レジーム別サイジング: AGGRESSIVE(0.50)/CONSERVATIVE(0.25)/COLLAPSED(0.00)
- DD%のみ3段階制御: NORMAL/REDUCED/STOP + ヒステリシス + 段階的リカバリ
- Optuna TPE 16次元最適化
- ParameterFreezeProtocol: JSON manifest + SHA256改ざん検知 + 二重検証

## Constraints

- **Tech stack**: Python 3.11, LightGBM, XGBoost, CatBoost, Optuna, pandas, pyarrow
- **Data**: EveryDB2 (2015-2025) — 新データ源は追加しない
- **Testing**: 全テスト mock使用(DB不要) — 1,540+テスト
- **Code style**: Ruff (py311, line-length=100), mypy (strict)
- **No external services**: ローカル実行のみ、クラウドサービス不使用

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 単勝に集中 | 複勝で100%超え困難、単勝は中穴でエッジが出やすい | ✓ Good — ダート107.4%、Aggressive 116.7%で黒字 |
| 既存データ活用 | EveryDB2のデータ量(2015-2025)で十分な学習データがある | ✓ Good |
| 2段階モデル維持 | P×E分解は理論的に正当、改善は精度面で行う | ✓ Good |
| 特徴量分析から開始 | モデル改善の前に、どの特徴量が効いているか把握が必要 | ✓ Good (v1.0) |
| 3モデルスタッキング | 単一モデルより精度向上が期待できる | ✓ Good (v1.1) |
| EMA halflife=3 | 金融時系列解析の標準値。3走前の重みは直近の50% | ✓ Good (v1.1) |
| ペース3サブ特徴量分解 | LightGBMが非線形組み合わせを自動学習 | ✓ Good (v1.1) |
| Optuna探索空間分離 | 各モデルに異なる木複雑度で多様性強制 | ✓ Good (v1.1) |
| confidenceをpair_scoresのみ | 組合せ爆発(4次元目)を回避 | ✓ Good (v1.1) |
| ベッティング戦略は後回し | モデル精度を先に最大化し、戦略は精度が十分になってから調整 | ✓ Good (v1.3) |
| DD% only for DD control | WIN的中率10%環境ではROIがノイジーすぎてDD制御信号として不適切 | ✓ Good (v1.3) |
| EV_lower NaN フォールバック | fillna(1.0) で既存のedge>0のみで判定を維持 | ✓ Good (v1.3) |
| ヒステリシス付き状態遷移 | min_stay_racesでDD制御の発振防止 | ✓ Good (v1.3) |
| 独自軽量WFループ | pipeline.run()変更リスク回避、fold定義のみ管理 | ✓ Good (v1.3) |
| JSON manifest + SHA256 | sort_keys=True + indent=2 でdeterministic保証 | ✓ Good (v1.3) |
| フィルター再キャリブレーション順序 | Gate → EV_lower → OddsBand → Optuna → Validationの依存順 | ✓ Good (v1.4) |
| EV_lower 25th percentile | 固定1.0の過剰除外を解消、OOF分布に適合 | ✓ Good (v1.4) |
| 4fold Walk-forward | 14+自由パラメータで2foldは過学習リスク高 | ✓ Good (v1.4) |
| ルックアヘッドバイアス修正 | training_bet_history生成にデフォルトパラメータ使用 | ✓ Good (v1.4) |
| PFP二重検証 | freeze + verify で改ざん検知を全return pathで保証 | ✓ Good (v1.4) |
| Multi-seed安定性検証 | seed 42/43/44でCV判定、不安定次元を自動固定 | ✓ Good (v1.4) |
| Isotonic EVキャリブレーション | P×E独立性仮定を緩和しOOF予測で直接キャリブレーション | ✓ Good (v1.5) |
| CQR残差学習への変更 | 主モデル出力をCQR入力特徴量に含める設計変更 | ⚠️ Revisit (v1.5) — CQR過学習を引き起こし修正が必要だった |
| POST_RACE_COLS共通化 | domain/types.pyで一元管理し漏れ防止 | ✓ Good (v1.5) |
| 高オッズ18特徴量追加 | クラストラジェクトリ/フォーム改善率/環境変化適性 | ⚠️ Revisit — ROI寄与は間接的 |
| POST_RACE whitelist化 | blacklistの脆弱性をwhitelist FEATURE_COLSで排除 | ✓ Good (v1.6) — 3層CI検出で安全保証 |
| DataKubun=3優先 | 直前予想(馬体重発表後)が情報量最大 | ✓ Good (v1.6) |
| Stage1にTE追加せず | TE target == Stage1 targetでOOFリークの可能性 | ✓ Good (v1.6) — 安全性優先 |
| 特徴量追加アプローチの限界 | 22新特徴量+12交互作用+3TEでROI+1.3ppのみ | ⚠️ Revisit (v1.6) — Echo Chamber脱却で解決 |
| Echo Chamber脱却アプローチ | race-level + market-cross特徴量で市場独立性を獲得 | ✓ Good (v1.7) — C-orth IC 0.2753, ROI +12.1pp |
| Harville公式採用 | 計算コスト低+90%以上のシグナル捕捉 | ✓ Good (v1.7) |
| GPD raw metrics only | PASS/FAIL判定は恣意的、数値ベースで判断 | ✓ Good (v1.7) |
| DataRepository DI pattern | ParquetStore注入でテスト容易性確保 | ✓ Good (v1.7) |
| Manifest freeze proceeds regardless | 特徴量凍結は検証結果に依存しない | ✓ Good (v1.7) |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-19 after v1.7 milestone shipped*
