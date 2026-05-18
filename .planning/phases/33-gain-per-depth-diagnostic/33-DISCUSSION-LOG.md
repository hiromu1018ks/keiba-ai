# Phase 33: Gain per Depth Diagnostic - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-18
**Phase:** 33-Gain per Depth Diagnostic
**Areas discussed:** 特徴量3分類の定義方法, 分析対象モデルの範囲, 出力形式と可視化, Two-Stage仮説の検証方法

---

## 特徴量3分類の定義方法

| Option | Description | Selected |
|--------|-------------|----------|
| コード内dict (Recommended) | gpd_diagnostics.py内にdict[str, str]を定義。最もシンプル | |
| YAML設定ファイル | config/gpd_categories.yamlに定義。管理箇所が増える | |
| 命名規則ベース | 接頭辞ベース自動分類。境界が曖昧 | |
| 単一dict + テスト検証 | 全特徴量をカバー、未分類をテストで自動検出 | ✓ |

**User's choice:** ベストプラクティス追求 → 単一dict + テスト検証
**Notes:** 実装難易度は問わない。明示的マッピングで確実性を優先。

### 3分類の境界基準

| Option | Description | Selected |
|--------|-------------|----------|
| 3分類基準案 | Market=オッズ系、Fundamental=能力系、Categorical=カテゴリ系 | ✓ |
| 独自基準 | ユーザーが独自に指定 | |

**User's choice:** 3分類基準案に確定
**Notes:** Market(オッズ・市場構造・市場クロス整合性・FLB・overround) / Fundamental(過去成績・血統・調教・馬体・フォーム・コース・ペース) / Categorical(騎手・調教師・種牡馬・TE・レース条件)

---

## 分析対象モデルの範囲

| Option | Description | Selected |
|--------|-------------|----------|
| 中核予測モデルのみ (Recommended) | Ability+Win2Stage+Market+StackedEnsemble LGBM (~8-10モデル) | |
| SubmodelSet内全Booster | 全LightGBM Booster (~22+モデル) | |
| 全LightGBMモデル | SubmodelSet + RegimeDetector + QualityScreener | |

**User's choice:** ベストプラクティス追求 → SubmodelSet内全LightGBM Booster（階層化出力）
**Notes:** 主要分析(主要モデル) + 詳細分析(補足モデル) の階層化。RegimeDetector/QualityScreenerは除外。

### StackedEnsemble内3モデル

| Option | Description | Selected |
|--------|-------------|----------|
| LightGBMのみ分析 (Recommended) | trees_to_dataframe()利用可能 | ✓ |
| 3モデル全部分析 | XGBoost/CatBoostも試みる | |

**User's choice:** LightGBM主対象。XGBoost将来拡張文書化のみ。CatBoost API制約でスキップ。
**Notes:** コードは拡張可能設計にして将来XGBoost追加時の変更を最小化。

---

## 出力形式と可視化

| Option | Description | Selected |
|--------|-------------|----------|
| JSON + Console (Recommended) | 既存診断パターン。依存関係追加なし | |
| JSON + Console + グラフ画像 | matplotlib/plotly PNG生成 | ✓ |
| You decide | Claudeの判断 | |

**User's choice:** JSON + Console + グラフ画像(PNG)
**Notes:** GPD-02の「可視化」要件により忠実に。

### グラフライブラリ

| Option | Description | Selected |
|--------|-------------|----------|
| matplotlib (Recommended) | 既存依存関係。CLI環境対応 | ✓ |
| plotly | インタラクティブHTML。CLIでは不要 | |
| Claudeの判断 | 最適なものを選択 | |

**User's choice:** matplotlib

### グラフ粒度

| Option | Description | Selected |
|--------|-------------|----------|
| 主要グラフ2-3枚 (Recommended) | サマリ重視 | |
| モデル毎の個別グラフ | 詳細分析 | ✓ |

**User's choice:** モデル毎の個別グラフ

### グラフ内容

| Option | Description | Selected |
|--------|-------------|----------|
| stacked bar + cumulative gain (Recommended) | 標準的な可視化 | |
| Claudeの判断 | 最適な可視化を設計 | ✓ |

**User's choice:** Claudeの判断で最適設計

---

## Two-Stage仮説の検証方法

| Option | Description | Selected |
|--------|-------------|----------|
| 二分法(Shallow/Deep) (Recommended) | depth閾値で二分 | |
| 5段階分類 | depth 1-5+の5段階 | |
| Claudeの判断 | 最適な検証方法 | |

**User's choice:** ベストプラクティス追求 → 連続depth分析
**Notes:** 二分法/5段階の恣意的閾値を避け、全depthレベルを連続的に分析。Market Dominance Ratio + Fundamental Activation Depthの2指標を自動計算。

### 仮説判定方法

| Option | Description | Selected |
|--------|-------------|----------|
| 指標のみ出力、人間が判定 (Recommended) | 診断ツールとして指標提供 | |
| 自動判定(ALERT/WARN/PASS) | 閾値ベース自動判定 | |
| You decide | Claudeの判断 | ✓ |

**User's choice:** You decide → Claude判断で「指標のみ出力、人間が判定」
**Notes:** 自動判定の閾値はデータを見る前に決める恣意的なもの。console_summary()で視覚的に提示。

---

## Claude's Discretion

- グラフの具体的なデザイン（stacked barの配置、色、サブプロット構成等）
- FEATURE_CATEGORY_MAPの完全な内容
- モデル名→Boosterアクセスパスの抽象化方法
- テストケースの具体的な設計
- JSON出力のスキーマ詳細
- モデル毎のPNGファイル命名規則

## Deferred Ideas

- XGBoost trees_to_dataframe() によるdepth別分析 — 将来フェーズ
- CatBoost 木構造分析 — API制約により将来検討
- GPD-05 多次元直交IC — REQUIREMENTS.md Future要件
