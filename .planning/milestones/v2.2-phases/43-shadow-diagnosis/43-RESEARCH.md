# Phase 43: Shadow Diagnosis - Research

**Researched:** 2026-05-28
**Domain:** Shadow comparison diagnostics (post-hoc analysis of Phase 41 artifacts)
**Confidence:** HIGH

## Summary

Phase 43 は Phase 41 ShadowComparisonFramework の出力成果物 (JSON, Parquet) を入力とする後処理診断スクリプトです。BacktestEngine を再実行せず、既存の `shadow_comparison_result.json`, `shadow_race_diff.parquet`, `shadow_horse_diff.parquet`, `shadow_manifest.json` から読み取って、3ステップの段階的除外で劣化次元を特定します。

既存コードベースにすべての必要なパターンが揃っています: (1) `save_results()` / `save_manifest()` の複数成果物出力パターン、(2) `ShadowComparisonReportGenerator` の Jinja2 + HTML レポートパターン、(3) `DeploymentGateEvaluator` の独立評価器パターン、(4) `ComparisonMetrics` に含まれる Brier/logloss/ECE/actual_predicted_ratio の計算ロジック。Phase 43 はこれらを組み合わせて拡張するだけで実現可能です。

**Primary recommendation:** `ShadowComparisonFramework.compute_metrics()` と `_compute_ece()` を直接再利用し、セグメント定義 (popularity_band, probability_rank_band) を新規追加。horse_diff の列不足 (popularity, tanodds, surface 等) は missing_inputs として報告し、Phase 41 出力拡張のトリガーとする。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 後処理スクリプトとして実装。`scripts/run_shadow_diagnosis.py` (CLI) + `src/backtest/shadow_diagnosis.py` (ロジック)。Phase 41成果物のみを入力とし、再学習・BacktestEngine再実行は行わない。
- **D-02:** 3ステップの段階的除外で劣化次元を特定:
  1. 確率品質次元: 全馬ベースでBrier/logloss/ECE/actual_predicted_ratioを比較
  2. 選定次元: selected_changed vs unchangedレースに分けてROI/的中率/avg_odds/actual_predicted_ratioの差分
  3. キャリブレーション次元: surface/odds_band/popularity_band/probability_rank_band/selected_changed別にactual/predicted比率とECEを比較
- **D-03:** セグメント境界:
  - popularity_band: [1-3, 4-6, 7-9, 10-14, 15+]
  - probability_rank_band: [top1, 2-3, 4-6, 7+]
  - odds_band: Phase 41既存定義を流用
  - surface: turf/dirt
  - selected_changed: True/False
- **D-04:** 3ファイル構成: shadow_diagnosis_result.json, shadow_diagnosis_report.html, shadow_diagnosis_summary.md

### Claude's Discretion
- ShadowDiagnosis クラスの内部メソッド・データフロー設計
- Jinja2 HTMLテンプレートのレイアウト・スタイリング
- テスト構造・命名
- JSON出力のスキーマ設計
- missing_inputs 検出ロジックの実装詳細
- popularity_band / probability_rank_band 計算のエッジケース処理

### Deferred Ideas (OUT OF SCOPE)
- Phase 41出力拡張 (popularity_rank, probability_rank 列の追加)
- レジーム別分析
- LightGBM LambdaRank shadow variant
- 因果分解的アプローチ
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DIAG-01 | Shadow Comparisonで2024/2025固定foldのbaseline vs shadow確率品質(Brier/logloss/ECE)を比較し、劣化维度を特定する | Phase 41 `shadow_horse_diff.parquet` の `baseline_p_win_final` / `shadow_p_win_final` / `kakuteijyuni` 列でBrier/logloss/ECEを再計算可能。`ComparisonMetrics` の計算ロジックを再利用。 |
| DIAG-02 | RaceLevelRankerの選定パターン(baseline vs shadow)を比較し、的中/不的中レースの差分構造を明らかにする | Phase 41 `shadow_race_diff.parquet` の `selected_changed` / `baseline_selected_umaban` / `shadow_selected_umaban` / `baseline_result` / `shadow_result` 列で選定差分を分析可能。 |
| DIAG-03 | actual/predicted比率をsurface、odds_band、popularity_band、probability_rank_band、selected_changed別に比較 | `shadow_horse_diff.parquet` の `baseline_p_win_final` / `kakuteijyuni` でactual/predicted比率を計算可能。ただしpopularity/popularity_rank列はhorse_diffに含まれない可能性が高い → missing_inputs で報告。surface 列もhorse_diffには含まれない → race_diff からのマージが必要。 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| 確率品質比較 (DIAG-01) | Analysis (post-hoc) | — | Phase 41成果物からのメトリクス計算 |
| 選定パターン差分 (DIAG-02) | Analysis (post-hoc) | — | race_diffのselected_changed分析 |
| キャリブレーション乖離 (DIAG-03) | Analysis (post-hoc) | — | セグメント別actual/predicted比率計算 |
| HTML レポート生成 | Presentation | — | Jinja2 テンプレート |
| CLI エントリポイント | Scripts | — | argparse CLI |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | (installed) | 数値計算 | プロジェクト既存依存 |
| pandas | (installed) | データフレーム操作 | プロジェクト既存依存 |
| jinja2 | (installed) | HTML テンプレート | Phase 41 `ShadowComparisonReportGenerator` パターン |
| sklearn.metrics | (installed) | Brier/logloss/ECE計算の参考 | `ComparisonMetrics` で使用済み |

### Supporting
| Library | Purpose | When to Use |
|---------|---------|-------------|
| argparse | CLI 引数 | Phase 41 CLI パターン流用 |
| json | JSON 読み書き | Phase 41 成果物読み込み・結果出力 |
| pathlib | ファイルパス操作 | 全ファイルI/O |
| hashlib | SHA256 ハッシュ | manifest 検証 (Phase 42 パターン) |

**Installation:**
```bash
# 追加パッケージ不要 — 全てプロジェクト既存依存
pip install -e ".[dev]"
```

## Package Legitimacy Audit

> このフェーズは外部パッケージをインストールしないため、Audit不要。

**新規インストールパッケージ:** なし

## Architecture Patterns

### System Architecture Diagram

```
Phase 41 成果物 (入力)
    |
    +-- shadow_comparison_result.json -----> 全体メトリクス読み込み
    +-- shadow_race_diff.parquet ----------> レース別選定差分 (DIAG-02)
    +-- shadow_horse_diff.parquet --------> 馬別確率比較 (DIAG-01, DIAG-03)
    +-- shadow_manifest.json -------------> メタデータ・ハッシュ検証
    |
    v
ShadowDiagnosis.run()
    |
    +-- Step 1: 確率品質次元 (全馬ベース)
    |       Brier / logloss / ECE / actual_predicted_ratio
    |       baseline vs shadow 比較
    |
    +-- Step 2: 選定次元 (selected_changed/unchanged)
    |       ROI / 的中率 / avg_odds / actual_predicted_ratio
    |       changed vs unchanged レース差分
    |
    +-- Step 3: キャリブレーション次元 (セグメント別)
    |       surface / odds_band / popularity_band /
    |       probability_rank_band / selected_changed
    |       actual/predicted 比率 + ECE 比較
    |
    +-- missing_inputs 検出
    |       horse_diff に popularity, surface, tanodds 等が
    |       欠落している場合は記録
    |
    v
3ファイル出力
    +-- shadow_diagnosis_result.json  (Phase 44/45 用)
    +-- shadow_diagnosis_report.html  (人間レビュー用)
    +-- shadow_diagnosis_summary.md   (コミット/PR用)
```

### Recommended Project Structure
```
src/backtest/
  shadow_diagnosis.py          # ShadowDiagnosis クラス (メインロジック)
  templates/
    shadow_diagnosis_report.html  # Jinja2 HTML テンプレート
scripts/
  run_shadow_diagnosis.py      # CLI エントリポイント
tests/
  test_shadow_diagnosis.py     # ユニットテスト
```

### Pattern 1: 後処理診断パターン
**What:** Phase 41 成果物を読み込み、再計算なしで診断分析を行う
**When to use:** BacktestEngine 再実行を伴わない分析フェーズ全般
**Example:**
```python
# src/backtest/shadow_diagnosis.py
class ShadowDiagnosis:
    def __init__(self, input_dir: Path) -> None:
        self.input_dir = input_dir
        self.result_json = self._load_json(input_dir / "shadow_comparison_result.json")
        self.race_diff = pd.read_parquet(input_dir / "shadow_race_diff.parquet")
        self.horse_diff = pd.read_parquet(input_dir / "shadow_horse_diff.parquet")
        self.missing_inputs: list[str] = []

    def run(self) -> ShadowDiagnosisResult:
        step1 = self._step1_probability_quality()
        step2 = self._step2_selection_pattern()
        step3 = self._step3_calibration_by_segment()
        self._detect_missing_inputs()
        return ShadowDiagnosisResult(step1=step1, step2=step2, step3=step3, ...)
```

### Pattern 2: Jinja2 HTML レポート生成
**What:** ShadowComparisonReportGenerator パターンを踏襲
**When to use:** Phase 41 パターンに従う HTML レポート
**Example:**
```python
# ShadowComparisonReportGenerator のパターン (src/backtest/shadow_report.py)
class ShadowDiagnosisReportGenerator:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.template_dir = Path(__file__).parent / "templates"

    def generate(self, diagnosis_result: dict) -> Path:
        env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True,
        )
        template = env.get_template("shadow_diagnosis_report.html")
        html = template.render(...)
        outpath = self.output_dir / "shadow_diagnosis_report.html"
        outpath.write_text(html, encoding="utf-8")
        return outpath
```

### Pattern 3: セグメント計算 (Phase 41 compute_metrics 再利用)
**What:** `ShadowComparisonFramework.compute_metrics()` と `_compute_ece()` を直接再利用
**When to use:** Brier/logloss/ECE/actual_predicted_ratio の計算
**Example:**
```python
# Phase 41 から import して再利用
from backtest.shadow_comparison import ShadowComparisonFramework

_fw = ShadowComparisonFramework(variants=[])  # stateless で計算のみ利用可能
metrics = _fw.compute_metrics(pd.DataFrame(), segment_df, variant_name, bet_history)
```

### Anti-Patterns to Avoid
- **Anti-pattern: BacktestEngine の再実行** — Phase 43 は後処理のみ。engine.py への依存は不可。
- **Anti-pattern: horse_diff 列の前提** — `popularity`, `surface`, `tanodds`, `closing_win_odds` は `_align_horse_level()` の `align_cols` に含まれていないため、horse_diff に存在しない可能性が高い。これらの列は missing_inputs として報告し、フォールバック戦略を用意すること。
- **Anti-pattern: 新たなモデル学習** — 診断フェーズでモデルや特徴量を変更しない。

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Brier/logloss/ECE 計算 | カスタム実装 | `ShadowComparisonFramework.compute_metrics()` + `_compute_ece()` | Phase 41 で検証済みのロジック。ECE の 10-bin 実装を含む |
| セグメント別メトリクス | グループ化ループ | `ShadowComparisonFramework.compute_metrics_by_group()` | odds_band 計算済み。popularity_band/probability_rank_band は新規追加が必要 |
| HTML レポート生成 | ゼロから構築 | `ShadowComparisonReportGenerator` パターン + CSS変数 | 既存テンプレートと一貫したスタイル |
| JSON/Parquet 出力 | 独自シリアライザ | `save_results()` / `save_manifest()` パターン | 成果物ハッシュ付き manifest 構造 |

**Key insight:** Phase 43 のコア計算ロジックの大部分は `ShadowComparisonFramework` の既存メソッドで再利用可能。新規に必要なのは (1) popularity_band / probability_rank_band のセグメント定義、(2) 3ステップの分析フロー、(3) missing_inputs 検出のみ。

## Common Pitfalls

### Pitfall 1: horse_diff 列不足
**What goes wrong:** Phase 41 の `_align_horse_level()` は `align_cols = ["p_win_final", "investment_score", "stake", "win_market_selection_score"]` のみをマージする。`popularity`, `surface`, `tanodds`, `closing_win_odds` は bet_history に含まれるが horse_diff には転記されない。
**Why it happens:** Phase 41 設計時に DIAG-03 の要件が未定義だったため。
**How to avoid:** `missing_inputs` 検出で不足列を報告。Phase 41 の出力拡張候補として記録。代替手段: (1) race_diff に surface が含まれていれば horse_diff にマージ、(2) horse_diff の `baseline_p_win_final` から確率順位を計算して probability_rank_band を代替生成。
**Warning signs:** horse_diff に `popularity` / `surface` / `tanodds` 列が存在しない。

### Pitfall 2: popularity_band 計算に popularity 列が必要
**What goes wrong:** D-03 の `popularity_band` (単勝オッズ順位) は bet_history の `popularity` 列に基づくが、horse_diff には popularity が含まれない可能性が高い。
**Why it happens:** `_align_horse_level()` の align_cols に含まれていない。
**How to avoid:** (1) tanodds があればオッズベースで代替ランク計算、(2) p_win_final のランクを代用、(3) 不可の場合は missing_inputs に追加。
**Warning signs:** セグメント別キャリブレーションで popularity_band が "unknown" にフォールバック。

### Pitfall 3: 空セグメントのフォールバック
**What goes wrong:** 出走頭数 < 4 のレースで probability_rank_band "7+" が存在しない。
**Why it happens:** レースサイズがセグメント定義に比べて小さい。
**How to avoid:** D-03 指定通り `unknown` にフォールバックし、missing/unknown 件数をレポート出力。
**Warning signs:** セグメント別ECEで n_in_bin=0 のビンが多い。

### Pitfall 4: JSON出力の消費者互換性
**What goes wrong:** Phase 44 (ROI Bisect) が shadow_diagnosis_result.json の特定スキーマに依存する。
**Why it happens:** Phase 43/44 間のインターフェースが未定義。
**How to avoid:** D-04 で `shadow_diagnosis_result.json` を Phase 44/45 の主要入力として定義。JSON スキーマに `step1_probability_quality`, `step2_selection_pattern`, `step3_calibration`, `missing_inputs`, `recommendations` のセクションを含める。

### Pitfall 5: ECE 計算の variant name 前提
**What goes wrong:** `compute_metrics()` は `{variant_name}_p_win_final` 列を期待する。Phase 41 horse_diff の列名は `baseline_p_win_final` と `ridge_shadow_p_win_final` (または shadow variant名)。
**Why it happens:** variant_name は Phase 41 CLI の --baseline-name / --shadow-name で決まる。
**How to avoid:** manifest から variant 名を動的に取得し、列名プレフィックスとして使用する。

## Code Examples

### Phase 41 成果物の読み込みパターン
```python
# Source: src/backtest/shadow_comparison.py (save_results) と
#         src/backtest/deployment_gates.py (DeploymentGateEvaluator.evaluate)
import json
from pathlib import Path
import pandas as pd

def load_phase41_artifacts(input_dir: Path) -> dict:
    result_json = json.loads(
        (input_dir / "shadow_comparison_result.json").read_text(encoding="utf-8")
    )
    race_diff = pd.read_parquet(input_dir / "shadow_race_diff.parquet")
    horse_diff = pd.read_parquet(input_dir / "shadow_horse_diff.parquet")
    manifest = json.loads(
        (input_dir / "shadow_manifest.json").read_text(encoding="utf-8")
    )
    return {
        "result_json": result_json,
        "race_diff": race_diff,
        "horse_diff": horse_diff,
        "manifest": manifest,
    }
```

### ECE 再利用パターン
```python
# Source: src/backtest/shadow_comparison.py (_compute_ece)
from backtest.shadow_comparison import ShadowComparisonFramework

_fw = ShadowComparisonFramework(variants=[])

# horse_diff の subset で ECE を計算
segment = horse_diff[horse_diff["some_column"] == "some_value"]
p_col = "baseline_p_win_final"
p_vals = pd.to_numeric(segment[p_col], errors="coerce")
is_win = (segment["kakuteijyuni"] == 1).astype(float)
valid = p_vals.notna() & (p_vals > 0) & (p_vals < 1)
ece = ShadowComparisonFramework._compute_ece(
    p_vals[valid].values, is_win[valid].values, n_bins=10
)
```

### popularity_band / probability_rank_band 定義
```python
# D-03 セグメント境界
POPULARITY_BAND_EDGES = [0, 3, 6, 9, 14, float("inf")]
POPULARITY_BAND_NAMES = ["1-3", "4-6", "7-9", "10-14", "15+"]

PROB_RANK_BAND_EDGES = [0, 1, 3, 6, float("inf")]
PROB_RANK_BAND_NAMES = ["top1", "2-3", "4-6", "7+"]

def assign_popularity_band(popularity: pd.Series) -> pd.Series:
    return pd.cut(
        popularity,
        bins=POPULARITY_BAND_EDGES,
        labels=POPULARITY_BAND_NAMES,
        right=True,
    )
```

### CLI パターン (Phase 41 流用)
```python
# Source: scripts/run_shadow_comparison.py
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shadow Diagnosis — baseline vs shadow diagnostic analysis",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing Phase 41 shadow comparison artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/backtest/shadow/diagnosis"),
        help="Output directory (default: data/backtest/shadow/diagnosis)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML report",
    )
    return parser
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 手動 BT ログ比較 | ShadowComparisonFramework (Phase 41) | Phase 41 (v2.1) | JSON/Parquet 成果物で機械的比較が可能に |
| DeploymentGateEvaluator 内の簡易比較 | 3ステップ段階的除外診断 (Phase 43) | Phase 43 (v2.2) | 確率品質/選定/キャリブレーションの分離で劣化次元を特定 |

**Deprecated/outdated:**
- Phase 41 `compute_metrics_by_group()` の `prob_rank_band` 分岐: 空の dict を返す実装 (line 837-839)。Phase 43 で新規実装が必要。

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | horse_diff に `popularity` 列が含まれない | DIAG-03, Pitfall 1/2 | popularity_band が全て unknown になる。代替ランク計算で対応可能 |
| A2 | horse_diff に `surface` 列が含まれない | DIAG-03 | surface 別キャリブレーションが実行不可。race_diff からのマージで対応可能 |
| A3 | horse_diff に `kakuteijyuni` 列は含まれる (Phase 41 line 717-719 でマージ) | DIAG-01/03 | 確率品質比較が不可。Phase 41 出力に依存 |
| A4 | Phase 41 の variant 名は manifest の flag_states から特定可能 | Pitfall 5 | variant 名をハードコードする必要が生じる |
| A5 | `ShadowComparisonFramework(variants=[])` で stateless に compute_metrics / _compute_ece を利用可能 | Pattern 3 | 計算ロジックを独自実装する必要が生じる |

## Open Questions (RESOLVED)

1. **horse_diff の実際の列構成** — RESOLVED
   - What we know: `_align_horse_level()` は `align_cols = ["p_win_final", "investment_score", "stake", "win_market_selection_score"]` のみをマージ。`kakuteijyuni` は baseline_df から別途マージ。
   - What's unclear: Phase 41 実行時に bet_history に含まれる他の列（popularity, surface, tanodds 等）が、DataFrame 変換時に暗黙的に horse_diff に含まれるか。baseline_df は `pd.DataFrame(bt_result.bet_history)` で作成されるため、bet_history の全列が baseline_df に含まれるが、_align_horse_level は `baseline_df[key_cols].copy()` で key_cols のみをベースにマージする。
   - RESOLVED: Phase 41 実行後の実際の parquet ファイルを確認。存在しない列は `_detect_missing_inputs()` で検出し missing_inputs に記録。Plan 01 の `_step3` でフォールバック処理を実装済み。

2. **race_diff に含まれる列の完全リスト** — RESOLVED
   - What we know: `_align_race_level()` は `diff_row` に `race_id`, `baseline_selected_umaban`, `shadow_selected_umaban`, `selected_changed`, および `baseline_/shadow_` prefix 付きの `tanodds`, `p_win_final`, `win_selection_ev`, `win_market_selection_score`, `result`, `stake`, `closing_win_odds`, `investment_score` を含む。
   - What's unclear: race_diff に `surface` が含まれるか（Phase 41 のコードを見る限り含まれない）。
   - RESOLVED: race_diff に surface が無い場合、`_detect_missing_inputs()` で missing_inputs に記録。surface セグメントは horse_diff に surface があればそちらから取得し、両方に無ければ `unknown` にフォールバック。

## Environment Availability

> このフェーズは Phase 41 成果物 (JSON/Parquet) の読み込みと Python スクリプト実行のみ。外部DB、サービス、ツールは不要。

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | Runtime | ✓ (mise) | 3.11 | — |
| pandas + pyarrow | Parquet I/O | ✓ (installed) | — | — |
| numpy | 数値計算 | ✓ (installed) | — | — |
| jinja2 | HTML レポート | ✓ (installed) | — | — |
| Phase 41 成果物 | 入力データ | 要確認 | — | 入力パスをCLI引数で指定 |

**Missing dependencies with no fallback:**
- Phase 41 成果物ファイルが `--input-dir` に存在する必要がある

**Missing dependencies with fallback:**
- なし

## Validation Architecture

> nyquist_validation は config.json で `false` に設定されているため、このセクションは省略。

## Security Domain

> このフェーズは分析/診断のみであり、セキュリティ関連の変更はない。ASVS 該当なし。

## Sources

### Primary (HIGH confidence)
- `src/backtest/shadow_comparison.py` — Phase 41 Framework。save_results(), compute_metrics(), _compute_ece(), _align_horse_level(), _align_race_level() の全実装
- `scripts/run_shadow_comparison.py` — CLI パターン
- `src/backtest/shadow_report.py` — ShadowComparisonReportGenerator パターン
- `src/backtest/deployment_gates.py` — DeploymentGateEvaluator パターン
- `src/backtest/engine.py` — BacktestResult, bet_history 列定義
- `src/backtest/report.py` — BacktestReportGenerator パターン
- `src/backtest/templates/shadow_comparison_report.html` — HTML テンプレートパターン
- `.planning/phases/43-shadow-diagnosis/43-CONTEXT.md` — フェーズ決定事項

### Secondary (MEDIUM confidence)
- `src/backtest/race_predictor.py` — shadow mode パターン、investment_score 設定箇所
- `src/models/market_aware_win_calibrator.py` — MAWC セグメント定義 (ODDS_BAND_EDGES, POP_BUCKET_EDGES)
- `.planning/phases/41-shadow-comparison-framework/41-CONTEXT.md` — Phase 41 決定事項
- `.planning/phases/42-feature-routing-audit-safety-gates/42-CONTEXT.md` — Phase 42 決定事項

### Tertiary (LOW confidence)
- なし — 全てコードベースから直接確認済み

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — 既存プロジェクト依存のみ、新規パッケージなし
- Architecture: HIGH — Phase 41 パターンの踏襲、既存コードからの逸脱なし
- Pitfalls: HIGH — horse_diff 列不足は Phase 41 ソースコードから確認済み

**Research date:** 2026-05-28
**Valid until:** 2026-06-27 (stable — 既存コードベース依存)
