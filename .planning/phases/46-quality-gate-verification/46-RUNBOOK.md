# Phase 46: Quality Gate Verification -- Runbook

**Purpose:** 各品質ゲートステップを手動再現するためのコマンド集。orchestration CLIのトラブル時、個別ステップの再実行、手動検証に使用する。

**When to use:**
- orchestration CLI (`run_phase46_quality_gates.py`) がエラーで停止した場合
- 特定ステップだけを再実行したい場合
- 個別ステップの結果を手動で確認したい場合

---

## 1. Prerequisites

### Required Data Files

| File | Purpose |
|------|---------|
| `data/oof/oof_predictions.parquet` | OOF予測データ (OOFHealthValidator用) |
| `data/models-backtest/` | Baselineモデルディレクトリ (2024/2025年サブディレクトリ必須) |
| `data/models-backtest-mawc-conservative/` | Stage 1で生成される保守的variant (Stage 2で必要) |

### Environment

```bash
# Python 3.11 + project dependencies
pip install -e ".[dev]"
```

---

## 2. Stage 1: MAWC Conservative Retrain

Stage 1はPhase 45で実装した保守的MAWC再学習を実行し、保守的variantを生成する。

### Full Command

```bash
python scripts/run_mawc_conservative_retrain.py \
  --oof-path data/oof/oof_predictions.parquet \
  --source-model-dir data/models-backtest \
  --target-root data/models-backtest-mawc-conservative \
  --years 2024 2025 \
  --report
```

### Expected Output

| File | Description |
|------|-------------|
| `data/models-backtest-mawc-conservative/manifest.json` | C grid探索結果、deployed/non-deployed判定 |
| `data/models-backtest-mawc-conservative/retrain_summary.md` | 品質ゲート詳細、推奨C値 |
| `data/models-backtest-mawc-conservative/mawc_conservative_report.html` | HTMLレポート |

### Decision Criteria

- **PASS:** `manifest.json`が存在し、`per_year_surface`内に少なくとも1つのsurfaceで`deployed=true`が存在する
- **FAIL:** manifestが存在しない、または全surfaceで`deployed=false`

```bash
# Verify manifest exists and has deployed surfaces
python -c "
import json
from pathlib import Path
m = json.loads(Path('data/models-backtest-mawc-conservative/manifest.json').read_text())
deployed = False
for year_data in m.get('per_year_surface', {}).values():
    for surface_data in year_data.values():
        if surface_data.get('deployed', False):
            deployed = True
            break
assert deployed, 'No deployed surfaces found in manifest'
print('Stage 1 PASS: deployed surfaces found')
"
```

### Troubleshooting

- **manifestにdeployed surfaceが無い場合:** `retrain_summary.md`で品質ゲートFAILの詳細を確認。D-04によりPhase 46内でのリトライは不可。Phase 45bまたは次フェーズで対応。
- **OOFファイルが見つからない場合:** `data/oof/oof_predictions.parquet`の存在確認。前回train実行で生成されている必要がある。
- **Expected runtime:** ~10 minutes

---

## 3. Stage 2 Step 1: FeatureRoutingAudit

50+28禁止特徴量がキャリブレータ/ランカーに漏洩していないことをCI安全監査で確認する。

### CLI Command

```bash
python scripts/run_feature_routing_audit.py --output-dir data/audit
```

### Function API (CLI代替)

```bash
python -c "
from audit.feature_routing_registry import run_feature_audit
import json
r = run_feature_audit()
print(json.dumps({'status': r['overall_status']}, indent=2))
assert r['overall_status'] == 'PASS', f'Audit FAILED: {r}'
print('FeatureRoutingAudit PASS')
"
```

### Decision Criteria

- **PASS:** `overall_status`が`"PASS"`
- **FAIL:** `overall_status`が`"FAIL"` (禁止特徴量の漏洩検出)

### What It Checks

- Critical models: 50禁止特徴量 (POST_RACE情報、OOF予測、ターゲットエンコーディング等)
- Advisory models: 28禁止特徴量 (診断情報、上位モデル出力等)
- 全モデルのFEATURE_COLSがレジストリと一致することを検証

### Expected Runtime

< 1 second

---

## 4. Stage 2 Step 2: OOFHealthValidator

OOF予測データの健全性を検証する。

### Function API (CLIなし -- Pitfall 5)

OOFHealthValidatorには専用CLIが存在しないため、関数APIを使用する。

```bash
python -c "
import pandas as pd
from validation.oof_health_validator import OOFHealthValidator, OOF_PREDICTIONS_PROFILE
from pathlib import Path

df = pd.read_parquet('data/oof/oof_predictions.parquet')
v = OOFHealthValidator()
r = v.validate(df, OOF_PREDICTIONS_PROFILE)
print(f'Status: {r[\"status\"]}')
assert r['status'] == 'PASS', f'OOF validation FAILED: {r}'
print('OOFHealthValidator PASS')
"
```

### Decision Criteria

- **PASS:** `status`が`"PASS"`
- **FAIL:** いずれかの検査(OOF-01~OOF-08)がFAIL

### What It Checks

| Check ID | Description |
|----------|-------------|
| OOF-01 | Empty DataFrame検出 |
| OOF-02 | Train/Test overlap検出 |
| OOF-03 | Top1 anomaly検出 |
| OOF-04 | Row coverage検証 |
| OOF-05 | Fold count検証 |
| OOF-06 | Multi-fold races検証 |
| OOF-07 | Required columns検証 |
| OOF-08 | SHA256 manifest検証 |

### Expected Runtime

< 1 second

---

## 5. Stage 2 Step 3: Shadow Comparison

Baseline vs保守的variantのShadow Comparisonを実行する。最も時間がかかるステップ。

### Full Command

```bash
python scripts/run_shadow_comparison.py \
  --baseline-root data/models-backtest \
  --shadow-root data/models-backtest-mawc-conservative \
  --folds 2024 2025 \
  --output-dir data/backtest/shadow_mawc_conservative \
  --baseline-name baseline \
  --shadow-name mawc_conservative \
  --report
```

### Decision Criteria

- **PASS:** `data/backtest/shadow_mawc_conservative/shadow_comparison_result.json`が正常に生成される
- **FAIL:** 実行エラー、または成果物生成失敗

### Important: Variant Naming (Pitfall 1)

`--shadow-name`は必ず`mawc_conservative`を指定すること。デフォルトの`shadow`を使用すると、下流のShadow Diagnosis/DploymentGateEvaluatorでvariant名不整合が発生する。

```bash
# Verify output
python -c "
import json
from pathlib import Path
p = Path('data/backtest/shadow_mawc_conservative/shadow_comparison_result.json')
assert p.exists(), 'shadow_comparison_result.json not found'
r = json.loads(p.read_text())
print(f'Generated: {r.get(\"generated_at\", \"unknown\")}')
print('Shadow Comparison PASS: result file generated')
"
```

### Expected Runtime

~82 minutes (2 years x ~41 min/year)

### Troubleshooting

- **メモリ不足:** 大規模データセットでのモデルロード。必要に応じて`--folds`を単年に減らす。
- **モデルディレクトリ不在:** Stage 1の`data/models-backtest-mawc-conservative/{year}/`が存在することを確認。
- **manifest collision:** 既存の`data/backtest/shadow/`と混同しないこと。出力先は`shadow_mawc_conservative`。

---

## 6. Stage 2 Step 4: Shadow Diagnosis

Shadow Comparisonの結果に対する3ステップ段階的除外診断を実行する。

### CLI Command

```bash
python scripts/run_shadow_diagnosis.py \
  --input-dir data/backtest/shadow_mawc_conservative \
  --output-dir data/backtest/shadow_mawc_conservative/diagnosis \
  --report
```

### Function API (CLI代替)

```bash
python -c "
from backtest.shadow_diagnosis import ShadowDiagnosis, save_diagnosis_results
from pathlib import Path

sd = ShadowDiagnosis(Path('data/backtest/shadow_mawc_conservative'))
r = sd.run()
out = Path('data/backtest/shadow_mawc_conservative/diagnosis')
out.mkdir(parents=True, exist_ok=True)
save_diagnosis_results(r, out)
print('Shadow Diagnosis PASS: diagnosis complete')
"
```

### Decision Criteria

- **PASS:** 診断がエラーなく完了し、成果物が生成される
- **FAIL:** 実行エラー

### Output Files

| File | Description |
|------|-------------|
| `diagnosis/shadow_diagnosis_result.json` | 機械可読JSON結果 |
| `diagnosis/shadow_diagnosis_summary.md` | 人間可読Markdown要約 |
| `diagnosis/shadow_diagnosis_report.html` | HTMLレポート (--report時) |

### Expected Runtime

seconds

---

## 7. Stage 2 Step 5: DeploymentGateEvaluator

確率品質・ベット数維持・再現性・診断の4ゲートを評価する。

### Function API

```bash
python -c "
from backtest.deployment_gates import run_deployment_gates
from pathlib import Path

r = run_deployment_gates(
    'data/backtest/shadow_mawc_conservative/shadow_comparison_result.json',
    'data/backtest/shadow_mawc_conservative/shadow_manifest.json',
    'data/backtest/shadow_mawc_conservative/gates'
)
print(f'Overall status: {r.overall_status}')
assert r.overall_status == 'PASS', f'Deployment gates FAILED: {r.overall_status}'
print('DeploymentGateEvaluator PASS')
"
```

### Decision Criteria

- **PASS:** `overall_status`が`"PASS"`
- **FAIL:** `overall_status`が`"FAIL"` または `"WARN"`

### Important: SKIP Gates (Pitfall 4)

`diagnostic_oof_health`と`diagnostic_feature_routing_audit`ゲートは常に`SKIP`となる。これはOOF/監査がDeploymentGateEvaluator内で実行されないことを意味し、ステップ1およびステップ2の個別確認を置き換えるものではない。

### 4 Gates Evaluated

| Gate | Description |
|------|-------------|
| probability_quality | Brier/logloss/ECE baseline比非悪化 |
| bet_count_maintenance | shadow bet_count >= baseline * 0.95 |
| reproducibility | 成果物の再現性検証 |
| diagnostics | Shadow Diagnosis結果の健全性 |

### Expected Runtime

< 1 second

---

## 8. Full Orchestration (Preferred)

上記の全ステップを自動実行するorchestration CLI。Phase 46の推奨実行方法。

### Commands

```bash
# Full execution (Stage 1 + Stage 2)
python scripts/run_phase46_quality_gates.py --years 2024,2025 --report

# Stage 1 only (conservative variant generation)
python scripts/run_phase46_quality_gates.py --stage 1

# Stage 2 only (when Stage 1 already complete)
python scripts/run_phase46_quality_gates.py --stage 2

# Force re-run (ignore existing artifacts)
python scripts/run_phase46_quality_gates.py --force --report

# Specific output directory
python scripts/run_phase46_quality_gates.py --output-dir data/backtest/phase46_quality_gates --report
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--stage` | auto | `1`, `2`, or `auto` (auto-detect from artifacts) |
| `--years` | `2024,2025` | Test fold years |
| `--baseline-root` | `data/models-backtest` | Baseline model directory |
| `--shadow-root` | `data/models-backtest-mawc-conservative` | Conservative variant directory |
| `--output-dir` | `data/backtest/phase46_quality_gates` | Output directory |
| `--force` | False | Force re-run all steps |
| `--report` | False | Generate Markdown summary |
| `--train-window` | 4 | Training window years |
| `--betting-target` | `win` | Betting target |

### Expected Runtime

~90 minutes total (Stage 1 ~10 min + Stage 2 ~82 min, dominated by Shadow Comparison)

### Output Files

| File | Description |
|------|-------------|
| `data/backtest/phase46_quality_gates/phase46_quality_gate_result.json` | JSON集約結果 (3-label判定含む) |
| `data/backtest/phase46_quality_gates/phase46_quality_gate_summary.md` | Markdown要約 (--report時) |

---

## 9. Known Pitfalls Reference

### Pitfall 1: Shadow Comparison Variant Naming

`--shadow-name`は必ず`mawc_conservative`を指定する。デフォルトの`shadow`を使用するとvariant名不整合が発生し、下流のShadow Diagnosis/DploymentGateEvaluatorで列名エラーとなる。

### Pitfall 2: Stage 1 Output Directory Does Not Exist Yet

Stage 1が初回実行時に`data/models-backtest-mawc-conservative/`を作成する。Stage 2を先に実行しようとするとディレクトリ不在エラーとなる。必ずStage 1を先に実行すること。

### Pitfall 4: DeploymentGateEvaluator SKIP Gates

DeploymentGateEvaluatorの`diagnostic_oof_health`と`diagnostic_feature_routing_audit`ゲートは常にSKIP。これはOOF/監査がPASSしたことを意味しない。Step 1 (FeatureRoutingAudit)とStep 2 (OOFHealthValidator)を独立して実行・確認すること。

### Pitfall 5: OOFHealthValidator Has No CLI

OOFHealthValidatorには専用CLIが存在しない。関数API (`OOFHealthValidator().validate()`)で呼び出す必要がある。上記Step 2のコマンドを参照。

### Pitfall 6: Manifest Key Naming

manifest.jsonは`per_year_surface`キーを使用する (`per_surface`ではない)。year-keyed dict構造でsurface結果にアクセスすること。

---

## 10. 3-Label Decision Framework (D-03)

品質ゲート結果は3つの独立したラベルで評価する。

### Quality Gate

- **PASS:** 全5ステップがPASS
- **FAIL:** いずれかのステップがFAIL

### ROI Trend

ROI判定は品質ゲート判定とは独立。Phase 46の必須PASS条件ではない。

| Label | Condition | Description |
|-------|-----------|-------------|
| `recovered` | ROI >= 90% | 回復達成 |
| `weak_recovery` | 87.8% <= ROI < 90% | 部分回復 |
| `not_recovered` | ROI < 87.8% | 回復未達 |

Baseline: 87.8% (v2.0 close)。Target: 100%+ (目標だが必須ではない)。

### Deployment

品質ゲート + ROIトレンドの組み合わせで判定。

| Label | Condition | Description |
|-------|-----------|-------------|
| `deployable` | Quality Gate PASS + ROI recovered/weak_recovery | 配備可能 |
| `not_deployable` | Quality Gate FAIL | 品質不足で配備不可 |
| `manual_review` | Quality Gate PASS + ROI not_recovered | 品質OKだがROI未達、人間判断必要 |

### Decision Matrix

| Quality Gate | ROI Trend | Deployment |
|-------------|-----------|------------|
| PASS | recovered | deployable |
| PASS | weak_recovery | deployable |
| PASS | not_recovered | manual_review |
| FAIL | any | not_deployable |

---

## 11. Error Recovery

### Stage 1 FAIL

Phase 46内ではリトライ不可 (D-04)。Phase 45bまたは次フェーズでC値再探索・特徴量構成変更を検討。

### Stage 2 Step 3 (Shadow Comparison) FAIL

ランタイムエラーの場合、`--force`フラグで再実行可能。モデル/データ不整合の場合はStage 1の再確認が必要。

### Intermediate Artifact Resume

Orchestration CLIは既存成果物を検出してskip/resume可能。`--force`で全ステップ強制再実行。

### Manual Step Isolation

各ステップは独立して実行可能。例: Shadow Comparisonだけを再実行する場合、Step 3のコマンドを単独実行し、その後Step 4, 5を順次実行する。
