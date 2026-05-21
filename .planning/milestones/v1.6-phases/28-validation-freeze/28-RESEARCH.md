# Phase 28: Validation & Freeze - Research

**Researched:** 2026-05-15
**Domain:** バックテスト統合検証 + 特徴量セット凍結
**Confidence:** HIGH

## Summary

Phase 28はv1.6 Feature Engineering Overhaulの最終検証フェーズである。Phase 23-27で追加した全特徴量（血統、タイム指数、相対比較、mining予想、交互作用項、ターゲットエンコーディング）を統合したマルチ年度バックテスト（2023/2024/2025）を実行し、ROI改善を確認した上で特徴量セットをJSON manifest + SHA256 hashで凍結する。

既存の`run_backtest.py`は`--years 2023 2024 2025 --train-window 4`で3年マルチ年度BTに対応済み。`analyze_feature_importance.py --all-models`で全モデルのpermutation+gain重要度計算も可能。ParameterFreezeProtocolパターンは`save_strategy_manifest()` / `verify_strategy_manifest()`で確立済みで、特徴量凍結にも同一パターンを適用できる。

**重要な発見:** `data/strategy_manifest.json`が現在存在しない。Phase 13で作成されたはずだが、phase間で消失または未コミットの可能性がある。D-02の「既存manifestをそのまま使用」は、manifestファイルが存在しない場合にBTがデフォルト戦略パラメータで実行されることを意味する（`_load_strategy_params`がwarning付きでNoneを返す）。これは blocker ではないが、計画時に明示的に扱うべきである。

**Primary recommendation:** 3つの検証ステップ（pytest -> BT -> feature importance）を順次実行し、最後にfreeze manifestを生成する。strategy_manifest不存在はBT実行の blocker ではなく、デフォルトパラメータで動作する。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** マルチ年度3年テスト（2023/2024/2025）で検証。`--train-window 4`で学習（~3時間）
- **D-02:** 既存のstrategy_manifestをそのまま使用。Optuna再最適化は行わない
- **D-03:** BTフラグ: `--ensemble --calibration-bt --report --strategy-manifest data/strategy_manifest.json`
- **D-04:** 完全BTコマンド: `run_backtest.py --years 2023 2024 2025 --train-window 4 --ensemble --calibration-bt --report --strategy-manifest data/strategy_manifest.json`
- **D-05:** ROI絶対値100%到達が目標だが、100%未達でも改善幅ベースで記録して完了。「v1.5: 84.4% → v1.6: XX% (+Y.Ypp)」の形式
- **D-06:** ROI未達時は追加チューニングや追加Phaseを行わず、結果を記録して完了
- **D-07:** ParameterFreezeProtocol（Phase 13）のパターンを踏襲: JSON manifest + SHA256 hash。sort_keys=True + indent=2で決定論的
- **D-08:** SHA256 hashは各モデルのFEATURE_COLS毎に記録
- **D-09:** pytest全テスト通過確認 + マルチ年度BT + Feature importance再計算の3本柱
- **D-10:** Feature importance再計算は`analyze_feature_importance.py --all-models`を使用
- **D-11:** WF検証（~4時間）はスコープ外

### Claude's Discretion
- バックテスト結果の具体的な分析・解釈
- Feature importanceの結果に基づく推奨事項の記述
- 凍結manifestファイルの出力パス
- テスト結果レポートのフォーマット
- ROADMAP.md/PROJECT.mdの更新内容

### Deferred Ideas (OUT OF SCOPE)
- なし
</user_constraints>

<phase_requirements>
## Phase Requirements

Phase 28 has no formal REQ-IDs. It validates outputs from Phases 23-27.

| Prior Phase | Requirements Validated | Validation Method |
|-------------|----------------------|-------------------|
| Phase 23 | SAFE-01 (POST_RACE漏洩修正) | pytest通過で確認 |
| Phase 23 | SAFE-02 (feature importance監査スクリプト) | `analyze_feature_importance.py --all-models`実行で確認 |
| Phase 24 | AUDIT-01/02/03 (Tier分類・プルーニング・キャッシュ無効化) | BT結果 + テスト通過で確認 |
| Phase 25 | WIRE-01/02/03 (12特徴量配線) | BT結果に新特徴量が含まれることで確認 |
| Phase 26 | DATA-01/02/03/04 (血統・タイム・相対・mining特徴量) | Feature importanceに新特徴量が含まれることで確認 |
| Phase 27 | INTER-01/02/03 (交互作用・TE) | Feature importance + BT ROIで確認 |

**Success Criteria (from ROADMAP.md):**
1. 統合BT ROIが100%以上である（D-05により、100%未達でも改善幅で記録して完了）
2. 既存テスト全通過（回帰なし）
3. FEATURE_COLSが凍結され、ハッシュが記録されている
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| pytest実行 | CLI/Local | - | テストはDB不要、ローカルで完結 |
| マルチ年度BT | API/Backend (TrainingPipeline) | Database (ParquetStore) | PostgreSQL環境必須、~3時間の計算処理 |
| Feature importance | CLI/Script | Database (ParquetStore) | モデル+特徴量データ必要 |
| 特徴量凍結 | API/Backend (Model classes) | - | 各モデルクラスのFEATURE_COLSからmanifest生成 |
| 結果記録 | Documentation (ROADMAP.md等) | - | ドキュメント更新 |

## Standard Stack

### Core（既存、インストール不要）

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| LightGBM | (既存) | MLモデル | プロジェクト基盤 [VERIFIED: codebase] |
| pytest | (既存) | テスト実行 | 1,527テスト [VERIFIED: `pytest --co -q`] |
| hashlib | stdlib | SHA256 hash生成 | ParameterFreezeProtocol [VERIFIED: codebase] |
| json | stdlib | manifest生成 | sort_keys=True + indent=2 [VERIFIED: codebase] |

### Supporting（既存スクリプト）

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `scripts/run_backtest.py` | マルチ年度BT実行 | Phase 28メイン検証 |
| `scripts/analyze_feature_importance.py` | Feature importance再計算 | Phase 28品質確認 |
| `src/backtest/parameter_freeze_protocol.py` | SHA256 manifest生成 | 特徴量凍結 |

**Installation:** 不要。全て既存依存関係。

## Architecture Patterns

### System Architecture Diagram

```
[Phase 28 Validation Flow]

pytest tests/ -v
      |
      v
 (全1,527テスト通過?)
      |
      | YES
      v
run_backtest.py --years 2023 2024 2025 --train-window 4
      |  --ensemble --calibration-bt --report
      |  --strategy-manifest data/strategy_manifest.json
      |
      +---> Year 2023: train 2019-2022 / test 2023
      +---> Year 2024: train 2020-2023 / test 2024
      +---> Year 2025: train 2021-2024 / test 2025
      |
      v
 data/backtest/multi_year_result.json  (ROI, ベット数, 利益)
 data/validation/multi_year_validation_report.json  (PASS/FAIL)
      |
      v
analyze_feature_importance.py --all-models
      |
      v
 feature_importance_report.json + .csv
 data/audit/tier_report.json  (Tier 1/2分類)
      |
      v
freeze_feature_manifest.py (新規スクリプト or inline)
      |
      v
 data/feature_freeze_manifest.json  (各モデルのFEATURE_COLS + SHA256)
      |
      v
ROADMAP.md / PROJECT.md 更新 (ROI結果記録)
```

### Recommended Project Structure

```
scripts/
├── run_backtest.py           # 既存: マルチ年度BT
├── analyze_feature_importance.py  # 既存: importance監査
└── freeze_feature_manifest.py     # 新規: 特徴量凍結manifest生成

data/
├── backtest/
│   ├── multi_year_result.json     # BT出力: 年度別ROI
│   ├── multi_year_bet_history.json # BT出力: 全ベット履歴
│   └── predictions/               # BT出力: 年度別parquet
├── validation/
│   └── multi_year_validation_report.json  # PASS/FAIL判定
├── audit/
│   └── tier_report.json           # importance Tier分類
├── feature_freeze_manifest.json   # 新規: 凍結manifest
└── strategy_manifest.json         # ※現在存在しない（warning付きでBT実行）
```

### Pattern 1: 特徴量凍結Manifest生成

**What:** 各モデルクラスのFEATURE_COLSをJSON manifest + SHA256 hashで記録
**When to use:** Phase 28完了時（全変更確定後）

```python
# Source: src/backtest/parameter_freeze_protocol.py (save_strategy_manifest pattern)
import hashlib
import json
from pathlib import Path

def freeze_feature_manifest(output_path: Path) -> str:
    """全モデルのFEATURE_COLSを凍結manifestとして保存"""
    from models.stage1_ability_model import AbilityModel
    from models.two_stage_return_model import WinTwoStageModel, PlaceTwoStageModel
    from models.ev_correction_model import EVCorrectionModel, PlaceEVCorrectionModel
    from models.conformal_ev_model import ConformalEVModel
    from models.regime_detector import RegimeDetector
    from models.market_model import MarketModel
    from models.place_ability_model import PlaceAbilityModel

    models = {
        "AbilityModel": AbilityModel.FEATURE_COLS,
        "WinTwoStageModel": WinTwoStageModel.FEATURE_COLS,
        "PlaceTwoStageModel.HIT": PlaceTwoStageModel.HIT_FEATURE_COLS,
        "PlaceTwoStageModel.RETURN": PlaceTwoStageModel.RETURN_FEATURE_COLS,
        "EVCorrectionModel": EVCorrectionModel.FEATURE_COLS,
        "PlaceEVCorrectionModel": PlaceEVCorrectionModel.FEATURE_COLS,
        "ConformalEVModel": ConformalEVModel.FEATURE_COLS,
        "RegimeDetector": RegimeDetector.FEATURE_COLS,
        "MarketModel": MarketModel.FEATURE_COLS,
        "PlaceAbilityModel": PlaceAbilityModel.FEATURE_COLS,
    }

    manifest: dict = {}
    for name, cols in models.items():
        cols_json = json.dumps(cols, sort_keys=True, indent=2)
        sha256 = hashlib.sha256(cols_json.encode()).hexdigest()
        manifest[name] = {
            "feature_count": len(cols),
            "features": cols,
            "sha256": sha256,
        }

    # sort_keys=True + indent=2 で決定論的 (D-07)
    full_json = json.dumps(manifest, sort_keys=True, indent=2)
    overall_sha = hashlib.sha256(full_json.encode()).hexdigest()

    output = {
        "version": "v1.6",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_sha256": overall_sha,
        "models": manifest,
    }

    output_path.write_text(
        json.dumps(output, sort_keys=True, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return overall_sha
```

### Pattern 2: マルチ年度BT実行

**What:** 3年テストのROI集計
**When to use:** Phase 28メイン検証

```bash
# Source: scripts/run_backtest.py (D-04 exact command)
python scripts/run_backtest.py \
  --years 2023 2024 2025 \
  --train-window 4 \
  --ensemble \
  --calibration-bt \
  --report \
  --strategy-manifest data/strategy_manifest.json
```

BT完了後のROI抽出:

```python
# Source: data/backtest/multi_year_result.json (既存スキーマ)
import json
result = json.loads(Path("data/backtest/multi_year_result.json").read_text())
roi = result["overall"]["roi"]  # 0.844 = 84.4%
profit = result["overall"]["profit"]
total_bets = result["overall"]["total_bets"]
```

### Anti-Patterns to Avoid
- **凍結前にFEATURE_COLSを変更する:** freeze manifest生成前にいかなるコード変更も行わないこと。manifestは最終確定状態を記録する
- **strategy_manifest不存在でBTを中止する:** `_load_strategy_params`はwarning付きでNoneを返し、BTはデフォルトパラメータで継続する。blockerではない
- **1年だけのBTで判定する:** 3年（2023/2024/2025）のマルチ年度で年度別ばらつきを確認することがD-01の趣旨

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SHA256 manifest生成 | 自作hash関数 | `parameter_freeze_protocol.save_strategy_manifest()` パターン | 既存のsort_keys=True + indent=2で決定論的 |
| ROI判定 | 手動ROI計算 | `validation_report.evaluate_validation()` | PASS/FAIL判定ロジック既存 |
| BT実行フロー | 自作BTループ | `run_backtest.py` | マルチ年度対応、レポート出力、validation report全自動 |
| Feature importance計算 | 自作permutation | `analyze_feature_importance.py --all-models` | 全モデル一括、pivot_df形式、Tier分類対応 |

**Key insight:** Phase 28は新規実装を最小限に抑える検証フェーズ。既存スクリプトを最大限活用し、唯一の新規成果物は凍結manifestのみ。

## Common Pitfalls

### Pitfall 1: strategy_manifest.json不存在
**What goes wrong:** D-04のコマンドに`--strategy-manifest data/strategy_manifest.json`が含まれるが、ファイルが存在しない
**Why it happens:** Phase 13で作成されたが、phase間で消失または未コミット
**How to avoid:** `_load_strategy_params`はwarning付きでNoneを返すためBT自体は実行可能。ただし、Optuna最適化済みパラメータではなくデフォルトパラメータでBTが実行される点に注意
**Warning signs:** ログに「Strategy manifest が見つかりません — デフォルトパラメータを使用」と出力される

### Pitfall 2: BT実行にPostgreSQL環境が必要
**What goes wrong:** PostgreSQL/EveryDB2が稼働していない環境でBTを実行しようとする
**Why it happens:** ParquetStoreが`data/raw/races.parquet`等の事前ETL出力に依存する
**How to avoid:** BT実行はユーザーのローカル環境（PostgreSQL稼働中）で行う。CIでは実行不可
**Warning signs:** 「Parquetデータが見つかりません。先に run_etl.py を実行してください。」

### Pitfall 3: BT実行時間（~3時間）
**What goes wrong:** 3年 × (学習~17分 + キャリブレーションBT + テスト)で長時間実行
**Why it happens:** `--calibration-bt`が各年度で直近12ヶ月の軽量BTを追加実行する
**How to avoid:** タイムアウトを設定せず、完了を待つ。途中でCtrl+Cすると年度モデルは保存済み
**Warning signs:** `--calibration-bt`なしでも~41分/年、ありで~57分/年

### Pitfall 4: 凍結manifestのタイミング
**What goes wrong:** BT結果を見てFEATURE_COLSを微調整した後にmanifestを生成しようとする
**Why it happens:** ROIが低いと改善したくなる心理
**How to avoid:** D-06によりROI未達でも追加変更は行わない。manifestは現状のコードベースに対して生成する

## Code Examples

### 特徴量凍結Manifest（検証済みパターン）

```python
# 実行結果（2026-05-15時点のFEATURE_COLSから生成）:
#
# AbilityModel: 95 features, sha256=ce2e2c4e6738...
# WinTwoStageModel: 81 features, sha256=d237171ccbd4...
# PlaceTwoStageModel.HIT: 84 features, sha256=f6acfd667d37...
# PlaceTwoStageModel.RETURN: 86 features, sha256=0e9cfb78f8a8...
# EVCorrectionModel: 24 features, sha256=96923d4c91d1...
# PlaceEVCorrectionModel: 24 features, sha256=3a3239bbdb40...
# ConformalEVModel: 131 features, sha256=d6f9d5d1fd8f...
# RegimeDetector: 8 features, sha256=83fd38d2d7e8...
# MarketModel: 7 features, sha256=6d9240d610be...
# PlaceAbilityModel: 61 features, sha256=2ded412b894c...
```

### ROI抽出（既存スキーマ）

```python
# data/backtest/multi_year_result.json スキーマ:
{
  "overall": {
    "total_bets": 2253,           # int
    "total_stake": 225300.0,      # float (円)
    "total_return": 174450.0,     # float (円)
    "profit": -50850.0,           # float (円)
    "roi": 0.774300932090546,     # float (1.0 = 100%)
    "best_year": 2024,            # int
    "worst_year": 2024            # int
  },
  "years": {
    "2024": {
      "roi": 0.774300932090546,
      "total_bets": 2253,
      ...
    }
  }
}
```

### validation_report判定（既存ロジック）

```python
# Source: src/backtest/validation_report.py
def evaluate_validation(roi: float, total_bets: int) -> str:
    """ROI>1.0 and total_bets>=100 の場合 'PASS'"""
    if roi > 1.0 and total_bets >= 100:
        return "PASS"
    return "FAIL"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 単一年度BT | マルチ年度BT (--years) | Phase 22 (v1.5) | 年度別ばらつきの確認が可能 |
| 手動ROI記録 | validation_report.json | Phase 18 (v1.4) | PASS/FAIL自動判定 |
| なし | Tier分類 (Tier 1/2) | Phase 24 (v1.6) | ノイズ特徴量の定量的特定 |
| なし | ParameterFreezeProtocol | Phase 13 (v1.3) | OOS期間中のパラメータ不改変保証 |

**Deprecated/outdated:**
- `run_multi_year_backtest.py`: 廃止済み。`run_backtest.py --years`に統合。

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `data/strategy_manifest.json`がBT実行時に存在しない可能性が高い。`_load_strategy_params`はwarning付きでNoneを返すため blocker ではないが、デフォルトパラメータでBTが実行される | Architecture Patterns | Low - BT自体は実行可能だが、Optuna最適化済みパラメータが使われない |
| A2 | v1.5ベースラインROI 84.4%はROADMAP.md記載値。最新のmulti_year_result.jsonは77.4%（2024年のみの単年度結果）を示している。84.4%はv1.5終了時の値と推測 | User Constraints | Medium - 比較対象のベースラインが間違っていると改善幅評価が不正確 |

## Open Questions

1. **strategy_manifest.json の取扱い**
   - What we know: ファイルが現在存在しない。D-02/D-03/D-04で使用が決定されている
   - What's unclear: Phase 13で作成されたか、どこで消失したか
   - Recommendation: D-04のコマンドをそのまま実行し、warningでデフォルトパラメータが使用されることを受け入れる。または`--strategy-manifest`フラグを外して実行する

2. **v1.5ベースラインROI 84.4%の根拠**
   - What we know: CONTEXT.md D-05とROADMAP.mdで84.4%が言及されている
   - What's unclear: この値がどのBT構成での結果か（単年度/マルチ年度、strategy_manifestあり/なし）
   - Recommendation: 84.4%をベースラインとして使用し、Phase 28の改善幅を記録する

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | 全スクリプト | Yes | 3.11.15 | - |
| PostgreSQL/EveryDB2 | run_backtest.py | Unknown (local) | - | ユーザーローカル環境で実行 |
| pytest | テスト実行 | Yes | (既存) | - |
| LightGBM | analyze_feature_importance.py | Yes | (既存) | - |
| Parquetデータ (data/raw/) | run_backtest.py | Unknown | - | ETL事前実行が必要 |

**Missing dependencies with no fallback:**
- PostgreSQL環境: ユーザーローカルでのみ実行可能。CI不可

**Missing dependencies with fallback:**
- strategy_manifest.json: 存在しない場合、BTはデフォルト戦略パラメータで実行（warning付き）

## Sources

### Primary (HIGH confidence)
- `scripts/run_backtest.py` - マルチ年度BT実行フロー、出力スキーマ [VERIFIED: codebase]
- `scripts/analyze_feature_importance.py` - 全モデルimportance計算フロー [VERIFIED: codebase]
- `src/backtest/parameter_freeze_protocol.py` - SHA256 manifest生成パターン [VERIFIED: codebase]
- `src/models/*.py` - FEATURE_COLS定義（10モデルクラス）[VERIFIED: codebase、実数確認済み]
- `src/backtest/validation_report.py` - PASS/FAIL判定ロジック [VERIFIED: codebase]

### Secondary (MEDIUM confidence)
- `data/backtest/multi_year_result.json` - 直近BT結果（2024年単年度、ROI 77.4%）[VERIFIED: file read]
- `data/validation/multi_year_validation_report.json` - 直近検証レポート [VERIFIED: file read]
- `data/audit/tier_report.json` - Tier分類レポート [VERIFIED: file read]

### Tertiary (LOW confidence)
- v1.5ベースラインROI 84.4% - CONTEXT.md/ROADMAP.md記載値 [ASSUMED - 厳密なBT構成不明]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - 全て既存コードベースの確認済みパターン
- Architecture: HIGH - 実行フローをコードから直接確認
- Pitfalls: HIGH - strategy_manifest不存在を実際に確認（glob検索で0件）

**Research date:** 2026-05-15
**Valid until:** 2026-06-15 (stable - コードベース変更なし前提)
