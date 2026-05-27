# Phase 37: OOF Health Infrastructure - Research

**Researched:** 2026-05-27
**Domain:** OOF artifact validation infrastructure / data integrity
**Confidence:** HIGH

## Summary

Phase 37は、全OOF成果物に対する統一的な健全性検査基盤を構築する。現状は`_validate_win_selection_oof_health()`が単一artifactのtop1 hit rate/ROI検査のみを担っており、空OOF・fold数不足・同一race_id複数fold混入・行数カバレッジ・メタデータ必須化などの検査が不在である。新設する`OOFHealthValidator`クラスでこれら全検査(OOF-01~08)を統一的に実装し、生産者側(OOF保存前)と消費者側(OOF読込時)の2段階validation戦略を確立する。

既存コードベースに強固な基盤が存在する: `_walk_forward_race_splits()`がrace_id単位splitを提供し、`_validate_win_selection_oof_health()`がtop1 hit rate/ROI検証の実装パターンを示し、`freeze_feature_manifest.py`がSHA256 manifestの確立パターン(sort_keys=True + indent=2)を提供する。新規外部依存の追加は不要(REQUIREMENTS.md "Out of Scope"で明記)。

**Primary recommendation:** `src/validation/oof_health_validator.py`にOOFHealthValidatorを新設し、artifact profile dataclassで検査設定を定義。既存の`_validate_win_selection_oof_health()`はOOFHealthValidatorのメソッドに抽出・統合する。fold列は各OOF生成器(AbilityModel.train_oof, generate_ev_oof_predictions, generate_win_oof_predictions)で直接DataFrameに記録する。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 共通`OOFHealthValidator`クラスを新設。全OOF検査(OOF-01~08)を統一的に実装
- **D-02:** 各artifactはexplicit artifact profileで検査設定を定義。profileは enabled checks, required columns, score_col, return/odds columns, fold_col, expected row source, manifest path を含む
- **D-03:** 常時有効な検査: OOF-01(empty), OOF-04(row coverage), OOF-05(min fold count), OOF-06(same race multiple fold), OOF-07(required metadata), OOF-08(manifest generation)
- **D-04:** Profile依存の検査:
  - OOF-02 (train/valid overlap): split metadataが利用可能な場合のみ有効
  - OOF-03 (top1 hit rate/ROI): profileがscore_colとreturn/odds columnsを定義している場合のみ有効
- **D-05:** fold列はOOF生成時に各生成器でDataFrameに記録。後からの推測・復元は禁止
  - `AbilityModel.train_oof()` -> `ability_oof_fold` 列を追加
  - `generate_ev_oof_predictions()` -> `ev_oof_fold` 列を追加
  - `generate_win_oof_predictions()` -> `win_oof_fold` / `win_selection_oof_fold` 列を整合的に記録
- **D-06:** `_prepare_oof_artifact()`や`OOFHealthValidator`でのfold推測は禁止
- **D-07:** レガシーartifact（fold列なし）はhealth validationでfail。明示的なone-time migrationコマンドでのみ受け入れ可能
- **D-08:** JSON manifest。artifact別に個別ファイル: `data/oof/manifests/oof_predictions.health.json`, `data/oof/manifests/win_selection_oof.health.json`
- **D-09:** Index file: `data/oof/manifests/index.json`
- **D-10:** Manifest内容の全フィールド(artifact_name, artifact_path, artifact_hash, artifact_version, schema_hash, schema_dtype_hash, row_count, expected_row_count, row_coverage_ratio, race_count, horse_count, fold_col, fold_count, fold_row_counts, fold_race_counts, fold_race_id_uniqueness, train_valid_overlap_count, same_race_multiple_fold_count, top1_score_col, top1_hit_rate, top1_roi, return_col_used, date_min, date_max, train_date_range, source_model_hash, source_code_version/git_commit, generated_at, validator_version, status, failures, warnings)
- **D-11:** schema_hash計算: 列名ソート->SHA256(schema_hash) + sorted "column_name:dtype" pairs->SHA256(schema_dtype_hash)
- **D-12:** JSONが信頼できる唯一の情報源。Parquet metadataとMLflowは将来のoptional mirror
- **D-13:** 生産者側: OOF artifact保存前にvalidate()必須実行。fail時はartifactを書き込まない。保存後にartifact_hash計算しmanifestにstatus=PASSで書き込む
- **D-14:** 消費者側: artifact-specific manifestを読み込み、status/artifact_hash/schema_hash/validator_version/required profileを確認。不一致でfail-fast
- **D-15:** デバッグ/CI用に`force_revalidate`オプションを提供

### Claude's Discretion
- OOFHealthValidatorの具体的なクラス構造（メソッド分割、profile class定義）
- artifact_hashの計算方法
- expected_row_countの計算方法
- fold_col名の統一命名規則
- 既存`_validate_win_selection_oof_health()`のOOFHealthValidatorへの移行方針
- legacy migrationコマンドのインターフェース
- テストファイル配置

### Deferred Ideas (OUT OF SCOPE)
- None (discussion stayed within phase scope)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| OOF-01 | OOF成果物の空保存を禁止し、fail-fastで異常終了 | `OOFHealthValidator.validate()`内で`df.empty`チェック。既存パターン: `_validate_win_selection_oof_health()` line 247の`df.empty`早期returnをValueError raiseに変更 |
| OOF-02 | race_id単位でtrain/validの重複を検査 | `_walk_forward_race_splits()`のsplit metadata利用。profile依存検査(D-04)として実装 |
| OOF-03 | top1 hit rate > 35% or top1 ROI > 200% で停止 | 既存`_validate_win_selection_oof_health()` lines 271-296のパターンをOOFHealthValidatorに統合。定数はprofileに移動 |
| OOF-04 | OOF行数が期待行数の70%未満で停止 | expected_row_count計算: 学習データ行数 * (1 - 1/n_folds)を基準。`row_coverage_ratio`をmanifestに記録 |
| OOF-05 | fold数 < 3 で停止 | `fold_count = df[fold_col].nunique()` でチェック。定数MIN_FOLD_COUNT=3 |
| OOF-06 | 同一race_idの複数fold混入検査 | race_id単位でfold一意性を確認: `df.groupby("race_id")[fold_col].nunique().max() == 1` |
| OOF-07 | is_oof=Trueとfold列を必須化 | profile.required_columnsに`is_oof`とfold_col名を含める。OOF-05/06と組み合わせてfold列の存在自体も検証 |
| OOF-08 | health manifest生成 | `freeze_feature_manifest.py`のSHA256パターン(sort_keys=True + indent=2)を踏襲。manifest内容はD-10参照 |
| XCT-05 | 同一入力から決定的な出力保証 | manifestのsort_keys=True + indent=2 + artifact_hash(SHA256)でdeterminismを保証。既存パターンと同一 |
| XCT-08 | 全artifactにversion/schema hash/source OOF manifest path含める | manifestの`artifact_version`, `schema_hash`, `source_oof_manifest_path`フィールドで対応 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|-----------|-------------|----------------|-----------|
| OOF health validation | ML Pipeline / Python | - | OOF artifactは学習パイプライン内で生成・検証されるため、validationもパイプライン層に属する |
| Health manifest persistence | Filesystem (JSON) | - | manifestはJSON fileとしてdata/oof/manifests/に保存。DB/外部サービスへの依存なし |
| Fold column recording | ML Models | - | AbilityModel, WinTwoStageModel等のOOF生成器がDataFrameに直接記録 |
| Consumer-side validation | ML Pipeline / Python | - | バックテスト/Phase 38-40のOOF読込時にmanifest-first checkを実行 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| hashlib (stdlib) | 3.11 | SHA256 hash計算 | Python標準ライブラリ。freeze_feature_manifest.pyと同じパターン [VERIFIED: codebase] |
| json (stdlib) | 3.11 | manifest読み書き | sort_keys=True + indent=2 でdeterministic。既存パターン [VERIFIED: codebase] |
| dataclasses (stdlib) | 3.11 | ArtifactProfile定義 | プロジェクト全体で@dataclass(frozen=True)が使用されている [VERIFIED: codebase] |
| pandas | 2.3.3 | DataFrame操作 | OOF artifactの検査・操作。既存依存 [VERIFIED: codebase] |
| numpy | >=1.26 | 数値計算 | NaN処理、統計計算。既存依存 [VERIFIED: codebase] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | 9.0.2 | テスト | OOFHealthValidatorの単体テスト。mock使用(DB不要) [VERIFIED: codebase] |
| pathlib (stdlib) | 3.11 | ファイルパス操作 | manifest fileの読み書き [VERIFIED: codebase] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|-----------|-----------|----------|
| dataclass (frozen) | TypedDict | TypedDictはruntime validationがない。frozen dataclassはimmutabilityを保証し、プロジェクトのdomain/models.pyパターンと一致 |
| hashlib SHA256 | Parquet metadata hash | D-11で明示的に否定。圧縮/エンジンmetadataが意味的変更なしに変わる可能性があるため |
| pydantic BaseModel | dataclass | pydanticはプロジェクト依存にない。追加不要(REQUIREMENTS.md Out of Scope) |

**Installation:**
外部パッケージのインストールは不要。stdlib + 既存依存のみで実装可能。

## Package Legitimacy Audit

> 新規外部パッケージのインストールなし。stdlib + 既存依存のみ。audit不要。

## Architecture Patterns

### System Architecture Diagram

```
[OOF Generators]                    [OOFHealthValidator]              [Health Manifests]
                                    (src/validation/)

AbilityModel.train_oof()     \
  + ability_oof_fold          |
                              |
generate_ev_oof_predictions() |-----> validate(df, profile)  -----> data/oof/manifests/
  + ev_oof_fold               |        |                            oof_predictions.health.json
                              |        |-- OOF-01: empty check      win_selection_oof.health.json
generate_win_oof_predictions()|        |-- OOF-04: row coverage      index.json
  + win_oof_fold              |        |-- OOF-05: min fold count
                              |        |-- OOF-06: fold uniqueness     ^
_prepare_oof_artifact()       /        |-- OOF-07: required metadata   |
  + is_oof, oof_artifact_version       |-- OOF-02: train/valid overlap |
                                        |-- OOF-03: top1 hit/ROI       |
                                        v                              |
                                   PASS/FAIL                         |
                                        |                              |
                                        v                              |
                               [Save Parquet] --> [Compute Hash] -----+
                               [Write Manifest with status=PASS]


[Consumer Side (Phase 38-40, Backtest)]
        |
        v
  load_manifest(artifact_name)
        |
        |-- Check status=PASS
        |-- Verify artifact_hash matches
        |-- Verify schema_hash matches
        |-- Check validator_version
        |
        v
  OK -> Proceed with OOF data
  FAIL -> ValueError (fail-fast)
```

### Recommended Project Structure
```
src/
├── validation/                           # NEW directory
│   ├── __init__.py                       # NEW
│   └── oof_health_validator.py           # NEW: OOFHealthValidator + ArtifactProfile
data/
├── oof/
│   ├── oof_predictions.parquet           # EXISTING: fold列追加後再生成
│   ├── win_selection_oof.parquet         # EXISTING: 既にwin_selection_oof_foldあり
│   └── manifests/                        # NEW directory
│       ├── index.json                    # NEW
│       ├── oof_predictions.health.json   # NEW
│       └── win_selection_oof.health.json # NEW
tests/
└── test_oof_health_validator.py          # NEW: 全検査の単体テスト
```

### Pattern 1: Artifact Profile (Frozen Dataclass)
**What:** OOF artifact種別ごとの検査設定を定義するimmutable dataclass
**When to use:** 全OOF検査で。profileごとに有効/無効な検査とパラメータを切り替える

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class OOFHealthProfile:
    """OOF artifact別の検査設定。"""
    artifact_name: str
    required_columns: tuple[str, ...]
    fold_col: str
    score_col: str | None = None          # OOF-03用 (profile依存)
    return_cols: tuple[str, ...] = ()     # OOF-03用 (profile依存)
    max_top1_hit_rate: float = 0.35       # OOF-03閾値
    max_top1_roi: float = 2.0             # OOF-03閾値
    min_fold_count: int = 3               # OOF-05閾値
    row_coverage_threshold: float = 0.70  # OOF-04閾値
    enable_train_valid_overlap: bool = False  # OOF-02 (D-04)
    manifest_path: str = ""               # D-08
    strict_schema: bool = False           # D-11: optional列mismatchでfailにするか

# 具体profile例
OOF_PREDICTIONS_PROFILE = OOFHealthProfile(
    artifact_name="oof_predictions",
    required_columns=("race_id", "race_date", "is_oof", "oof_artifact_version"),
    fold_col="ability_oof_fold",  # D-05: train_oof()で追加される列名
    score_col="p_win_oof",        # profile依存 (D-04)
    return_cols=("confirmed_odds", "tanodds"),
    manifest_path="data/oof/manifests/oof_predictions.health.json",
)

WIN_SELECTION_OOF_PROFILE = OOFHealthProfile(
    artifact_name="win_selection_oof",
    required_columns=("race_id", "race_date", "is_oof", "oof_artifact_version", "kakuteijyuni"),
    fold_col="win_selection_oof_fold",
    score_col="win_market_selection_score",  # D-04
    return_cols=("win_return_unit", "win_return", "confirmed_odds", "tanodds"),
    max_top1_hit_rate=0.35,
    max_top1_roi=2.0,
    manifest_path="data/oof/manifests/win_selection_oof.health.json",
)
```

### Pattern 2: Two-Stage Validation (Producer/Consumer)
**What:** 生産者側(full validation + save)と消費者側(manifest check only)の2段階
**When to use:** 全OOF保存時(生産者)と全OOF読込時(消費者)

```python
# 生産者側 (D-13): training_pipeline.py内
def _save_oof_with_validation(df, profile, path, train_date_range):
    validator = OOFHealthValidator()
    result = validator.validate(df, profile, train_date_range=train_date_range)
    if result["status"] != "PASS":
        raise ValueError(f"OOF health check failed: {result['failures']}")
    # validation通過後にのみ保存
    df.to_parquet(path, index=False)
    # artifact_hash計算 + manifest書き込み
    artifact_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = validator.generate_manifest(df, profile, artifact_hash, train_date_range)
    manifest_path = Path(profile.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _update_index(profile.artifact_name, manifest_path, artifact_hash)

# 消費者側 (D-14): Phase 38-40のOOF読込時
def load_validated_oof(artifact_name: str) -> pd.DataFrame:
    manifest = _load_manifest(artifact_name)  # index.json -> manifest path -> load
    if manifest["status"] != "PASS":
        raise ValueError(f"OOF manifest status is {manifest['status']}")
    artifact_path = Path(manifest["artifact_path"])
    actual_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if actual_hash != manifest["artifact_hash"]:
        raise ValueError("artifact_hash mismatch")
    return pd.read_parquet(artifact_path)
```

### Pattern 3: SHA256 Schema Hash (D-11)
**What:** 列名ソート->SHA256 と 列名:dtypeソート->SHA256 の2種のhash
**When to use:** 全OOF artifactのschema一意性確認

```python
def _compute_schema_hashes(df: pd.DataFrame) -> tuple[str, str]:
    """D-11: 2種のschema hashを計算する。"""
    cols_sorted = sorted(df.columns.tolist())
    schema_hash = hashlib.sha256(
        json.dumps(cols_sorted).encode()
    ).hexdigest()

    dtype_pairs = sorted(f"{col}:{df[col].dtype}" for col in df.columns)
    schema_dtype_hash = hashlib.sha256(
        json.dumps(dtype_pairs).encode()
    ).hexdigest()
    return schema_hash, schema_dtype_hash
```

### Anti-Patterns to Avoid
- **Fold推測禁止 (D-06):** `_prepare_oof_artifact()`や`OOFHealthValidator`でfold列を後から推測・復元してはならない。実際の学習splitと不一致の可能性があり、偽の安全性を与える
- **空OOFの保存:** `df.empty`の場合はValueErrorで即停止。既存の`_validate_win_selection_oof_health()`は`df.empty`時にearly returnでmetrics返していたが、OOF-01ではfail-fastが必須
- **Parquet metadata hash:** 圧縮パラメータやエンジンmetadataが変わるとhashが変わるため、semantic hash(列名+dtype)のみを使用(D-11)
- **health check通過前のartifact書き込み:** D-13でvalidate fail時はartifactを書き込まない。順序は generate -> validate -> save -> hash -> manifest

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Fold splitのtrain/valid境界 | 独自の境界計算 | 既存`_walk_forward_race_splits()` | TimeSeriesSplitベースのexpanding window。既にテスト済み |
| Top1 hit rate/ROI計算 | 新しい集計ロジック | 既存`_validate_win_selection_oof_health()` lines 271-296のパターン | `_win_selection_oof_return_unit()`と組み合わせた実績ある計算 |
| Deterministic JSON manifest | 独自のserializer | `json.dumps(sort_keys=True, indent=2)` パターン | `freeze_feature_manifest.py`と同一パターン。既にXCT-05準拠 |

**Key insight:** このPhaseは全てstdlib + 既存依存で完結する。新規外部パッケージ不要(REQUIREMENTS.md "Out of Scope"で明記: 既存LightGBM 4.6.0 + sklearn 1.8.0 + betacal 1.1.0で全Phase対応可能)。

## Common Pitfalls

### Pitfall 1: generate_ev_oof_predictions()が配列を返すためfold列の追加方法が異なる
**What goes wrong:** `generate_ev_oof_predictions()`は3つのnumpy配列`(oof_ev_corrected, oof_actual_return, oof_odds)`を返す。DataFrameではないため、単純にfold列を追加できない。
**Why it happens:** AbilityModel.train_oof()はDataFrameを返すが、generate_ev_oof_predictions()は配列を返すという設計上の違い。
**How to avoid:** generate_ev_oof_predictions()の呼び出し元(training_pipeline.py内)で、配列を受け取った後にDataFrameにfold列を関連付ける。または、generate_ev_oof_predictions()自体のシグネチャを変更してfold情報も返す。D-05の「generate_ev_oof_predictions() -> ev_oof_fold列を追加」をどう実現するかは設計判断。
**Warning signs:** EV OOF fold列がNaNまたは欠損している。

### Pitfall 2: win_selection_oof_fold列は既に存在するが、oof_predictionsにはfold列がない
**What goes wrong:** `win_selection_oof.parquet`には既に`win_selection_oof_fold`列があるが、`oof_predictions.parquet`にはfold列がない。AbilityModel.train_oof()を修正する必要がある。
**Why it happens:** win_selection_oofは後から追加されたartifactで、fold列の重要性が認識されていた。oof_predictionsはPhase 30で追加された際にfold列が考慮されていなかった。
**How to avoid:** AbilityModel.train_oof()の戻り値DataFrameに`ability_oof_fold`列を追加する。boundaries配列のindexをfold番号としてtest_maskの行に設定する。
**Warning signs:** oof_predictions.parquetにfold列がなく、D-07によりhealth validationがfailする。

### Pitfall 3: generate_win_oof_predictions()内のfold記録箇所
**What goes wrong:** `generate_win_oof_predictions()`は配列を返すため、fold情報を直接記録できない。呼び出し元の`_walk_forward_race_splits()`ループ内でfold番号を記録する必要がある。
**Why it happens:** win_benter_gate.pyのgenerate_win_oof_predictions()は純粋な配列を返す設計。fold情報はtraining_pipeline.py側のWinSelectionGate学習ループ内で管理されている(既存: `fold_val["win_selection_oof_fold"] = fold_idx` at line 1692)。
**How to avoid:** 既存のfold記録パターン(line 1692)を確認し、generate_win_oof_predictions()の戻り値を使う側で正しくfold列を設定する。
**Warning signs:** win_oof_foldとwin_selection_oof_foldの整合性が取れていない。

### Pitfall 4: テストにおけるmockパターンの不一致
**What goes wrong:** 既存テスト(test_training_pipeline.py)で`_validate_win_selection_oof_health`をmockしている箇所が3箇所(lines 434, 496, 550)ある。これらをOOFHealthValidatorのmockに置き換える必要がある。
**Why it happens:** mockのimport pathが変わるため、既存テストのpatch decoratorを更新しないとテストが失敗する。
**How to avoid:** OOFHealthValidator導入時に、`_validate_win_selection_oof_health`の代わりに`OOFHealthValidator.validate`をmockするようテストを更新。
**Warning signs:** `pip install -e ".[dev]" && python -m pytest tests/test_training_pipeline.py`でimport errorやmock error。

### Pitfall 5: Legacy artifactの取り扱い
**What goes wrong:** 既存のdata/oof/oof_predictions.parquetにはfold列がない。D-07によりhealth validationでfailするため、次回の学習実行までこのファイルはlegacy扱いになる。
**Why it happens:** Phase 37の実装前から保存されているartifactはfold列を持たない。
**How to avoid:** D-07に従い、legacy artifactはhealth validationでfailさせる。学習パイプラインの次回実行時に新しくfold列付きのartifactが生成される。migrationコマンドはone-time用として別途実装。
**Warning signs:** 既存のoof_predictions.parquetを使おうとしてvalidation failが発生。

## Code Examples

### AbilityModel.train_oof() へのfold列追加パターン (D-05)
```python
# src/models/stage1_ability_model.py train_oof() 内
# 現在の実装 (line 364):
oof_preds = pd.Series(np.nan, index=df.index, dtype=np.float64)

# 変更後: fold列も同時に記録
oof_preds = pd.Series(np.nan, index=df.index, dtype=np.float64)
oof_folds = pd.Series(np.nan, index=df.index, dtype=pd.Int64Dtype())

for i in range(n_folds):
    # ... existing split logic ...
    oof_preds.loc[test_mask] = test_df["p_ability_win"].values
    oof_folds.loc[test_mask] = i  # D-05: fold番号を記録

# 最後にDataFrameに設定
df["p_ability_win"] = oof_preds
df["ability_oof_fold"] = oof_folds  # NEW
return df
```

### OOFHealthValidator.validate() の基本構造
```python
# src/validation/oof_health_validator.py
@dataclass(frozen=True)
class ValidationResult:
    status: str          # "PASS" or "FAIL"
    failures: list[str]
    warnings: list[str]
    metrics: dict[str, Any]

class OOFHealthValidator:
    VALIDATOR_VERSION = "1.0.0"

    def validate(
        self,
        df: pd.DataFrame,
        profile: OOFHealthProfile,
        *,
        train_date_range: tuple[str, str] | None = None,
        expected_row_count: int | None = None,
    ) -> dict[str, Any]:
        """全OOF検査を実行し、結果dictを返す。"""
        failures: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, Any] = {}

        # OOF-01: empty check (常時有効 D-03)
        if df.empty:
            raise ValueError("OOF artifact is empty (OOF-01)")

        # OOF-07: required metadata (常時有効 D-03)
        missing_required = set(profile.required_columns) - set(df.columns)
        if profile.fold_col not in df.columns:
            missing_required.add(profile.fold_col)
        if missing_required:
            raise ValueError(
                f"Missing required columns: {sorted(missing_required)} (OOF-07)"
            )

        # OOF-05: min fold count (常時有効 D-03)
        fold_count = df[profile.fold_col].nunique()
        metrics["fold_count"] = fold_count
        if fold_count < profile.min_fold_count:
            failures.append(
                f"Fold count {fold_count} < minimum {profile.min_fold_count} (OOF-05)"
            )

        # OOF-06: same race multiple fold (常時有効 D-03)
        race_fold_counts = df.groupby("race_id")[profile.fold_col].nunique()
        multi_fold_races = (race_fold_counts > 1).sum()
        metrics["same_race_multiple_fold_count"] = int(multi_fold_races)
        if multi_fold_races > 0:
            failures.append(
                f"{multi_fold_races} races appear in multiple folds (OOF-06)"
            )

        # OOF-04: row coverage (常時有効 D-03)
        if expected_row_count and expected_row_count > 0:
            coverage = len(df) / expected_row_count
            metrics["row_coverage_ratio"] = coverage
            if coverage < profile.row_coverage_threshold:
                failures.append(
                    f"Row coverage {coverage:.1%} < {profile.row_coverage_threshold:.0%} (OOF-04)"
                )

        # OOF-03: top1 hit rate / ROI (profile依存 D-04)
        if profile.score_col and profile.return_cols:
            # ... existing _validate_win_selection_oof_health pattern ...

        # OOF-02: train/valid overlap (profile依存 D-04)
        if profile.enable_train_valid_overlap:
            # ... split metadata check ...

        status = "PASS" if not failures else "FAIL"
        return {
            "status": status,
            "failures": failures,
            "warnings": warnings,
            **metrics,
        }
```

### Manifest生成パターン (D-10, D-11, XCT-08)
```python
def generate_manifest(
    self,
    df: pd.DataFrame,
    profile: OOFHealthProfile,
    artifact_hash: str,
    train_date_range: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """D-10: health manifestを生成する。"""
    schema_hash, schema_dtype_hash = _compute_schema_hashes(df)

    manifest = {
        "artifact_name": profile.artifact_name,
        "artifact_path": f"data/oof/{profile.artifact_name}.parquet",
        "artifact_hash": artifact_hash,
        "artifact_version": int(
            df["oof_artifact_version"].iloc[0]
            if "oof_artifact_version" in df.columns
            else 0
        ),
        "schema_hash": schema_hash,                   # XCT-08
        "schema_dtype_hash": schema_dtype_hash,       # D-11
        "row_count": len(df),
        "race_count": int(df["race_id"].nunique()),
        "fold_col": profile.fold_col,
        "fold_count": int(df[profile.fold_col].nunique()),
        "date_min": str(df["race_date"].min()),
        "date_max": str(df["race_date"].max()),
        "train_date_range": train_date_range,          # XCT-08
        "validator_version": self.VALIDATOR_VERSION,   # D-14
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # ... remaining D-10 fields ...
    }
    return manifest
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|-------------|-----------------|-------------|--------|
| `_validate_win_selection_oof_health()` (単一関数) | `OOFHealthValidator` (統合クラス) | Phase 37 | 全OOF検査を統一的に管理。profile駆動で新artifact追加が容易 |
| fold列なしのOOF artifact | fold列必須(D-05/D-07) | Phase 37 | OOF-05/06検査が可能に。legacy artifactはfail扱い |
| top1閾値のハードコード(定数) | artifact profile内の設定値 | Phase 37 | artifact別の柔軟な閾値設定。profile変更で再コンパイル不要 |
| OOF artifact直接読込 | manifest-first消費者側チェック | Phase 37 | 不正/古いOOFの流入を防止 |

**Deprecated/outdated:**
- `WIN_SELECTION_OOF_MAX_TOP1_HIT_RATE` / `WIN_SELECTION_OOF_MAX_TOP1_ROI` / `WIN_SELECTION_OOF_MIN_GUARD_RACES` (lines 62-64): profileに移動後、training_pipeline.pyのmodule-level定数は不要になる
- `_validate_win_selection_oof_health()`: OOFHealthValidatorに統合後、この関数はdeprecated

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `generate_ev_oof_predictions()`の戻り値を呼び出し元でDataFrameにfold列を紐付ける実装でD-05を満たせる | Architecture Patterns | 呼び出し元の構造を誤解している場合、大幅な再設計が必要 |
| A2 | `oof_predictions.parquet`のfold列名は`ability_oof_fold`でよい(D-05 CONTEXT.md記載) | Code Examples | AbilityModel以外のOOF生成器もこのartifactに寄与する場合、列名の整合性に問題が生じうる |
| A3 | 既存の`win_selection_oof_fold`列はtraining_pipeline.py line 1692で正しく設定されており、追加の変更は不要 | Common Pitfalls | 実際の値に不整合がある場合、OOF-06検査がfalse positiveを出す |

**Note:** A1-A3は低リスク。CONTEXT.mdのD-05が明確に定義しており、コードベースの確認で裏付けられている。

## Open Questions

1. **generate_ev_oof_predictions()のfold列追加アプローチ**
   - What we know: 現在3つのnumpy配列を返す。呼び出し元はtraining_pipeline.py内の複数箇所
   - What's unclear: 戻り値をtupleに拡張してfold配列も返すか、呼び出し元でDataFrame構築時にfold列を設定するか
   - Recommendation: 呼び出し元でDataFrame構築時にfold列を設定する方が影響範囲が小さい。generate_ev_oof_predictions()のシグネチャ変更は下流のfit_ev_calibration()等への影響が大きい

2. **expected_row_countの具体的な計算**
   - What we know: 学習データ行数 * (1 - 1/n_folds) を基準とする(CONTEXT.md specifics)
   - What's unclear: training_pipeline.run()内で期待行数をどう渡すか。full_features_dfの行数を基準にするか、train_dfの行数を基準にするか
   - Recommendation: validate()呼び出し時にexpected_row_count=len(full_features_df)を渡すのが最も単純

3. **source_model_hashの計算方法**
   - What we know: D-10でmanifestに含める必要がある
   - What's unclear: 学習済みモデルファイルのhashか、ソースコードのgit commit hashか
   - Recommendation: source_code_version/git_commitをmanifestフィールドに含め(D-10)、source_model_hashは学習済みモデルディレクトリのhashとする。既存のCQR params SHA256パターン(line 2533)を参考

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|-----------|------------|-----------|---------|----------|
| Python 3.11 | 全コード | ✓ | 3.11.15 | - |
| pytest | テスト | ✓ | 9.0.2 | - |
| pandas | DataFrame操作 | ✓ | 2.3.3 | - |
| numpy | 数値計算 | ✓ | >=1.26 | - |
| lightgbm | 既存依存(変更なし) | ✓ | 4.6.0 | - |
| scikit-learn | 既存依存(変更なし) | ✓ | 1.8.0 | - |
| hashlib (stdlib) | SHA256 | ✓ | 3.11 | - |

**Missing dependencies with no fallback:** なし

**Missing dependencies with fallback:** なし

Step 2.6: SKIPPED (no external dependencies beyond stdlib and existing packages)

## Validation Architecture

> workflow.nyquist_validation = false in .planning/config.json. Section skipped.

## Security Domain

> security_enforcement not explicitly set. Including for completeness.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | N/A (内部パイプライン) |
| V3 Session Management | no | N/A |
| V4 Access Control | no | N/A (学習パイプライン内) |
| V5 Input Validation | yes | OOFHealthValidator自体が入力検証。fail-fast ValueErrorで不正データの流入を防止 |
| V6 Cryptography | yes (minimal) | SHA256 hash (hashlib stdlib)。暗号鍵なし |

### Known Threat Patterns for Python ML Pipeline

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Data poisoning (crafted OOF artifact) | Tampering | artifact_hash SHA256検証で改ざん検出 |
| Legacy artifact reuse | Repudiation | D-07: fold列なしはfail。migrationコマンドでのみ受け入れ |
| Schema drift | Tampering | schema_hash/schema_dtype_hashで検出 |

## Sources

### Primary (HIGH confidence)
- Codebase: `src/pipelines/training_pipeline.py` lines 80-297, 520-560, 1690-1746 - OOF生成・検証・保存の全パターン
- Codebase: `src/models/stage1_ability_model.py` lines 350-401 - train_oof()のK-fold expanding window実装
- Codebase: `src/models/win_benter_gate.py` lines 127-190 - generate_win_oof_predictions()の配列返却パターン
- Codebase: `scripts/freeze_feature_manifest.py` - SHA256 manifest生成の確立パターン
- Codebase: `tests/test_oof_leakage.py` - train_oof()テストパターン(mock使用、DB不要)
- Codebase: `tests/test_training_pipeline.py` - 既存テストのmockパターン(patch decorator)

### Secondary (MEDIUM confidence)
- `.planning/phases/34-validation-and-manifest-update/34-CONTEXT.md` - feature manifest SHA256パターンの先行実装
- `.planning/phases/37-ev-calibration-layers/37-CONTEXT.md` - Phase 37のlocked decisions

### Tertiary (LOW confidence)
- なし (全てcodebase直接確認済み)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - 新規外部依存なし。stdlib + 既存依存のみ
- Architecture: HIGH - CONTEXT.mdのD-01~D-15が詳細に定義済み。既存コードパターンと整合
- Pitfalls: HIGH - 既存コードベースの直接確認で特定。generate_ev_oof_predictions()の配列返却が主要な設計判断点

**Research date:** 2026-05-27
**Valid until:** 2026-06-27 (stable domain: internal validation infrastructure, no external API dependency)
