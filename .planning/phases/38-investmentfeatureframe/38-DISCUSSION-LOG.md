# Phase 38: InvestmentFeatureFrame - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-27
**Phase:** 38-InvestmentFeatureFrame
**Areas discussed:** フェーズ全体像の定義, CAL-01~05の取扱い, BT検証とROI目標, 投資特徴量フレーム設計

---

## フェーズ全体像の定義

### Phase 38のスコープ

| Option | Description | Selected |
|--------|-------------|----------|
| 全て統合 | CAL + 投資フレーム + BT検証を全てPhase 38に含める | |
| 検証のみ(CAL除外) | 投資フレーム + BT検証のみ。CALはv2.1+先送り | |
| 2フェーズ分割 | Phase 38a=CAL, 38b=投資フレーム+検証 | |

**User's choice:** Phase 38 scope is InvestmentFeatureFrame only. CAL-01~05 go to Phase 39. Ranker to Phase 40. No ROI threshold gate. Build 80-150 OOF-safe investment features with dual mode (train/infer).

### 特徴量カテゴリ構成

| Option | Description | Selected |
|--------|-------------|----------|
| 全9カテゴリ実装 | model_prob ~ uncertainty 全9カテゴリ。各8-17列 | ✓ |
| コア5カテゴリのみ | model_prob + market_prob + gap + odds_band + uncertainty (40-60列) | |
| 既存特徴量の再構成のみ | FeatureEngine出力をそのまま再利用 | |

**User's choice:** 全9カテゴリ実装。Schema-stable with graceful degradation. Not a passthrough of FeatureEngine output.

### デュアルモード設計

| Option | Description | Selected |
|--------|-------------|----------|
| 単一メソッド + mode引数 | build_frame(df, mode="train"|"infer") | ✓ |
| 別メソッド分離 | build_train_frame() + build_inference_frame() | |
| 自動判定 | source columnsの有無で自動判定 | |

**User's choice:** Single build_frame(df, mode=Literal["train","infer"]) API. Thin convenience wrappers. No auto-detect. Output schema identical. Mode controls source priority only.

### キャッシュ戦略

| Option | Description | Selected |
|--------|-------------|----------|
| Parquetキャッシュ | Source artifact hashをキーにしたParquet + sidecar manifest | ✓ |
| メモリキャッシュのみ | LRU cache。再起動で消失 | |
| キャッシュなし | 毎回再計算 | |

**User's choice:** Parquet cache keyed by source artifact identity. Sidecar manifest JSON. Deterministic output.

---

## CAL-01~05の取扱い

### CAL行き先

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 39に移行 | 非レジームCAL部分をPhase 39に | ✓ |
| v2.1+に延期 | マイルストーンスコープ外 | |
| 部分含める | 一部(CAL-03/04等)をPhase 38に | |

**User's choice:** Non-regime CAL → Phase 39 as segment conditioning in MarketAwareWinCalibrator. Regime propagation OUT OF SCOPE for v2.0.

### 要件更新タイミング

| Option | Description | Selected |
|--------|-------------|----------|
| CONTEXT.mdに記録のみ | Phase 39/40計画時に更新 | |
| 今ROADMAP/要件を更新 | ROADMAP.mdとREQUIREMENTS.mdを即時更新 | ✓ |

**User's choice:** Update ROADMAP.md and REQUIREMENTS.md now. Record in CONTEXT.md for traceability.

---

## BT検証とROI目標

### BT/ROIの扱い

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 38から完全除外 | BT実行もROI確認もしない | ✓ |
| Smoke testのみ | 100レース程度でフレーム実用性確認(ROI閾値なし) | |
| 全量BT + ROI確認 | BT 2024全量実行 + ROI >= 97.8%確認 | |

**User's choice:** No ROI gate. No full BT. Smoke test allowed (no ROI threshold). Deferred ROI check retired or moved to Phase 39/40.

### VAL-01~06配分

| Option | Description | Selected |
|--------|-------------|----------|
| VAL-01のみ残す | 漏洩テストのみPhase 38 | |
| 全てPhase 39/40に移行 | Phase 38はスキーマテストのみ | |
| VAL-01 + VAL-06を含める | 漏洩テスト + Manifest凍結 | |

**User's choice:** VAL-01 scoped to InvestmentFeatureFrame leakage. New artifact manifest requirement (IFF-07). VAL-02~05 → Phase 39/40. VAL-06 v1.8 freeze → retired.

---

## 投資特徴量フレーム設計

### 列数配分

| Option | Description | Selected |
|--------|-------------|----------|
| 9カテゴリ均等配分 | 各8-17列、計120列程度 | |
| 信号密度重視の非均等 | gap/relative/uncertainty厚く | |
| Claude裁量 | 必須列定義、残りは実装時決定 | |

**User's choice:** Required-core + optional-extension design. 90-130 columns initially, hard upper 150. Signal density prioritization (gap, race_relative, uncertainty, ability_form get more). Every feature must have metadata.

### Source column mapping

| Option | Description | Selected |
|--------|-------------|----------|
| Schema registry dict | 規約dictでtrain/infer sourceを定義 | |
| TypedDict dataclass | 型安全だがコード量増 | |
| YAML設定ファイル | 外部変更可能だが保守が必要 | |

**User's choice:** In-code typed schema registry using frozen dataclass (InvestmentFeatureSpec). Code is source of truth.

### モジュール配置

| Option | Description | Selected |
|--------|-------------|----------|
| 新規 src/investment/ | feature_frame.py, schema_registry.py, manifest.py, cache.py, leakage.py | ✓ |
| src/features/ 内 | FeatureEngine拡張として扱う | |
| src/models/ 内 | モデルレイヤーとして扱う | |

**User's choice:** New src/investment/ package. Independent from FeatureEngine and models.

---

## Claude's Discretion

- 各カテゴリの具体的特徴量選定(8-12列中のどの列か)
- 派生特徴量の計算式(race-relative, uncertainty等)
- キャッシュinvalidationロジック
- sidecar manifest JSONの完全スキーマ
- テストケース設計
- ビルダー内部アーキテクチャ
- frame_builder.py vs feature_frame.pyのファイル名
- OOF health manifestとの統合インターフェース
- Phase 37 OOFHealthValidatorとの接続方法

## Deferred Ideas

- 人気帯キャリブレーション (CAL-01~05) → Phase 39 MarketAwareWinCalibratorセグメント条件付け
- レジーム伝播 → v2.0全体でスコープ外
- Race-Level Ranker → Phase 40
- BT 2024 ROI検証 → Phase 39/40統合検証信号
- 芝IC b_difference (VAL-02) → Phase 39/40診断
- 芝pop 4-12 ratio (VAL-03) → Phase 39/40 shadow comparison
- ROI 100%超え (VAL-04) → Phase 39/40統合結果
- Turf conservative ROI (VAL-05) → 廃止またはPhase 39/40
- v1.8 Manifest凍結 (VAL-06) → 廃止、v2.0 artifact manifestに置き換え
