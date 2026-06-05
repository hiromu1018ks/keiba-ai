# Phase 50: Safety & Validation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-05
**Phase:** 50-Safety & Validation
**Areas discussed:** BT ROI判定基準と失敗時対応, Feature Routing Audit拡張方針, IC評価スコープと閾値, WF Fold0 NaN許容閾値と対応

---

## BT ROI判定基準と失敗時対応

### ROI測定方法

| Option | Description | Selected |
|--------|-------------|----------|
| 通算ROI | 2024+2025を通算した単一ROIで判定。サンプルサイズ最大 | |
| 各年単独ROI | 2024年ROI >= 97% AND 2025年ROI >= 97% | |
| 通算主+年別最低ライン | 通算ROI >= 97% + 各年ROI >= 90% | |

**User's choice:** カスタム段階判定 — ①2025年単独BT ROI>=97% → PASSなら②2024+2025通算ROI>=97%かつ各年>=90%。2025単独FAIL時は過去年度検証へ進まず見直し
**Notes:** 高コストな過去BTを有望な構成だけに限定しつつ、2025年への過適合も2024年で検出する設計

### ROI未達時の対応

| Option | Description | Selected |
|--------|-------------|----------|
| 除外イテレーション(max3) | 失敗構成で分析し問題特徴量除外して再BT、最大3回 | |
| 即FAIL(v2.2パターン) | ROI < 97%なら即FAIL、not_deployableで閉じる | |
| 1回のみ再試行 | 1回だけ試み、未達ならnot_deployable | |

**User's choice:** 診断ベース再試行1回のみ。IC符号反転・高NaN率・MarketModel支配・routing違反など構造的異常のみ修正。ROI根拠の調整禁止。再試行FAIL→not_deployable
**Notes:** v2.2はnot_deployableで閉じたが、今回は新特徴量初回なので調整余地あり。ただしROIだけの根拠では調整しないという明確な原則

### BT実行フラグ

| Option | Description | Selected |
|--------|-------------|----------|
| --ensembleのみ | manifest/safety filterなし。純粋な特徴量効果 | |
| --ensemble + manifest | 既存最適化パラメータ使用 | |
| --ensemble + safety filter | --min-win-ev 1.03 --min-win-odds 3.0 | |

**User's choice:** 二段階評価。一次: --ensembleのみ(v2.2 baseline vs v2.3 candidate同一条件比較)。二次: 一次PASS後のみsafety filter付きで実運用ROI確認。strategy-manifestはVLD-01主判定に不使用
**Notes:** 純粋効果測定と実運用確認を明確分離

---

## Feature Routing Audit拡張方針

### Audit統合方法

| Option | Description | Selected |
|--------|-------------|----------|
| 既存registry拡張 | Phase 42の50+28禁止特徴量に新特徴量追加 | |
| 専用audit別作成 | トラック条件専用スクリプト | |
| registry拡張 + surface-aware検証 | 既存拡張 + dirt/turf NaN分布CI検証 | ✓ |

**User's choice:** registry拡張 + surface-aware検証。既存audit再利用しつつ新特徴量特有の誤配線も検出。surface-awareはFEATURE_COLS登録禁止ではなく実データ適用範囲の確認。dirt系→芝行NaN、turf系→ダート行NaNをデータレベルCIで確認。submodel共通FEATURE_COLS登録自体は許容

### 外科的ルーティング(D-24/D-25保留判断)

| Option | Description | Selected |
|--------|-------------|----------|
| 全除外維持 | Phase 48/49の6登録/4除外をそのまま | ✓ |
| BT結果で判断 | ROI結果を見てから追加判断 | |
| Audit時支配チェック | MarketModel支配なければ追加許可 | |

**User's choice:** Phase 50では全除外維持。再試行時もrouting違反の診断根拠がない限り追加しない。追加検証は別Phase ablation対象

---

## IC評価スコープと閾値

### IC評価対象

| Option | Description | Selected |
|--------|-------------|----------|
| 全新特徴量 | TRACK_CONDITION_COLS + TRACK_DERIVED_COLS + RACE_CONDITION_COLS + T3 | ✓ |
| T1/T2のみ | 8列のみ。T3/T4は間接評価 | |
| Tier別代表列のみ | 各Tierから1-2列 | |

**User's choice:** 全新特徴量を評価。各列OOFベース単変量IC・C直交IC・欠損率・有効サンプル数。Tier別・horse/race-level別集計。カテゴリ列は数値化せずカテゴリ別ターゲット統計で別評価

### IC閾値

| Option | Description | Selected |
|--------|-------------|----------|
| 微弱正の方向性 | IC >= 0.01、C直交IC >= 0.005 | |
| 符号反転のみFAIL | IC負のみFAIL、それ以外保留 | |
| 閾値なし(情報提供のみ) | 明示的閾値設けず | |

**User's choice:** ICは情報提供目的(個別FAILなし)。abs(C直交IC)>=0.005をsignal分類。fold間符号反転/有効サンプル不足を診断対象。最終判定はROI+Auditが主基準。ICの方向性は特徴量定義に依存するため正方向固定閾値は不適切

---

## WF Fold0 NaN許容閾値と対応

### NaN閾値

| Option | Description | Selected |
|--------|-------------|----------|
| < 30%許容 | 2020年1-8月は48ヶ月中~17% | |
| < 50%許容 | NaN過半でも学習可能 | |
| 閾値なし(実測で判断) | NaN率レポートのみ | |

**User's choice:** Surface-aware 3段階。芝レース行のみ分母。turf_cushion元データNaN率: <30% PASS、30-50% WARN、>=50% FAIL。派生特徴量は元値欠損NaNと統計不足NaNを分離報告。ダート行仕様上NaNは集計対象外

### NaN WARN/FAIL時対応

| Option | Description | Selected |
|--------|-------------|----------|
| WARN記録/FAIL除外 | FAIL時当該特徴量除外して再BT | |
| 情報記録のみ | ROI結果で総合判断 | |
| FAIL時は学習開始延期 | 学習開始を2021年に延期 | |

**User's choice:** WARN=記録のみ。FAIL=原因別対応。元データ>=50%→芝系除外候補。派生処理>=50%→当該のみ除外。除外は診断ベース再試行1回の一部。学習開始時期はbaseline比較条件維持のため変更しない

---

## Claude's Discretion

- Feature Routing Audit registryへの新特徴量追加方法（既存パターンに従う）
- Surface-aware CI テストの具体的実装
- IC評価実行詳細（run_ic_eval.py出力解析、集計スクリプト）
- NaN原因分離報告フォーマット
- 診断レポートフォーマットと出力先
- BT結果比較分析スクリプト
- テスト構成・テストケース詳細

## Deferred Ideas

- T4-03 MarketModel追加検証 — 別Phase ablation対象
- T4-02 RaceQualityScreener追加検証 — 別Phase ablation対象
