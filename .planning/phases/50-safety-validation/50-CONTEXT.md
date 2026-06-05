# Phase 50: Safety & Validation - Context

**Gathered:** 2026-06-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 47-49で実装した全トラック条件特徴量（含水率・クッション値由来）の安全性をCI検証し、BT ROI 97%+を確認してデプロイ可能状態にする。Feature Routing Audit・POST_RACE CI検証・BT ROI検証・IC評価・WF NaN率検証の5つの安全基盤を完了する。

**In scope:** REG-02 (Feature Routing Audit拡張 + surface-aware検証), REG-03 (POST_RACE 3層CI検証), VLD-01 (段階BT ROI判定), VLD-02 (IC評価), VLD-03 (WF Fold0 NaN率検証)
**Out of scope:** 新特徴量の追加実装, MAWC修正 (v2.4+), モデルアーキテクチャ変更, レジーム依存ロジック変更, T4-03/T4-02のMarketModel/RaceQualityScreener追加 (別Phase ablation)

</domain>

<decisions>
## Implementation Decisions

### BT ROI判定基準と失敗時対応 (VLD-01)

- **D-01:** 段階判定フロー。①2025年単独BT ROI >= 97%を第一関門 → PASS時のみ ②2024+2025通算ROI >= 97% かつ各年ROI >= 90%を最終判定。2025年単独FAIL時は過去年度検証へ進まず、特徴量・ルーティング・閾値見直しへ
- **D-02:** 診断ベース再試行1回のみ。IC符号反転・高NaN率・MarketModel支配・routing違反など構造的異常のみを修正対象とする。ROIのみを根拠とした閾値調整・特徴量選択は明示的に禁止。再試行でも2025年ROI < 97%なら `not_deployable` で閉じる
- **D-03:** 二段階BT評価。一次判定: `--ensemble`のみ。v2.2 baseline vs v2.3 candidateを同一フラグ・同一期間・同一seedで比較し、特徴量追加の純粋効果を測定。strategy-manifestは不使用（戦略効果混入を避ける）。二次確認: 一次PASS後のみ `--min-win-ev 1.03 --min-win-odds 3.0` 付きで実運用ROI確認

### Feature Routing Audit拡張方針 (REG-02)

- **D-04:** 既存registry拡張 + surface-awareデータ検証。Phase 42のaudit registry (50+28禁止特徴量) に新トラック条件特徴量を追加登録。既存 `run_feature_routing_audit.py` で一括検証
- **D-05:** Surface-aware CIテストを追加。dirt系特徴量が芝行でNaN、turf系特徴量がダート行でNaNになることをデータレベルで確認。submodel共通FEATURE_COLSへの登録自体は許容（LightGBMネイティブNaN対応）
- **D-06:** Phase 48/49の外科的ルーティング（6モデル登録: AbilityModel, Win/Place/WideTwoStage, EVCorrection, PlaceEVCorrection / 4モデル除外: MarketModel, RaceQualityScreener, RegimeDetector, ConformalEVModel）をそのまま維持。T4-03異常値フラグのMarketModel追加・T4-02のRaceQualityScreener追加は行わない

### IC評価スコープと閾値 (VLD-02)

- **D-07:** Phase 48/49の全新特徴量を評価対象とする。各列についてOOFベースの単変量IC・C直交IC・欠損率・有効サンプル数を出力。Tier別（T1/T2/T3/T4）およびhorse-level/race-level別にも集計
- **D-08:** カテゴリ列（sire_x_cushion_band等）は数値化せず、カテゴリ別ターゲット統計として別評価。Pearson/Spearman ICへの無理な変換はしない
- **D-09:** ICは情報提供目的であり、個別閾値によるFAILは設けない。abs(C直交IC) >= 0.005をsignalあり、未満をweakと分類。fold間符号反転または有効サンプル不足を診断対象としてフラグ付け。最終PASS/FAIL判定はROI + Routing Auditを主基準とする

### WF Fold0 NaN許容閾値と対応 (VLD-03)

- **D-10:** Surface-aware NaN率判定。芝系特徴量はWF Fold0の芝レース行のみを分母とする（ダート行の仕様上NaNは集計対象外）
- **D-11:** 3段階閾値（turf_cushion元データNaN率基準）: < 30% → PASS、30-50% → WARN、>= 50% → FAIL
- **D-12:** NaN原因分離報告。元データturf_cushion欠損によるNaNと、派生処理（最低出走数不足・track×month統計不足等）によるNaNを別々に集計・報告
- **D-13:** NaN対応: WARNは記録のみ措置なし。FAIL(元データ>= 50%)は芝系特徴量群全体を除外候補。FAIL(派生処理>= 50%)は当該特徴量のみ除外/修正。除外はD-02の診断ベース再試行1回の一部として実施
- **D-14:** 学習開始時期はv2.2 baselineとの比較条件を維持するため変更しない

### REG-03: POST_RACE CI検証

- **D-15:** 新トラック条件特徴量のPOST_RACE分類を3層CI検証(whitelist/forbidden/manual)で確認。含水率/クッション値由来特徴量はレース当日JRA発表値（締切前利用可能情報）としてPOST_RACE_COLSに含めないことをCIで担保。Phase 47 D-11で既にCI test追加済み、Phase 50ではPhase 48/49追加特徴量を含めて再確認

### Claude's Discretion

- Feature Routing Audit registryへの新特徴量追加方法（FORBIDDEN_CALIBRATOR / FORBIDDEN_RANKER リスト拡張等の既存パターンに従う）
- Surface-aware CI テストの具体的な実装（turf/dirt行のNaN率検証ロジック）
- IC評価の実行詳細（run_ic_eval.pyの出力解析、集計スクリプト等）
- NaN原因分離報告の出力フォーマット
- 診断レポートのフォーマットと出力先
- BT結果の比較分析スクリプト（v2.2 baseline vs v2.3 candidate）
- テスト構成・テストケースの詳細設計（既存パターンに従う）

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Audit & CI Infrastructure
- `src/audit/feature_routing_registry.py` — Phase 42のFeature Routing Audit基盤。50+28禁止特徴量registry。Phase 50で新トラック条件特徴量を追加
- `scripts/run_feature_routing_audit.py` — Feature Routing Audit CLIスクリプト
- `src/domain/types.py` — POST_RACE_COLS定義（41列）。含水率/クッション値は含めない。3層CI検証の基準
- `src/backtest/deployment_gates.py` — DeploymentGateEvaluator。frozen GatePolicy。Phase 50ではreport-only使用
- `src/validation/oof_health_validator.py` — OOFHealthValidator。fail-fast + SHA256 manifest

### Backtest & Validation
- `scripts/run_backtest.py` — BT CLIスクリプト。--years, --ensemble, --min-win-ev, --min-win-odds フラグ
- `scripts/run_ic_eval.py` — IC評価CLIスクリプト。OOF予測ベース
- `src/backtest/engine.py` — BacktestEngine。WF Fold定義、feature pre-computation
- `src/backtest/race_predictor.py` — RacePredictor。推論パイプライン

### Track Condition Features (Phase 48/49 artifacts)
- `src/features/track_condition_features.py` — T1/T2/T4-01/T4-03/T4-04特徴量。TRACK_CONDITION_COLS + TRACK_DERIVED_COLS + RACE_CONDITION_COLS
- `src/features/horse_track_aptitude.py` — T3馬個体適性precompute。APTITUDE_COLS
- `data/raw/track_conditions.parquet` — 23,259行の含水率/クッション値データ
- `data/raw/horse_track_aptitude.parquet` — T3馬個体適性データ
- `config/settings.yaml` — track_conditionセクション（閾値設定）

### Prior Phase Context
- `.planning/phases/48-core-edge-features/48-CONTEXT.md` — Phase 48外科的ルーティング決定(D-04~D-06)、track_statsパターン
- `.planning/phases/49-derived-higher-order-features/49-CONTEXT.md` — Phase 49派生特徴量決定(D-01~D-26)、track_month_statsパターン
- `.planning/phases/47-etl-data-pipeline/47-CONTEXT.md` — Phase 47 track_conditions.parquet設計

### Requirements & Configuration
- `.planning/REQUIREMENTS.md` — REG-02, REG-03, VLD-01, VLD-02, VLD-03 要件定義
- `config/backtest_config.yaml` — WF parameters, holdout period, pass criteria
- `.planning/STATE.md` — Phase 49完了状態、blockers/concerns

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/audit/feature_routing_registry.py`: FORBIDDEN_CALIBRATOR (50) / FORBIDDEN_RANKER (28) リスト。新特徴量のallow/denyルールを追加可能
- `scripts/run_feature_routing_audit.py`: 既存audit CLI。--output-dir でJSON + Markdown出力
- `scripts/run_ic_eval.py`: OOF予測Parquetを受け取りIC/B差分/C直交/E Incrementalを計算
- `scripts/run_backtest.py --years 2025 --ensemble`: 2025年単独BT。--train-window 4で2021-2024学習
- `POST_RACE_COLS CI test`: `tests/test_track_condition_data.py` に既存CI test (Phase 47 D-11)。Phase 48/49特徴量も含めて拡張
- `DeploymentGateEvaluator`: 4-gate評価 (probability quality, bet count, reproducibility, diagnostics)

### Established Patterns
- Feature Routing Audit: registry拡張 + CLI実行 + JSON/Markdown レポート出力
- 3層POST_RACE CI検証: whitelist (FEATURE_COLS) / forbidden (POST_RACE_COLS) / manual review
- BT段階判定: v2.2 baseline比較 → v2.3 candidate比較の差分分析
- 診断ベース再試行: 構造的異常(IC符号反転等)のみ修正、ROI根拠の調整禁止

### Integration Points
- `run_feature_routing_audit.py --output-dir data/audit`: audit実行エントリポイント
- `run_backtest.py --years 2025 --ensemble`: 一次BT判定
- `run_backtest.py --years 2024 2025 --ensemble`: 二次BT判定(一次PASS後)
- `run_ic_eval.py data/oof/oof_predictions.parquet --output data/baseline/ic_baseline.json`: IC評価
- `data/backtest/bt_2025_*.csv`: BT結果出力
- `data/audit/`: audit結果出力ディレクトリ

### Key Validation Flow
```
Phase 50 Validation Pipeline:
  1. REG-02: run_feature_routing_audit.py → PASS/FAIL
  2. REG-03: POST_RACE CI test → PASS/FAIL
  3. VLD-03: WF Fold0 NaN率チェック → PASS/WARN/FAIL
  4. VLD-01: BT一次(2025 --ensemble) → ROI >= 97%?
     ├─ YES → BT二次(2024+2025) → 通算>=97% & 各年>=90%?
     │         ├─ YES → VLD-02: IC評価(情報提供) → 二次safety filter確認 → DEPLOY
     │         └─ NO  → not_deployable
     └─ NO  → 診断(IC/NaN/Audit/routing) → 構造的異常あれば1回再試行
               ├─ 再試行ROI >= 97% → BT二次へ
               └─ 再試行ROI < 97% → not_deployable
```

</code_context>

<specifics>
## Specific Ideas

- 2025年単独BTは計算コスト~41分/年（CLAUDE.mdより）。段階判定により不適格構成での2024年BT(~41分)をスキップ可能
- Surface-aware NaN検証は、芝レースのみを分母とすることで正確なNaN率を測定。2020年1-8月の芝レース(推定~17% of Fold0)がNaN源
- ICの方向性は特徴量定義に依存: kickback_risk_score等は負方向が期待値。「正方向固定」は不適切
- v2.2 baseline vs v2.3 candidateの同一条件比較が純粋効果測定の鍵。seed固定で再現性確保
- strategy-manifestは戦略最適化パラメータが含まれるため、特徴量効果の純粋測定には不適

</specifics>

<deferred>
## Deferred Ideas

- T4-03異常値フラグのMarketModel追加検証 — 別Phase ablation対象
- T4-02 race-level集約のRaceQualityScreener追加検証 — 別Phase ablation対象
- Conservative MAWC redesign / selective interaction experiment — v2.4+
- デプロイゲート自動判定 (DEP-01) — v2.4+
- Optuna 19次元パラメータ最適化 (DEP-02) — v2.4+

</deferred>

---

*Phase: 50-Safety & Validation*
*Context gathered: 2026-06-05*
