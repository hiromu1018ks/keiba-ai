# Phase 19: EV推定キャリブレーション - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-07
**Phase:** 19-EV推定キャリブレーション
**Areas discussed:** Isotonic適用方式, OOF生成・保存設計, パイプライン統合

---

## Isotonic適用方式

| Option | Description | Selected |
|--------|-------------|----------|
| 既存P×Eの後にIsotonicを適用 | ev_win_correctedを入力にIsotonic→ev_win_calibrated。二重補成 | ✓ |
| Isotonicに完全置き換え | P補正・E補正LightGBM廃止、Isotonicのみ | |
| オッズ帯別に切り替え | 低オッズはP×Eのみ、高オッズはIsotonic追加 | |

**User's choice:** ベストプラクティスを追求 (Recommended = 既存P×Eの後にIsotonicを適用)

| Option | Description | Selected |
|--------|-------------|----------|
| サーフェス別(芝/ダート) | 芝とダートで独立Isotonic | ✓ |
| 全体単一 | 単一IsotonicRegression | |
| サーフェス×オッズバンド | 最大8パターン(データ不足リスク) | |

**User's choice:** サーフェス別(芝/ダート) (Recommended)

| Option | Description | Selected |
|--------|-------------|----------|
| EV→actual_return直接 | X=ev_win_corrected(OOF), y=actual_return(OOF) | ✓ |
| P/E別々にIsotonic適用 | 2段階Isotonic(サンプル効率悪い) | |
| EV→actual_return (capping付き) | capping付きで過制御防止 | |

**User's choice:** ベストプラクティスを追求 (Recommended = EV→actual_return直接)

| Option | Description | Selected |
|--------|-------------|----------|
| y_min=0 + clip | 標準的で安全 | ✓ |
| y_min=0 + NaN | 学習範囲外はNaN | |
| y_min=0 + y_max=cap | 上限設定(閾値チューニング必要) | |

**User's choice:** ベストプラクティスを追求 (Recommended = y_min=0 + clip)

**Notes:** ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針を示した。

---

## OOF生成・保存設計

| Option | Description | Selected |
|--------|-------------|----------|
| 学習パイプライン内で生成 | TrainingPipelineV5._train_submodel()内でOOF EV生成 | ✓ |
| 独立スクリプトで生成 | 別ステップでOOF生成 | |
| run_train.py --oof フラグ | フラグ制御でOOF生成 | |

**User's choice:** ベストプラクティスを追求 (Recommended = 学習パイプライン内で生成)

| Option | Description | Selected |
|--------|-------------|----------|
| SubmodelSetに追加 | 既存パターンと統一 | ✓ |
| 独立ファイルとして保存 | 管理が分散 | |
| EVCorrectionModel内部に格納 | 責務が増える | |

**User's choice:** ベストプラクティスを追求 (Recommended = SubmodelSetに追加)

| Option | Description | Selected |
|--------|-------------|----------|
| ロードのみ(再学習なし) | テスト期間では学習済みIsotonicを適用のみ | ✓ |
| バックテストでも再学習 | テスト期間中に再学習(データリーク) | |

**User's choice:** ベストプラクティスを追求 (Recommended = ロードのみ)

**Notes:** バックテストの学習フェーズ(pipeline.run())でIsotonicは毎回再学習されるため、テスト期間ではロードして適用するのみが正しい。

---

## パイプライン統合

| Option | Description | Selected |
|--------|-------------|----------|
| correct_ev()内で適用 | ev_win_corrected→Isotonic→ev_win_calibrated。呼び出し側変更不要 | ✓ |
| 独立クラスで適用 | EVIsotonicCalibrator新規作成 | |
| RacePredictor内で適用 | 推論側で制御(分散リスク) | |

**User's choice:** ベストプラクティスを追求 (Recommended = correct_ev()内で適用)

| Option | Description | Selected |
|--------|-------------|----------|
| _train_submodel()内でOOF生成 | 既存学習フローにK-foldループ追加 | ✓ |
| 学習後に別ステップで生成 | 既存フロー変更なし(データ再読込必要) | |

**User's choice:** ベストプラクティスを追求 (Recommended = _train_submodel()内でOOF生成)

**Notes:** EVC-03要件「EVCorrectionModel統合」に合致する統合方式を選択。

---

## Claude's Discretion

- オッズバンド別補正の具体的な手法(EVC-02)
- _train_submodel()内のOOF K-fold実装詳細(分割数、時系列ソート)
- SubmodelSetへのIsotonicフィールド命名規則と保存形式
- ModelLoader拡張の実装詳細
- correct_ev()へのIsotonic統合の具体的な実装
- テストのfixtureデータとモック構成

## Deferred Ideas

None — discussion stayed within phase scope
