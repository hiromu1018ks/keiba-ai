# 特徴量 Parquet 出力 & PT/BT 乖離修正

**日付:** 2026-04-12
**対象ブランチ:** main (1e27dd6 → 3db1d8c)
**設計書:** `docs/superpowers/specs/2026-04-12-feature-export-design.md`
**計画書:** `docs/superpowers/plans/2026-04-12-feature-export-plan.md`

---

## 概要

バックテスト (BT) とペーパートレード (PT) の成績乖離調査のため、両パイプラインで計算した全馬の特徴量+予測値を horse-level parquet として出力できるようにした。同時に、PT と BT のパイプライン差分を修正し、挙動を統合した。14コミット。

## セッションで実施したこと

### 1. 特徴量 Parquet 出力 (6コミット)

`DiagnosticLogger` を拡張し、`result_df` の全行を parquet として保存する機能を追加。

| コミット | 内容 |
|---------|------|
| `62603af` | `log_horse_features()` メソッド追加 |
| `25d655c` | `save()` に parquet 出力追加 |
| `a080cc3` | list/dict 型のネスト列除外フィルタ |
| `6c05547` | BacktestEngine に追加 (2箇所) |
| `dd331c6` | PT predict モードに追加 (2箇所) |
| `8b5e984` | PT diagnose モードに追加 (2箇所) |

**出力ファイル:**

| ソース | 出力先 |
|--------|--------|
| バックテスト | `data/backtest/bt_{year}_horse_features.parquet` |
| PT predict | `data/paper_trading/diag_{YYYYMMDD}_horse_features.parquet` |
| PT diagnose | `data/paper_trading/diag_parquet_{start}_{end}_horse_features.parquet` |

**仕様:** 1行=1頭、~270列 (特徴量+予測値+EveryDB2生列)。`is_bet` は含まない (診断CSV側にあり)。

### 2. PT/BT パイプライン統合 (5コミット)

PT と BT でパイプラインの挙動が異なっていた点を修正。Subagent-Driven Development で実装 (各タスク → 仕様レビュー → 品質レビュー → 修正 → 再レビュー)。

| コミット | 内容 | 影響 |
|---------|------|------|
| `3d6ee5f` | `compute_odds_dynamics` の切り詰めを無条件化 | PT で odds 時系列が短い場合の fallback |
| `56738bd` | PT に POST_RACE 列の DROP を追加 | BT と同等のリーク防御 |
| `6130161` | PT に JRA フィルタを追加 (NAR除外) | BT と同等のフィルタリング |
| `0c927d8` | ruff format 修正 | — |
| `3db1d8c` | pandas import を TYPE_CHECKING 外に移動 | `_apply_jra_filter` の NameError 修正 |

**開発プロセス (Subagent-Driven Development):**

1. **Worktree 分離:** `worktree-pt-bt-divergence-fix` ブランチで作業 → main にマージ
2. **3タスクを順次実行:** 各タスクは implementer → spec reviewer → code quality reviewer の順で検証
3. **コード品質レビューでの指摘と修正:**
   - Task 1: テストが実際の切り詰めを検証していなかった → spike データで動作検証に改善、parametrize で統合
   - Task 2: 3箇所の copy-paste → `_drop_post_race_cols()` ヘルパー抽出、命名を UPPER_CASE に統一
   - Task 3: 3箇所の copy-paste → `_apply_jra_filter()` ヘルパー抽出
4. **実行時エラー:** `pd` が `TYPE_CHECKING` 内にあったため `_apply_jra_filter` で NameError → import を実行時にも有効に移動

**新規テスト (8件):**

| テスト | ファイル | 検証内容 |
|--------|---------|---------|
| `test_truncation_always_applies[100]` | `test_odds_dynamics_fix.py` | 100ポイントでも切り詰め発動 |
| `test_truncation_always_applies[200]` | 同上 | 200ポイントでも切り詰め発動 |
| `test_truncation_limit_60_points` | 同上 | spike データで60ポイント制限を動作検証 |
| `test_post_race_cols_removed` | `test_paper_trading_guards.py` | POST_RACE 列が正しく DROP される |
| `test_post_race_cols_missing_no_error` | 同上 | 列不在でもエラーにならない |
| `test_jra_filter_removes_nar` | 同上 | jyocd >= 30 が除外される |
| `test_jra_filter_preserves_all_jra` | 同上 | jyocd 1-10 が全て保持される |
| `test_jra_filter_handles_missing_jyocd` | 同上 | jyocd 列不在でスキップ |

### 3. その他 (3コミット)

| コミット | 内容 |
|---------|------|
| `8a53195` | `save_year_parquet` の merge 型不一致修正 |
| `79d95a6` | README に出力ファイル情報を反映 |
| (複数) | 設計書・計画書の作成とレビュー |

### 4. バックテスト実行 (2025年テスト)

```
学習: 2021-2024 / テスト: 2025 / flat ¥100 / --ensemble
ROI: 216.6% / 利益: +¥240,030 / DD: 0.8% / 2,058ベット
特徴量出力: data/backtest/bt_2025_horse_features.parquet (46,499行, 16.8MB)
```

### 5. ペーパートレード実行 (4/4, 4/5, 4/11, 4/12) — 修正前の成績

4日分のペーパートレードを実行し、特徴量 parquet の出力を確認。以下は**パイプライン修正前**の成績 (odds_drop_rate バグ・POST_RACE DROP 欠落・JRAフィルタ欠落の状態)。

**全体:**
- ROI: **57.1%** (大幅赤字) / 的中率: **12.8%** (20/156) / 総投資: ¥15,600 / 総払戻: ¥8,910
- 対比 BT: ROI 216.6%, 的中率 48.2% — 乖離の主因は `compute_odds_dynamics` の条件付き切り詰め

**日別:**

| 日付 | ベット数 | 投資 | 払戻 | ROI | 的中 |
|------|---------|------|------|-----|------|
| 4/4 | 37 | ¥3,700 | ¥1,370 | 37.0% | 5 |
| 4/5 | 27 | ¥2,700 | ¥1,830 | 67.8% | 3 |
| 4/11 | 53 | ¥5,300 | ¥3,740 | 70.6% | 8 |
| 4/12 | 39 | ¥3,900 | ¥1,970 | 50.5% | 4 |

**馬場別:**

| 馬場 | ベット数 | 投資 | 払戻 | ROI | 的中 |
|------|---------|------|------|-----|------|
| dirt | 82 | ¥8,200 | ¥5,230 | 63.8% | 12 |
| turf | 74 | ¥7,400 | ¥3,680 | 49.7% | 8 |

**出力ファイル:**

| 日付 | horse_features | サイズ |
|------|---------------|--------|
| 20260404 | diag_20260404_horse_features.parquet | 30KB |
| 20260405 | diag_20260405_horse_features.parquet | 313KB |
| 20260411 | diag_20260411_horse_features.parquet | 374KB |
| 20260412 | diag_20260412_horse_features.parquet | 374KB |

---

## 既知の制約

- **JRA フィルタ**: PT predict/diagnose に NAR 除外フィルタを追加したが、`_apply_jra_filter` の pandas import で NameError が発生 → 修正済み (`3db1d8c`)
- **`_run_dry_run` はスコープ外**: DiagnosticLogger を使用しないため特徴量出力なし
- **`is_bet` 列不在**: 特徴量 parquet には含まれない。診断 CSV (`*_horse_diagnostics.csv`) と JOIN が必要
- **`p_place_pred_corrected` 列不在**: ペーパートレードの result_df に補正後の p/e 列が含まれていない (BT 側との比較時に注意)

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `src/backtest/diagnostic_logger.py` | `feature_records`, `log_horse_features()`, parquet 出力 |
| `src/backtest/engine.py` | `log_horse_features()` 呼び出し 2箇所 |
| `scripts/run_paper_trading.py` | `log_horse_features()` 呼び出し 4箇所 + `_drop_post_race_cols()` ヘルパー + `_apply_jra_filter()` ヘルパー + pandas import 修正 |
| `tests/test_diagnostic_logger.py` | 4テスト追加 (計9テスト) |
| `tests/test_odds_dynamics_fix.py` | 新規作成: 切り詰め検証3テスト |
| `tests/test_paper_trading_guards.py` | 新規作成: POST_RACE DROP 検証2テスト + JRA フィルタ検証3テスト |
| `src/features/odds_dynamics_features.py` | 切り詰めを無条件化 |
| `README.md` | 出力ファイル情報を追加 |

## 次のステップ (Phase 2)

バックテストとペーパートレードの特徴量 parquet を比較する分析スクリプトを作成:
1. **分布比較** — 数値列ごとに mean/std/quantile を比較、乖離列を自動検出
2. **同一レース差分** — race_id で JOIN (可能な場合) して列値の差分確認
3. 特にオッズ系特徴量 (発走前オッズのタイミング差) に注目
