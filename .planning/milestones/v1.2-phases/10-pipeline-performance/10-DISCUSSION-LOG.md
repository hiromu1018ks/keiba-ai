# Phase 10: Pipeline Performance - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-04
**Phase:** 10-Pipeline Performance
**Areas discussed:** ワイド払戻ベクトル化手法, キャッシュ対象と無効化戦略, pyinstrument統合方法, groupby辞書のメモリ戦略

---

## ワイド払戻ベクトル化手法

| Option | Description | Selected |
|--------|-------------|----------|
| 正規表現ベース | pandas str.extract() で kumi 文字列を一括パース。文字列長で3パターンをwhere条件で処理 | ✓ |
| ゼロ埋め正規化 | kumi文字列を4桁ゼロ埋めに正規化してから前半/後半に分割 | |
| ワイドのみiterrows残置 | ワイド払戻マップのみiterrows()を維持し、他6箇所をベクトル化 | |
| You decide | Claude裁量で最適な手法を選択 | |

**User's choice:** You decide → 正規表現ベースを選択
**Notes:** ユーザーは一貫して「ベストプラクティスを追求」と回答

---

## ベクトル化の範囲

| Option | Description | Selected |
|--------|-------------|----------|
| 全7箇所ベクトル化 | payout map 3種 + final_odds_map + final_win_odds_map + top3抽出の全7箇所 | ✓ |
| ボトルネック4箇所のみ | top3抽出(3箇所)は残置。残り4箇所(payout maps + odds maps)をベクトル化 | |

**User's choice:** 全7箇所ベクトル化
**Notes:** 統一性を重視。top3抽出もnsmallest()で置き換え

---

## キャッシュ対象範囲

| Option | Description | Selected |
|--------|-------------|----------|
| 履歴系4つのみ | HorseHistory + JockeyContext + TrainerContext + JockeyTrainerCombo | |
| 全6種キャッシュ | 事前計算される全特徴量(6種)をParquetキャッシュ | ✓ |
| HorseHistoryのみ | 最も計算コストが高い1つのみ | |

**User's choice:** 全6種キャッシュ（ベストプラクティスを追求）
**Notes:** 実装難易度は問わない

---

## キャッシュ無効化戦略

| Option | Description | Selected |
|--------|-------------|----------|
| コンテンツハッシュ | 入力Parquetのハッシュ値をキャッシュキーに含める | |
| タイムスタンプ比較 | 更新日時で比較。シンプルだが内容変更を保証しない | |
| ハイブリッド | タイムスタンプ高速チェック + ハッシュ検証 | ✓ |

**User's choice:** ハイブリッド（ベストプラクティスを追求）
**Notes:** タイムスタンプで高速チェック、変更ありならハッシュで検証

---

## キャッシュ保存場所

| Option | Description | Selected |
|--------|-------------|----------|
| data/features/ | 既存の特徴量ディレクトリに配置 | |
| data/features/cache/ | 専用ディレクトリで分離 | ✓ |

**User's choice:** data/features/cache/（ベストプラクティスを追求）
**Notes:** 既存ファイルとキャッシュファイルの分離

---

## pyinstrument起動方法

| Option | Description | Selected |
|--------|-------------|----------|
| --profile フラグ | CLIフラグで制御。run_backtest.pyに統合 | |
| 環境変数制御 | KEIBA_PROFILE=1で制御 | |
| 常時有効 | 毎回出力ファイル生成 | ✓ |

**User's choice:** --profile フラグ + run_wf_validation.py にも統合（ベストプラクティスを追求）
**Notes:** ユーザーから「run_wf_には統合しなくていいの？」という質問あり。run_backtest.pyとrun_wf_validation.pyの両方に統合することに決定。共通ユーティリティ(src/utils/profiling.py)に抽出

---

## pyinstrument出力形式

| Option | Description | Selected |
|--------|-------------|----------|
| HTML + テキスト両方 | HTML(data/profiles/) + テキスト(stdout) | ✓ |
| HTMLのみ | 折りたたみ可能なコールツリー | |
| テキストのみ | ターミナルで即座に確認 | |

**User's choice:** HTML + テキスト両方
**Notes:** 両方の形式で分析可能に

---

## groupby辞書化の範囲

| Option | Description | Selected |
|--------|-------------|----------|
| 5つ全て辞書化 | feat_df + hist/jockey/trainer/jt | ✓ |
| feat_dfのみ | 最も大きいDataFrameのみ | |

**User's choice:** 5つ全て辞書化（ベストプラクティスを追求）
**Notes:** メモリ安全性に懸念あり。pandas≥2.0のgroupbyはviewを返すため実質メモリ増は1.1-1.2倍。データサイズ(~38,000行)でリスクなし

---

## groupby辞書の実装方法

| Option | Description | Selected |
|--------|-------------|----------|
| ヘルパー関数 | build_race_groups()でカプセル化 | ✓ |
| インライン実装 | BacktestEngine.run()に直接埋め込み | |

**User's choice:** ヘルパー関数（ベストプラクティスを追求）
**Notes:** 再利用性と保守性を優先

---

## Claude's Discretion

- ワイド払戻ベクトル化の具体的な正規表現パターン設計
- build_race_groups() のシグネチャと返り値の型
- キャッシュ無効化ハッシュの計算方法
- pyinstrumentユーティリティのAPI設計
- ベクトル化後のテスト範囲

## Deferred Ideas

なし — 全ての議論がフェーズスコープ内に収まった
