# 競馬AI ドキュメント整備 デザイン仕様書

**日付:** 2026-03-26
**ステータス:** Approved (review iteration 2)
**対象:** README.md + docs/guide/ + docs/concepts/ + docs/reference/

---

## 1. 目的

競馬AI予測システム v5.5 のアーキテクチャと使用方法を、技術知識のない競馬ファンから上級開発者まで、全レベルの読者が理解できるドキュメントを整備する。

### 1.1 対象読者

| レベル | 対象 | 目的 |
|--------|------|------|
| 入門 | 競馬ファン（技術知識なし） | システムが何をするのかを理解し、使えるようになる |
| 中級 | 興味のある利用者 | 仕組みの概念を理解し、信頼性を判断できる |
| 上級 | 開発者・改造希望者 | コード構造を理解し、改造・拡張ができる |

### 1.2 スコープ

含む:
- プロジェクト概要 (README)
- AI/機械学習の仕組み解説（素人向け）
- 競馬ドメイン知識（素人向け）
- セットアップ・運用ガイド
- トラブルシューティング / FAQ
- 開発者向けリファレンス
- ノートブックガイド

含まない:
- `docs/design.md` の内容は残す（相互リンクのみ）
- Phase 実装計画（`docs/superpowers/plans/`）は残す
- API リファレンス（自動生成ドキュメントは別途検討）

---

## 2. ドキュメント構成

### 2.1 ファイル一覧

```
README.md                              ← プロジェクト窓口・全体地図
docs/
  guide/                               ← 【入門編】素人向け
    01_keiba_basics.md                 ← 競馬の基礎知識
    02_ai_prediction_basics.md         ← AI予測の基礎
    03_system_overview.md              ← システム概要（検証結果セクション含む）
    04_getting_started.md              ← セットアップ + トラブルシューティング
  concepts/                            ← 【中級編】仕組みの理解
    01_data_pipeline.md                ← データの流れ
    02_prediction_models.md            ← 予測モデル（2段階・Stage1/Stage2）
    03_advanced_models.md              ← 高度なモデル（Market Model・EV補正・レジーム）
    04_betting_strategy.md             ← 投資戦略
    05_backtest_validation.md          ← バックテストと検証 + ノートブックガイド
  reference/                           ← 【上級編】開発者向け
    01_architecture.md                 ← アーキテクチャ詳細
    02_code_structure.md               ← コード構造（安定性アノテーション付き）
    03_configuration.md                ← 設定リファレンス
    04_contributing.md                 ← 開発ガイド（CLAUDE.md参照）
```

合計: 15ファイル（README.md + 14 Markdown ファイル）

### 2.2 ファイル長の目安

一貫性を保つため、各ファイルの行数ターゲットを定める:

| ファイル | 目安行数 | 備考 |
|----------|---------|------|
| README.md | 100-150行 | プロジェクト窓口 |
| guide/ (各) | 100-200行 | 素人向け、簡潔に |
| concepts/ (各) | 150-300行 | 概念解説、図解を含む |
| reference/ (各) | 200-400行 | 技術リファレンス |

超過しそうな場合はセクションを分割すること。

---

## 3. 各ファイルの詳細仕様

### 3.1 README.md — プロジェクトの窓口

**目的:** 「このプロジェクトは何か？」を30秒で理解できる窓口。

**構成:**
1. プロジェクト名と概要 — 1行で何をするシステムか
2. 特徴のハイライト — 3〜5個の箇条書き（専門用語を使わず）
3. クイックスタート — 環境構築の最小手順（3ステップ程度）
4. ドキュメントマップ — レベル別の読み方ガイド
5. 期待できる成果 — 目標ROIやリスクの現実的な説明
6. 免責事項・注意 — ギャンブルリスクに関する注意喚起
7. ライセンス

**図解:**
- Mermaid: システム全体像（データ流入→予測→投票までの流れ）

**トーン:** 親しみやすく、専門用語を避ける。バナー画像やロゴは含まない（テキストのみ）。

---

### 3.2 docs/guide/01_keiba_basics.md — 競馬の基礎知識

**目的:** 競馬のルールや用語をゼロから解説。このシステムの理解に必要最小限に絞る。

**構成:**
1. 競馬とは — 簡潔な概要
2. 馬券の種類 — 単勝・複勝・ワイド（このシステムが扱う3種類に集中）
3. オッズの仕組み — オッズの数学的意味（人気=確率ではない）
4. 払戻しの計算 — 100円買いでいくら戻るか
5. 人気と実力の違い — なぜAI予測に価値があるのか
6. 用語集 — 競馬用語のミニ辞書

**トーン:** 中学校卒業程度の読者が理解できるレベル。比喩を多用。

---

### 3.3 docs/guide/02_ai_prediction_basics.md — AI予測の基礎

**目的:** 「AIって何？」からこのシステムで使っている手法の概念まで。**汎用的なML概念**に留め、このシステム特有のアーキテクチャは concepts/ で扱う。

**構成:**
1. AI/機械学習とは — 「過去のデータからパターンを見つける」の比喩で
2. LightGBMとは — 「決定木の森」の比喩で解説
3. 学習と予測の違い — 訓練データとテストデータ
4. なぜ競馬予測にAIが有効なのか — 人間の認知バイアス vs 統計的アプローチ
5. AIの限界 — 完璧ではないこと、過去のデータに依存すること

**図解:**
- Mermaid: 決定木の概念図（分岐のイメージ）
- Mermaid: 学習→予測の流れ

**トーン:** 全くの初心者向け。数式は使わない。全て比喩で説明。

**境界:** このファイルは「機械学習の一般概念」のみを扱う。このシステム特有の2段階モデルやEV補正については `concepts/02_prediction_models.md` に譲る。重複を避けること。

---

### 3.4 docs/guide/03_system_overview.md — このシステムは何をするのか

**目的:** システムの全体像を概念レベルで。技術用語を最小限に。

**構成:**
1. このシステムがやること — データ収集→予測→投票の3ステップ
2. どんな馬券を買うのか — 単勝・複勝・ワイド
3. どうやって「どの馬が勝つか」を予測するのか — 特徴量の概念
4. どうやって「いくら買うか」を決めるのか — DD制御の概念
5. どうやって「どのレースに参加するか」を決めるのか — レース品質スクリーニング
6. 検証結果 — バックテストの成果（ROI、DD等のサマリ）
7. 期待できる成果とリスク — 現実的な目標と注意点

**図解:**
- Mermaid: 全体フロー図（データ→特徴量→モデル→予測→投票）

**トーン:** 「システムを使う人」が全体像を掴めるように。

---

### 3.5 docs/guide/04_getting_started.md — セットアップ・最初の1歩

**目的:** 実際にシステムを動かすための手順。依存関係が多いため、トラブルシューティングを充実させる。

**構成:**
1. 必要なもの — PCのスペック、PostgreSQL、EveryDB2等
2. インストール手順 — ステップバイステップ
3. 設定 — config/settings.yaml の最小設定
4. 最初の予測を動かすまで
5. よくあるエラーと対処法（拡充版） — 以下のカテゴリ別に記載:
   - PostgreSQL 接続エラー（PGPASSWORD、ポート、権限）
   - Python/mise 環境エラー（バージョン不一致、パス）
   - EveryDB2 データ取得エラー（テーブル未存在、権限）
   - import エラー（pythonpath、パッケージ未インストール）
6. FAQ — 頻繁に寄せられる質問と回答
7. 次のステップ — 中級編へのリンク

**トーン:** 手順書。コマンドはコピペで動くように。スクリーンショットは含まない（テキストのみ）。

---

### 3.6 docs/concepts/01_data_pipeline.md — データの流れ

**目的:** データがどのように流れて予測に至るかを解説。

**構成:**
1. データソース — EveryDB2/JRA-VAN DataLab とは
2. データベース構造 — 5スキーマの概要（raw, odds_history, feature, prediction, betting）
3. 特徴量エンジニアリングとは — 「馬の実力を数字で表す」の比喩
4. 特徴量の種類 — オッズ動態、市場バイアス、情報非対称性、レース内特徴量
5. リーク防止 — 未来の情報が混入しない仕組み

**図解:**
- Mermaid: データフロー図（外部DB→特徴量→予測）
- Mermaid: 5スキーマの関係図

**トーン:** 概念レベル。コードは示さないが、モジュール名は言及する。

---

### 3.7 docs/concepts/02_prediction_models.md — 予測モデル（基礎）

**目的:** このシステムのコアである予測モデルの基礎構造を解説。元の8トピックから分離し、中核モデルに集中させる。

**構成:**
1. 2段階モデルの概念 — P(hit) × E(odds|hit) の比喩で
2. Stage1: 能力モデル — 「当たる確率」を予測
3. Stage2: 払戻回帰モデル — 「当たった時の払戻し」を予測
4. サブモデル — 芝/ダート分割の理由
5. ワイド予測の基礎 — ワイド馬券の特殊性と基本アプローチ

**図解:**
- Mermaid: 2段階モデルのフロー図
- Mermaid: サブモデル分割のイメージ

**トーン:** 数式は使わず、比喩と図で説明。基礎概念に集中。

---

### 3.8 docs/concepts/03_advanced_models.md — 高度なモデル

**目的:** Market Model、EV補正、レジーム検知など、高度なモデル群を解説。`02_prediction_models.md` の続き。

**構成:**
1. Market Model — オッズと予測の差異を学ぶ（差分 log_error のみ出力）
2. EV補正モデル — 予測値を実績に近づける調整（P補正 × E補正）
3. レジーム検知 — 市場の状態（攻撃的/保守的/崩壊）を判定
4. ワイド予測の高度な話題 — Var_proxy とリスク調整
5. MLflow による実験管理 — モデルのバージョン管理と追跡

**図解:**
- Mermaid: EV補正の概念図（P補正 × E補正）
- Mermaid: レジーム検知の状態遷移図

**トーン:** 概念レベルだが、やや技術的。`design.md` への深掘りリンク。

---

### 3.9 docs/concepts/04_betting_strategy.md — 投資戦略

**目的:** 「どのレースに、いくら賭けるか」を決める仕組みを解説。

**構成:**
1. 投資判断の全体フロー — レース選定→馬券選定→金額決定
2. レース品質スクリーニング — 参加するレースを選ぶ基準
3. ゲートキーパー — エントリー条件の最終チェック
4. ステーク計算 — 「いくら賭けるか」の計算方法
5. ドローダウン（DD）制御 — 連敗時の資金防衛（Rolling ROI + EWMA + ヒステリシス）
6. レートマネー・フィルター — 発走直前のオッズ変動への対応（t-3min/t-2min）
7. メタスイッチャー — 状況に応じた戦略切り替え
8. オーケストレーター — 全戦略を統合して発走3分前に確定

**図解:**
- Mermaid: 投資判断フロー図
- Mermaid: DD制御の概念図

**トーン:** 投資のリスク管理に焦点を当てる。

---

### 3.10 docs/concepts/05_backtest_validation.md — バックテストと検証

**目的:** システムの信頼性を検証する方法を解説。ノートブックガイドを統合。

**構成:**
1. バックテストとは — 過去データでシミュレーションする概念
2. ウォークフォワード交差検証 — 時間を考慮した検証方法（4年学習→1年検証→1年進む）
3. ホールドアウト検証 — 「見たことないデータ」での最終テスト（2022-2024）
4. パラメータ凍結プロトコル — 過学習を防ぐ仕組み
5. 評価指標 — ROI、最大DD、月次勝率、プロフィットファクター
6. 合格基準 — バックテストの合格ライン
7. 分析ノートブックガイド — 12ノートブックの目的と結論を表形式で紹介:
   - NB00: 環境セットアップ
   - NB01: 探索的データ分析 (EDA)
   - NB02: オッズ動態分析
   - NB03: Market Model 差分分析
   - NB04: 2段階 vs 1段階 ABテスト
   - NB05: ワイド・リスク調整スコア
   - NB06: レース品質独立性検証
   - NB07: 2分割 vs 7分割 サブモデル比較
   - NB08: DD Rolling ROI シミュレーション
   - NB09: EV補正分析
   - NB10: log_error 正規化
   - NB11: ホールドアウト最終評価

**図解:**
- Mermaid: ウォークフォワードCVの時間軸イメージ
- Mermaid: 検証パイプライン図

**トーン:** 「なぜこのシステムを信頼できるのか」に答えるスタンス。

---

### 3.11 docs/reference/01_architecture.md — システムアーキテクチャ詳細

**目的:** 技術的な全体像を開発者向けに。

**構成:**
1. パッケージ依存関係 — 全10パッケージの依存グラフ
2. 設計思想 — 19の絶対ルールの要約
3. 主要な設計決定 — SQLAlchemy Core only、ORM不使用、2段階モデル等
4. データフロー（技術版） — SQLAlchemy Core を使ったデータの流れ
5. design.md へのリンク — 深い技術詳細はこちらへ

**図解:**
- Mermaid: パッケージ依存関係図

**トーン:** 技術的。開発者向け。

---

### 3.12 docs/reference/02_code_structure.md — コード構造・モジュール解説

**目的:** 各モジュールの詳細な解説。安定度に応じて記述量を調整する。

**安定性アノテーション:** 各パッケージに以下のマークを付与:
- ✅ 安定 — Phase A で実装済み、変更少ない（domain/, db/）
- 🔧 ほぼ安定 — Phase B-F で実装済み、微修正の可能性あり（features/, models/, betting/, backtest/, pipelines/）
- 🚧 開発中 — インタフェース定義済み、実装が浅い（automation/, monitoring/, ingestion/）

**構成:**
1. ディレクトリ構成図 — 全ファイル一覧
2. domain/ ✅ — データクラス（Race, Entry, Bet, OddsSnapshot, DDState等）
3. db/ ✅ — データベース層（schema.py, connection.py）
4. features/ 🔧 — 特徴量エンジニアリング（7モジュール: feature_engine, intra_race_features, odds_dynamics_features, market_bias_features, info_asymmetry_features, race_difficulty_model, leakage_validators）
5. models/ 🔧 — MLモデル（12モジュール: submodel_manager, market_model, stage1_ability_model, two_stage_return_model, ev_correction_model, wide_two_stage_model, wide_pair_builder, race_quality_screener, regime_detector, robust_confidence_estimator, walk_forward_cv, + __init__.py）
6. betting/ 🔧 — 投資戦略（9モジュール: stake_calculator, drawdown_controller, late_money_filter, gate_keeper, meta_switcher, win_strategy, place_strategy, wide_strategy, orchestrator）
7. backtest/ 🔧 — バックテスト（3モジュール: engine, validation_suite, parameter_freeze_protocol）
8. pipelines/ 🔧 — MLパイプライン（1モジュール: training_pipeline）
9. automation/ 🚧 — 自動化（PAT投票、スケジューラ、安全ガード）※インタフェース中心に記述
10. monitoring/ 🚧 — 監視（モデル監視、再学習トリガー、通知）※インタフェース中心に記述
11. ingestion/ 🚧 — データ取得（JVLink, オッズ収集）※インタフェース中心に記述
12. テスト構造 — 43テストファイル、モック戦略

**トーン:** 技術リファレンス。各モジュールの主要クラスと責任を箇条書きで。🚧 パッケージはインタフェースと責任の概要に留め、詳細な実装解説は避ける。

---

### 3.13 docs/reference/03_configuration.md — 設定ファイルリファレンス

**目的:** 全設定項目の解説。

**構成:**
1. config/settings.yaml — 全項目の解説
   - database: 接続設定
   - paths: ファイルパス
   - logging: ログ設定
   - feature_engine: 特徴量エンジニアリング設定
   - late_money: レートマネー閾値
   - submodel: サブモデル設定
2. config/backtest_config.yaml — 全項目の解説
   - walk_forward: パラメータ
   - holdout: ホールドアウト期間
   - pass_criteria: 合格基準
3. 環境変数 — PGPASSWORD等
4. MLflow 設定 — mlflow_tracking_uri、実験管理の基本設定

**トーン:** リファレンスマニュアル。各項目の型、デフォルト値、説明。

---

### 3.14 docs/reference/04_contributing.md — 開発への参加方法

**目的:** 開発環境のセットアップとコーディング規約。`CLAUDE.md` の内容を重複させず参照する。

**構成:**
1. 開発環境のセットアップ — mise, pip, Python 3.11
2. コーディング規約 — Ruff (lint+format), Mypy (型チェック) ※詳細は `CLAUDE.md` を参照
3. コミットメッセージ — Conventional Commits（日本語）
4. テストの書き方 — unittest.mock の使い方
5. プルリクエストの作成方法
6. 関連ドキュメント — `CLAUDE.md`、`docs/design.md` へのリンク

**トーン:** 実用的な開発ガイド。`CLAUDE.md` と重複する内容はリンクで代替する。

---

## 4. 執筆ガイドライン

### 4.1 トーンとスタイル

| レベル | トーン | 専門用語 | 比喩 |
|--------|--------|----------|------|
| 入門 | 親しみやすい | 使わない/使う場合は必ず解説 | 積極的に使用 |
| 中級 | 丁寧な解説 | 使うが解説を添える | 適度に使用 |
| 上級 | 技術的 | 自由に使用 | 必要に応じて |

### 4.2 図解

- 全て Mermaid 記法を使用
- GitHub でレンダリングされることを確認
- 複雑な図は分割して段階的に提示

### 4.3 ナビゲーション

- **README.md** に「ドキュメントマップ」を設け、レベル別の読み方を案内
- **各ファイルの末尾**に双方向リンク（「次へ進む」+「前のページ」）
- **中級編の各ファイル**に `design.md` への深掘りリンク
- **上級編**に `design.md` の該当セクションへの直接リンク
- **クロス階層リンク:** 関連する他階層のファイルへ適宜リンク（例: concepts/02 に guide/02 への「基礎を復習」リンク）

### 4.4 言語

- 全て日本語
- 技術用語は日本語表記を優先（カッコ内に英語を併記可）
  - 例: 特徴量エンジニアリング (Feature Engineering)

### 4.5 ファイルテンプレート

全ファイルで統一したフォーマットを使用する:

```markdown
# [タイトル]

[1段落の概要説明]

## [セクション1]
[内容]

## [セクション2]
[内容]

---

> **次のドキュメント:** [タイトル](相対パス) | **前のドキュメント:** [タイトル](相対パス)
```

README.md のみテンプレートを適用しない（プロジェクト窓口として独自フォーマット）。

### 4.6 既存ドキュメントとの関係

- `docs/design.md` — そのまま残す。中級編・上級編からリンクを張る
- `docs/everydb2-data-reference.md` — そのまま残す。`concepts/01_data_pipeline.md` からリンク
- `docs/superpowers/plans/` — そのまま残す。開発者向けリファレンスからリンク
- `CLAUDE.md` — そのまま残す。`reference/04_contributing.md` から参照リンク
- `notebooks/` — そのまま残す。`concepts/05_backtest_validation.md` 内のノートブックガイドで紹介。各ノートブックへのリンクを張る

---

## 5. 実装順序

以下の順序でファイルを作成する:

1. `README.md` — プロジェクトの窓口（他のファイルのリンク先になるため最初）
2. `docs/guide/01_keiba_basics.md`
3. `docs/guide/02_ai_prediction_basics.md`
4. `docs/guide/03_system_overview.md`
5. `docs/guide/04_getting_started.md`
6. `docs/concepts/01_data_pipeline.md`
7. `docs/concepts/02_prediction_models.md`
8. `docs/concepts/03_advanced_models.md`
9. `docs/concepts/04_betting_strategy.md`
10. `docs/concepts/05_backtest_validation.md`
11. `docs/reference/01_architecture.md`
12. `docs/reference/02_code_structure.md`
13. `docs/reference/03_configuration.md`
14. `docs/reference/04_contributing.md`

各ファイル完了時に git commit。全15ファイル完了後に全体のリンク整合性を確認。
