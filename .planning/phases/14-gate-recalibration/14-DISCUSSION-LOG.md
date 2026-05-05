# Phase 14: Gate Recalibration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-06
**Phase:** 14-Gate Recalibration
**Areas discussed:** 分布ドリフト診断の設計, use_ensemble伝播テスト戦略, ゲート再学習検証方法

---

## 分布ドリフト診断の設計

### 診断実装方式

| Option | Description | Selected |
|--------|-------------|----------|
| パイプライン統合 | バックテスト実行時に自動でドリフト診断を実行し、結果をログ+JSONに出力 | ✓ |
| 独立スクリプト | scripts/run_drift_diagnostics.pyとして独立実装 | |
| 統合 + CLI両対応 | パイプライン統合をメインに単独実行可能な関数も公開 | |

**User's choice:** パイプライン統合

### 診断出力フォーマット

| Option | Description | Selected |
|--------|-------------|----------|
| JSONファイル | JSONファイルにKS統計量/p-value/Wasserstein距離を保存 | |
| ログ出力のみ | PythonロガーでINFO/WARNINGレベルで出力 | |
| JSON + コンソール | JSON保存 + コンソールにサマリを表示 | ✓ |

**User's choice:** JSON + コンソール

### 分布比較の粒度

| Option | Description | Selected |
|--------|-------------|----------|
| 主要列の全データ比較 | p_win_final等の主要確率・EV列の分布を全データで比較 | |
| サーフェス別も比較 | 上記+芝/ダート別の分割比較 | |
| 年度別推移も追跡 | 全確率列+年度別時系列でドリフト推移を追跡 | |

**User's choice:** ベストプラクティス追求 — 主要列全データ比較 + サーフェス別 + 年度別推移の全てを実装

### ドリフト検出時の対応

| Option | Description | Selected |
|--------|-------------|----------|
| 警告のみ | 診断結果を出力するのみ。バックテストは継続 | |
| 警告 + 再学習推奨 | WARNINGログ + 再学習を推奨 | |

**User's choice:** ベストプラクティス追求 — WARNING + 再学習推奨 + drift_detectedフラグ付きJSON出力

**Notes:** ユーザーは一貫して「ベストプラクティス追求」を選択。実装難易度より品質・網羅性を優先。

---

## use_ensemble伝播テスト戦略

### テストアプローチ

| Option | Description | Selected |
|--------|-------------|----------|
| フラグ経路のモックテスト | 各コンポーネントにuse_ensemble=Trueが正しく渡ることをモックで検証 | ✓ |
| 値レベルのアサーション | fixtureデータでStackedEnsemble OOFを生成し値を確認 | |
| 経路テスト + キーポイント値検証 | モック基本 + 重要箇所で値アサーション | |

**User's choice:** フラグ経路のモックテスト

### テストスコープ

| Option | Description | Selected |
|--------|-------------|----------|
| 統合テスト1つで全体検証 | 1つのテストクラスでend-to-endのフラグ伝播を確認 | ✓ |
| コンポーネント別個別テスト | ModelLoader, TrainingPipeline, RacePredictor別々 | |
| 個別 + E2Eテスト | 個別テスト + BacktestEngine.run()のE2Eテスト | |

**User's choice:** 統合テスト1つで全体検証

### テスト対象経路

| Option | Description | Selected |
|--------|-------------|----------|
| True経路のみ | use_ensemble=Trueの経路だけテスト | ✓ |
| True/False両方 | デフォルト動作も確認 | |

**User's choice:** True経路のみ

---

## ゲート再学習検証方法

### 検証アプローチ

| Option | Description | Selected |
|--------|-------------|----------|
| バックテスト時自動比較 | 単一モデルのゲート値を保持しアンサンブル再学習後と自動比較 | |
| 別ユニットテスト | fixtureデータで単一/アンサンブルのゲート値を比較するテスト | |
| アサーション追加 | edgesが既知の単一モデルedgesと異なることをassert | |

**User's choice:** ベストプラクティス追求 — ユニットテスト(確定検証) + パイプラインassertion(ランタイム安全性)の二段構え

---

## Claude's Discretion

- 診断のWasserstein距離閾値(warn/error)の具体的な値
- JSON出力のスキーマ設計
- テストfixtureデータの内容

## Deferred Ideas

None — discussion stayed within phase scope
