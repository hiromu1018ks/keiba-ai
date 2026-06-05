# Phase 51: Settlement Integrity & Training Pipeline - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-06
**Phase:** 51-Settlement Integrity & Training Pipeline
**Areas discussed:** 精算アーキテクチャ, Win精算の統合方法, 学習パイプライン --betting-target, ベット記録スキーマ移行

---

## 精算アーキテクチャ (Reconciler Architecture)

### Reconciler主軸の選択

| Option | Description | Selected |
|--------|-------------|----------|
| PaperReconciler クラスを主軸 | クラスを拡張、_run_reconcile は薄いラッパー。Phase 52の共有ビルダーと同じパターン | |
| インラインを主軸 | _run_reconcile を直接拡張。Phase 52で改めて統合 | |
| 新規モジュールに抽出 | src/paper_trading/settlement.py に新規抽出 | |

**User's choice:** Other — PaperReconcilerを精算の唯一実装とし、_run_reconcileは引数構築・結果表示・終了コード制御のみ。Phase 52ではなくPhase 51で二重実装を解消すべき。精算処理はbet_id単位で冪等にし、払戻マップ生成はBTのwin/place共通ヘルパーを再利用。

### 精算状態モデル

| Option | Description | Selected |
|--------|-------------|----------|
| 4値outcome (Recommended) | settlement_status + outcome(won/lost/refunded/voided) + payout。ROI計算はsettled only、refunded/voidedは除外 | |
| 2値outcome + refund flag | outcome は won/lost のみ。取消・除外は refund flag で別途管理 | |

**User's choice:** Other — voided(レース不成立)とrefunded(出走取消・除外)を区別。両方ともpayout=stake、effective_stake/ROI分母から除外。同着はwonとして実払戻額記録。cancelled名称はvoidedに変更(ユーザー取消と混同回避)。ROI集計公式を固定(effective_stake/return/ROI/net_profit)。

### bet_id生成規則

| Option | Description | Selected |
|--------|-------------|----------|
| SHA256 composite (Recommended) | SHA256(race_id+umaban+bet_type+predicted_at)[:16] | |
| UUID4 random | ランダム生成 | |

**User's choice:** Other — bet_id = SHA256(session_id | race_id | bet_type | canonical_selection)[:32]。session_idは当日run開始時に永続化しクラッシュ復旧時も再利用。canonical_selectionは馬番。時刻・stakeは含めない。

### リトライ戦略

| Option | Description | Selected |
|--------|-------------|----------|
| per-race + batch 3回×60s (Recommended) | 各レース1回+最終レース後3回60s間隔 | |
| per-race + batch 5回×120s | 各レース1回+最終レース後5回120s間隔 | |

**User's choice:** Other — 回数制限より絶対期限。per-race 1回+最終レース後60s間隔で最大10分間。DB接続エラーと払戻未掲載を区別。期限後pending残りは保存して終了コード2。部分精算は一時ファイル経由atomic replace。

---

## Win精算の統合方法

### 実装場所

| Option | Description | Selected |
|--------|-------------|----------|
| 共通ヘルパーに抽出 (Recommended) | src/betting/payout_maps.py に純粋関数として抽出。BT/PT両方が同一関数使用 | ✓ |
| PaperReconciler内に実装 | クラス内にWin精算ロジックを直接実装 | |

**User's choice:** 共通ヘルパーに抽出。入力列の正規化、出力は倍率統一、同着対応、EveryDB2/I/Oなし。精算判定は返還→払戻マップ→won/lostの順序。不正払戻値はpending維持。

---

## 学習パイプライン --betting-target

### 学習範囲

| Option | Description | Selected |
|--------|-------------|----------|
| 全学習 + target記録 (Recommended) | 全モデル常時学習。targetは記録のみ | |
| Target別部分学習 | 共通+指定target固有のみ。高速 | |

**User's choice:** Target別部分学習。win=共通+Win固有、place=共通+Win基盤+Place固有、wide=v2.4拒否。学習targetをMLflow+meta.jsonに保存。PT起動時にtarget一致を必須検証。

### track_stats永続化先

| Option | Description | Selected |
|--------|-------------|----------|
| data/models/ 内JSON (Recommended) | モデル成果物と同じ場所 | |
| MLflow artifacts | バージョン管理されるが取り回しが増える | |

**User's choice:** Other — ローカルとMLflowの両方に保存。モデル成果物の必須ファイルとする。ModelLoaderは両経路で復元、欠落時fail-fast。SHA256をmeta.json+MLflow params/tagsに記録。

### ModelLoader優先度

| Option | Description | Selected |
|--------|-------------|----------|
| run_id指定時はMLflow優先 (Recommended) | run_id明示指定時MLflow、未指定時ローカル | |
| PTではMLflow必須 | PTは--run-id必須、ローカルフォールバックなし | |

**User's choice:** PTではMLflow必須。--run-id必須、最新run自動選択禁止、--models-dir明示指定時のみローカル。--run-idと--models-dir同時指定禁止。ロード元を実行記録・レポートに保存。

---

## ベット記録スキーマ移行

### result列の扱い

| Option | Description | Selected |
|--------|-------------|----------|
| result + payout 両方 (互換) | result列を残して後方互換。payoutは新列 | |
| result → payout 置換 | result列をpayoutに完全置換。すっきりする | ✓ |

**User's choice:** result → payout完全置換。旧スキーマ(result列のみ)は自動変換せず明示的に拒否。schema_version=2を追加。書き込み前に必須列・型・状態整合性を検証。整合性制約: pending(outcome=NULL,payout=NULL), settled(outcome!=NULL,payout>=0), lost(payout=0), won(payout>0), refunded/voided(payout=stake), bet_id非NULL一意, stake>0。

---

## Claude's Discretion

- payout_maps.py の内部実装詳細(正規化ロジック、統合方法)
- PaperReconciler の内部リトライループ実装
- Pre-training Parquet検証の具体的なチェック内容(NaN率閾値等)
- Feature cache dependency tracking のキャッシュキー計算方式
- atomic replace の一時ファイル命名規則

## Deferred Ideas

- Wide bet settlement — v2.5+ (WID-01, WID-02)
- SafetyGuard integration — v2.5+ (SAF-01, SAF-02)
- Shared feature builder extraction — Phase 52 (PLN-01)
- Conservative MAWC redesign — v2.5+
