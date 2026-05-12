# Phase 24: Feature Audit & Pruning - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-12
**Phase:** 24-Feature Audit & Pruning
**Areas discussed:** プルーニング基準, モデル別 vs 統一プルーニング, キャッシュ無効化戦略, ROI検証手法

---

## プルーニング基準

### Q1: ノイズ特徴量の除外基準

| Option | Description | Selected |
|--------|-------------|----------|
| 保守的: ゼロ/負のみ | Gain重要度=0 AND Perm≤0 の特徴量のみ除外 | |
| 中庸: 低重要度も含める | Gain<閾値 OR Perm<閾値 の特徴量を除外 | |
| 監査のみ: 自動除外なし | 監査結果を出力するのみ、実際の除外は手動判断 | |

**User's choice:** ベストプラクティス追求 — Tier 1自動除外 + Tier 2レポート出力の多段階プルーニング
**Notes:** ユーザーは常に「ベストプラクティスを追求」方針

### Q2: 多段階プルーニングの設計

| Option | Description | Selected |
|--------|-------------|----------|
| Tier 1のみ自動除外 | Gain=0 AND Perm≤0 のみ除外 | |
| Tier 1除外 + Tier 2レポート | Tier 1自動除外 + 低重要度をフラグ付けしてユーザー判断 | ✓ |
| OR条件で広域除外 | Gain=0 OR Perm≤0 のいずれかを満たすものを全て除外 | |

**User's choice:** Tier 1除外 + Tier 2レポート
**Notes:** ベストプラクティス — 確実なノイズは自動除去、低重要度は可視化して判断

### Q3: 除外の適用単位

| Option | Description | Selected |
|--------|-------------|----------|
| 全モデル共通 | 全モデルでノイズなら除外 | |
| モデル別個別プルーニング | 各モデルのFEATURE_COLSを独立に最適化 | ✓ |

**User's choice:** モデル別個別プルーニング
**Notes:** ベストプラクティス — 各モデルに最適な特徴量セットを提供

### Q4: 除外の安全性確認方法

| Option | Description | Selected |
|--------|-------------|----------|
| OOF logloss/AUC比較 | 高速な品質チェック、フルBT不要 | ✓ |
| 重要度のみ: 再検証なし | 実装簡単だが安全性低い | |
| フルバックテスト | 最も確実だが~57分/年 | |

**User's choice:** OOF logloss/AUC比較
**Notes:** 速度と精度のバランス重視

---

## モデル別 vs 統一プルーニング

この領域はプルーニング基準の議論（Q3）で決定済み。モデル別個別プルーニングを採用。

---

## キャッシュ無効化戦略

### Q5: キャッシュ無効化手法

| Option | Description | Selected |
|--------|-------------|----------|
| コードハッシュ方式 | 対象Pythonファイルの内容ハッシュをキャッシュキーに含める | ✓ |
| バージョン文字列方式 | VERSION定数を追加しキャッシュキーに含める（手動更新） | |
| キャッシュ廃止 | 常に再計算（実行時間増） | |

**User's choice:** コードハッシュ方式
**Notes:** ベストプラクティス — 手動操作なしで確実に無効化

### Q6: ハッシュ対象範囲

| Option | Description | Selected |
|--------|-------------|----------|
| src/features/ 全ファイル | 特徴量関連の全変更を確実に捕捉 | ✓ |
| build_all()関連ファイルのみ | 範囲は狭いが主要変更は捕捉 | |

**User's choice:** src/features/ 配下の全.pyファイル
**Notes:** ベストプラクティス

### Q7: 古いキャッシュファイルの処理

| Option | Description | Selected |
|--------|-------------|----------|
| 自動削除 | ディスク容量を節約 | ✓ |
| 残置: 手動クリーンアップ | 手動でクリーンアップ可能だがディスク消費 | |

**User's choice:** 自動削除
**Notes:** ベストプラクティス

---

## ROI検証手法

### Q8: ROI検証フロー

| Option | Description | Selected |
|--------|-------------|----------|
| 段階的: OOF確認→フルBT | 二重検証で確実、所要時間~57分 | ✓ |
| OOFのみ: BTなし | 高速だがROI直接影響は未確認 | |
| フルBTのみ | OOFでの早期検出なし | |

**User's choice:** 段階的: OOF確認→フルBT
**Notes:** ベストプラクティス — 高速なOOF確認で安全性チェック後、フルBTで最終確認

### Q9: ROI比較のベースライン

| Option | Description | Selected |
|--------|-------------|----------|
| v1.5結果を流用 | Phase 22のROI 84.4%を使用、再実行なし | ✓ |
| Phase 24で新ベースライン確立 | 正確だが~57分追加 | |

**User's choice:** v1.5結果を流用
**Notes:** 再実行コストを節約

### Q10: ROI悪化時の対応

| Option | Description | Selected |
|--------|-------------|----------|
| ロールバック+分析レポート | 即座にロールバック、原因分析レポート出力 | ✓ |
| ロールバック+自動再試行 | より保守的な基準で自動再試行 | |

**User's choice:** ロールバック+分析レポート
**Notes:** 安全優先。自動再試行は複雑性が高く、Phase 24のスコープに収まらないリスク

---

## Claude's Discretion

- Tier 2「低重要度」の具体的な閾値定義
- 監査レポートの出力形式とファイル配置
- OOF logloss/AUC比較の具体的な実装
- キャッシュキー計算のハッシュ対象ファイルリスト
- ロールバック時の原因分析レポートのフォーマット
- プルーニング後のフルバックテスト実行コマンド構成

## Deferred Ideas

None — discussion stayed within phase scope
