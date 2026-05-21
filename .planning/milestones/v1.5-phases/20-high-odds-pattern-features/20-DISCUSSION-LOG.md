# Phase 20: 高オッズ的中パターン特徴量 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-09
**Phase:** 20-高オッズ的中パターン特徴量
**Areas discussed:** HODDS-01 高オッズパターン分析手法, HODDS-02 クラストラジェクトリ設計, HODDS-03 フォーム改善率の測定, HODDS-04 環境変化適性の範囲

---

## HODDS-01: 高オッズパターン分析手法

### Q1: 高オッズ的中パターンの分析方法

| Option | Description | Selected |
|--------|-------------|----------|
| 統計プロファイリング | 高オッズ的中馬 vs 非的中馬の特徴量分布差を統計検定で特定 | |
| SHAP値分析 | 既存LightGBMで高オッズ馬のみSHAP値分析 | |
| ハイブリッド | 統計プロファイリング + SHAPの組み合わせ | ✓ |

**User's choice:** ベストプラクティスを追求、学習/バックテスト時間を延ばさない
**Notes:** ハイブリッドが最も網羅的。分析はフェーズ内一回限りの計算なのでランタイムへの影響なし

### Q2: 分析モジュールと特徴量生成の分割

| Option | Description | Selected |
|--------|-------------|----------|
| 単一モジュール | 分析+特徴量一体 | |
| 分析と特徴量を分離 | scripts/に分析、src/features/に特徴量 | ✓ |
| Claude's discretion | | |

**User's choice:** ベストプラクティスを追求
**Notes:** 分離により分析結果に基づいて特徴量設計を独立して反復可能

### Q3: 高オッズの定義範囲

| Option | Description | Selected |
|--------|-------------|----------|
| オッズ20+のみ | ROADMAP定義に一致 | |
| オッズ10+に拡張 | サンプル増加 | |
| Claude's discretion | 分析で動的に調整 | ✓ |

**User's choice:** Claude's discretion
**Notes:** 初期はオッズ20+、サンプル不足時は10+に拡張

### Q4: 分析結果の特徴量化方法

| Option | Description | Selected |
|--------|-------------|----------|
| 手動特徴量設計 | 分析結果に基づいて特徴量を設計。実行時オーバーヘッドなし | ✓ |
| 学習済みクラスタリングモデル | クラスタリングモデルで推論時に分類 | |

**User's choice:** ベストプラクティスを追求
**Notes:** 手動設計は確定的・解釈性高・オーバーヘッドなし

---

## HODDS-02: クラストラジェクトリ設計

### Q1: クラストラジェクトリの特徴量化方法

| Option | Description | Selected |
|--------|-------------|----------|
| 数値シーケンス分解 | 昇級/降級/ネット変化/最高クラスをスカラー値に分解 | ✓ |
| パターン分類 | カテゴリ変数として分類（上昇型、下降型等） | |
| Claude's discretion | | |

**User's choice:** ベストプラクティスを追求
**Notes:** 数値分解はLightGBMが非線形に学習、カテゴリ爆発なし

### Q2: トラジェクトリの期間

| Option | Description | Selected |
|--------|-------------|----------|
| 直近5走 | n_past=5に一致 | ✓ |
| 全過去走 | 長期パターン | |
| Claude's discretion | | |

**User's choice:** 直近5走 (推奨)

### Q3: V字回復パターン

| Option | Description | Selected |
|--------|-------------|----------|
| V字回復を含める | 降級→再昇級のバイナリフラグ+降級期間 | ✓ |
| パターンバイナリなし | 統計サブ特徴量のみ | |
| Claude's discretion | | |

**User's choice:** V字回復パターンを含める (推奨)

---

## HODDS-03: フォーム改善率の測定

### Q1: フォーム改善率の測定方法

| Option | Description | Selected |
|--------|-------------|----------|
| EMAベース指数改善率 | halflife=3のEMA重み付けで非線形回復を捉える | ✓ |
| 2期間差分 | 直近2走 vs その前3走の差 | |
| Claude's discretion | | |

**User's choice:** ベストプラクティスを追求
**Notes:** EMAはPhase 5で確立したhalflife=3標準に従う

### Q2: タイムベース vs 着順ベース

| Option | Description | Selected |
|--------|-------------|----------|
| タイム+着順の両方 | z-scoreタイムと正規化着順の両方 | ✓ |
| 着順のみ | シンプル | |
| Claude's discretion | | |

**User's choice:** タイム+着順の両方 (推奨)

---

## HODDS-04: 環境変化適性の範囲

### Q1: 環境変化の対象範囲

| Option | Description | Selected |
|--------|-------------|----------|
| 既存3変化の適性履歴 | 距離・サーフェス・馬場の過去適性 | ✓ |
| 拡張5変化 | 騎手・調教師変更も追加 | |
| Claude's discretion | | |

**User's choice:** 既存3変化の適性履歴 (推奨)

### Q2: 適性の特徴量化方法

| Option | Description | Selected |
|--------|-------------|----------|
| サブ特徴量分解 | 3変化×3特徴量=9特徴量（平均着順・勝率・経験回数） | ✓ |
| 集約スコア | 3変化×1=3特徴量 | |
| Claude's discretion | | |

**User's choice:** ベストプラクティスを追求
**Notes:** サブ特徴量分解はペース特徴量の確立パターンに一致

---

## Claude's Discretion

- 高オッズの定義範囲（初期20+、必要に応じて10+に拡張）
- 新特徴量の具体的な命名規則
- HorseHistoryFeatures.compute()内の計算統合方法
- 分析スクリプトの出力形式
- サンプル不足時のフォールバック戦略

## Deferred Ideas

None — discussion stayed within phase scope
