---
status: resolved
trigger: "TargetEncoder の fillna で category dtype の列に float 値を設定できず TypeError"
created: 2026-06-07
updated: 2026-06-07
---

# Debug Session: target-encoder-cat-fillna

## Symptoms

- **Expected behavior:** TargetEncoder.transform() が category dtype の列に対しても正常にターゲットエンコーディングを適用する
- **Actual behavior:** TypeError: Cannot setitem on a Categorical with a new category (0.067...) が発生
- **Error messages:** src/features/target_encoding.py:193 で df[cat_col].map(mapping).fillna(self.global_mean_) が失敗
- **Timeline:** 前回 cat-feature-list-incomplete 修正（カテゴリカル列の動的変換追加）の副作用。blood_keito_cd 等が category dtype になり、fillna(float) が不可に
- **Reproduction:** PT predict 実行時、TargetEncoder.transform() 呼び出しで発生

## Root Cause (user-provided)

- 先ほどのカテゴリカル修正で列を category dtype に変換するようになった
- TargetEncoder は df[cat_col].map(mapping).fillna(global_mean_) を実行
- category dtype の Series に対して fillna(float) を呼ぶと、float 値を新しいカテゴリとして追加しようとして TypeError

## Proposed Fix

```python
# Before:
df[te_col] = df[cat_col].map(mapping).fillna(self.global_mean_)

# After:
mapped = df[cat_col].astype(object).map(mapping)
df[te_col] = mapped.fillna(self.global_mean_)
```

## Current Focus

- **hypothesis:** category dtype の列に対する fillna(float) がTypeErrorを引き起こす
- **test:** TargetEncoder.transform() に category dtype の列を渡して正常動作を確認
- **expecting:** category/object dtype に関わらずターゲットエンコーディングが正常完了
- **next_action:** target_encoding.py の該当箇所を修正し、テストで確認
- **reasoning_checkpoint:** map 結果は float になるため、入力を object にキャストすれば fillna(float) が正常動作する

## Evidence

- 2026-06-07: transform() line 193: df[cat_col].map(mapping).fillna(self.global_mean_) -- category dtype Series に float で fillna すると TypeError
- 2026-06-07: fit_transform_oof() line 150-154: test_cats.map(smoothed).fillna(fold_global_mean) -- 同じパターンの別の NaN 埋めパス
- 2026-06-07: fit_transform_oof() line 172-173: nan_mask パスの3箇所目
- 2026-06-07: 修正後のテスト全14件合格（カテゴリカル dtype 回帰テスト2件含む）

## Eliminated

- groupby(observed=True) は category dtype で正常動作 -- 問題なし
- _compute_cat_stats() の to_dict() はプレーン dict を返す -- 問題なし

## Resolution

- **root_cause:** category dtype の Series に float 値で fillna() を呼ぶと pandas が新しいカテゴリの追加を試み TypeError が発生。前回 cat-feature-list-incomplete 修正で列が category dtype になるよう変更されたが、TargetEncoder の3箇所の map+fillna パスがそれに対応していなかった。
- **fix:** map() の前に .astype(object) を追加し、結果を float の Series にしてから fillna() を呼ぶよう変更。3箇所すべてを修正: transform() L193, fit_transform_oof() L150, fit_transform_oof() L172。
- **verification:** tests/test_target_encoding.py 全14件合格（新規回帰テスト2件含む）。フルテストスイート 2712 passed / 7 failed (全て既存の unrelated failures)。
- **files_changed:**
  - src/features/target_encoding.py (3箇所に .astype(object) を追加)
  - tests/test_target_encoding.py (TestCategoricalDtype クラス追加、回帰テスト2件)
