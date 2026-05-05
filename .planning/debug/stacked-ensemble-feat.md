---
slug: stacked-ensemble-feat
status: resolved
trigger: "AttributeError: 'StackedEnsemble' object has no attribute 'feature_name' — extract_feature_ranking() が lgb.Booster の .feature_name() を呼び出す前提だが、アンサンブル有効時は hit_model が StackedEnsemble になりこのメソッドがない。run_wf_validation.py 固有の問題。"
created: 2026-05-05
updated: 2026-05-05
---

# Debug: StackedEnsemble feature_name AttributeError

## Symptoms

- **Expected**: extract_feature_ranking() が StackedEnsemble でも動作する
- **Actual**: AttributeError: 'StackedEnsemble' object has no attribute 'feature_name'
- **Reproduction**: run_wf_validation.py 実行時（アンサンブル有効）
- **Impact**: WF検証のみ（バックテストはこのコードパスを通らない）
- **Timeline**: アンサンブル機能追加後に発生

## Current Focus

- hypothesis: confirmed — StackedEnsemble に feature_name() / feature_importance() が未実装
- next_action: fix applied

## Evidence

- `run_wf_validation.py:70-103` — `_extract_all_feature_rankings()` が `extract_feature_ranking(sub.win.hit_model, ...)` を呼び出す
- `walk_forward_cv.py:259` — `extract_feature_ranking()` 内で `model.feature_name()` を呼び出す
- `stacked_ensemble.py` — StackedEnsemble クラスには feature_name() も feature_importance() も定義されていなかった
- docstringには「lgb.Booster インターフェース互換」と記載されているが、互換メソッドが不足していた

## Eliminated

- `_extract_all_feature_rankings()` 自体のバグ — ではなく、呼び出し先の StackedEnsemble に必要なメソッドが欠けていた

## Resolution

- **root_cause**: StackedEnsemble クラスが lgb.Booster インターフェース互換を謳いながら feature_name() と feature_importance() を実装していなかった。ensemble 有効時に hit_model が StackedEnsemble になると extract_feature_ranking() が model.feature_name() を呼び出し AttributeError が発生する。
- **fix**: StackedEnsemble に feature_name() と feature_importance() メソッドを追加。feature_name() は内部 lgbm_model に委譲。feature_importance() は3ベースモデルの正規化平均を返す。テスト6件追加 (TestFeatureNameImportanceCompat)。全1257テストPASS。
