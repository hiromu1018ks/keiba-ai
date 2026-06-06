---
phase: 52-shared-feature-builder-consistency
plan: 03
subsystem: features
tags: [pipeline-consistency, data-cutoff, pfp-verifier, session-manifest, paper-trading]
dependency_graph:
  requires: [52-01, 52-02]
  provides: [DataCutoffManifest, PFPVerifier, SessionManifest, write_session_manifest, get_code_version]
  affects: [src/features/, scripts/run_paper_trading.py]
tech_stack:
  added: []
  patterns: [frozen-dataclass, SHA256-hash, atomic-file-write, git-dirty-detection, fail-fast-validation]
key_files:
  created:
    - src/features/data_cutoff_manifest.py
    - src/features/pipeline_consistency.py
    - src/features/session_manifest.py
    - tests/test_pipeline_consistency.py
  modified:
    - scripts/run_paper_trading.py
decisions:
  - PFPVerifier は FeatureManifest/FeatureState オブジェクトを保持し verify() 時に compute_hash() を再呼出で変更検出
  - DataCutoffManifest.from_config() は戦略マニフェスト未指定時にモデル学習終了日をフォールバック
  - SessionManifest のアトミック書き込みに os.replace を使用 (Windows 対応)
  - run_paper_trading.py の --allow-dirty フラグで開発時の dirty 許可を制御
metrics:
  duration: 605s
  completed: "2026-06-06T05:05:10Z"
  tasks: 2
  files: 4
  tests: 25
---

# Phase 52 Plan 03: PT パイプライン整合性検証インフラ Summary

DataCutoffManifest, PFPVerifier, SessionManifest の3層検証インフラを構築し、
run_paper_trading.py の _run_predict() に startup/pre-race/end の3点検証を統合。
BT 検証済みパイプラインが PT で同一結果を生成することを担保する。

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | DataCutoffManifest + PFPVerifier 新設 | `66cd703` | `src/features/data_cutoff_manifest.py`, `src/features/pipeline_consistency.py`, `tests/test_pipeline_consistency.py` |
| 2 | SessionManifest + PT 3点検証統合 | `5db1089` | `src/features/session_manifest.py`, `scripts/run_paper_trading.py`, `tests/test_pipeline_consistency.py` |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] PFPVerifier のハッシュ変更検出ロジック**
- **Found during:** Task 1
- **Issue:** PFPVerifier がコンストラクタで取得したハッシュを freeze() 時にキャプチャしていたため、verify() 時にオブジェクトのハッシュが変更されても検出できなかった
- **Fix:** FeatureManifest/FeatureState オブジェクト自体を保持し、verify() 時に compute_hash() を再呼出して凍結ハッシュと比較するよう変更
- **Files modified:** `src/features/pipeline_consistency.py`
- **Commit:** `66cd703`

**2. [Rule 3 - Blocking] Windows tempfile の PermissionError**
- **Found during:** Task 1
- **Issue:** tempfile.NamedTemporaryFile(delete=False) が Windows でハンドルをロックし os.unlink() が PermissionError
- **Fix:** tempfile.mkdtemp() + Path.write_text() に変更し、finally ブロックで unlink/rmdir
- **Files modified:** `tests/test_pipeline_consistency.py`
- **Commit:** `66cd703`

**3. [Rule 3 - Blocking] actual_cutoff 辞書の閉じ括弧誤り**
- **Found during:** Task 2
- **Issue:** run_paper_trading.py の actual_cutoff dict を `)` で閉じていた (`}` であるべき)
- **Fix:** `)` を `}` に修正
- **Files modified:** `scripts/run_paper_trading.py`
- **Commit:** `5db1089`

## Verification Results

```
25 tests passed (8 DataCutoff + 7 PFPVerifier + 4 SessionManifest + 3 CodeVersion + 2 WriteSession + 1 Import)
ruff check: All checks passed (src/features/*.py)
Import check: All imports OK
--allow-dirty flag: confirmed in run_paper_trading.py
```

## Key Decisions

1. **PFPVerifier の verify() で再ハッシュ計算**: コンストラクタ時に一度だけハッシュを取得する設計では、オブジェクトの変更を検出できない。FeatureManifest/FeatureState オブジェクト自体を保持し、verify() 呼び出し毎に compute_hash() を再実行することで、パラメータの動的変更を確実に検出する。

2. **DataCutoffManifest.from_config() のフォールバック**: 戦略マニフェストが存在しない・解析に失敗した場合は、モデル学習終了日を全データソースのカットオフ日として使用。PT 実行の安全性を保ちつつ、設定ファイル不在でも動作可能にする。

3. **run_paper_trading.py の3点検証**: startup (freeze + cutoff + session書込み), pre-race (verify), end (verify + session更新) の3箇所に検証を挿入。startup で全パラメータをスナップショットし、pre-race で不変性を確認、end で最終検証して session_manifest に記録。

4. **--allow-dirty フラグ**: デフォルトでは git dirty 状態で PT を実行すると非ゼロ終了。開発時は --allow-dirty で警告のみに緩和可能。

## Threat Flags

計画の `<threat_model>` 通りの対応:
- T-52-06: session_manifest.json はアトミック書き込み (os.replace) + SHA256 ハッシュ
- T-52-07: code hash + manifest hash + PFP result + exit code の監査証跡
- T-52-08: DataCutoffManifest.fail-fast で未来情報の PT 流入を防止

## Known Stubs

なし。全ての公開 API はテスト済み。

## Self-Check: PASSED

- `src/features/data_cutoff_manifest.py`: FOUND
- `src/features/pipeline_consistency.py`: FOUND
- `src/features/session_manifest.py`: FOUND
- `tests/test_pipeline_consistency.py`: FOUND
- `66cd703`: FOUND
- `5db1089`: FOUND
