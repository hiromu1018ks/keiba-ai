---
status: fixing
trigger: "prune_noise_features.py --apply --full-bt --rollback fails with UnicodeDecodeError cp932"
created: 2026-05-12T00:00:00Z
updated: 2026-05-12T00:00:00Z
---

## Current Focus

hypothesis: subprocess.run() in run_full_bt_roi_check() lacks encoding='utf-8', so Windows defaults to cp932 which cannot decode UTF-8 Japanese output from run_backtest.py
test: Add encoding='utf-8' and errors='replace' to the subprocess.run() call at line 529, then run tests
expecting: Tests pass, subprocess no longer crashes on encoding
next_action: Apply the encoding fix to line 529-534 and run tests

## Symptoms

expected: prune_noise_features.py --apply --full-bt --rollback runs backtest via subprocess and compares ROI
actual: UnicodeDecodeError: 'cp932' codec can't decode byte 0x84 in position 953: illegal multibyte sequence
errors: UnicodeDecodeError on subprocess.run()
reproduction: Run prune_noise_features.py --apply --full-bt --rollback on Windows
started: Since script was written (always broken on Windows)

## Eliminated

(none needed - root cause is clear from code inspection)

## Evidence

- timestamp: 2026-05-12
  checked: subprocess.run() call at line 529-534 of scripts/prune_noise_features.py
  found: No encoding parameter specified. On Windows, text=True uses locale.getpreferredencoding() which returns cp932 for Japanese locale. run_backtest.py outputs UTF-8 Japanese text.
  implication: cp932 decoder chokes on UTF-8 byte sequences. Fix is to add encoding='utf-8'.

- timestamp: 2026-05-12
  checked: src/models/ directory for .backup files and git diff
  found: No .backup files exist, no uncommitted changes to model .py files
  implication: Previous run's rollback succeeded (or modifications were never applied). Model files are clean. No restore needed.

## Resolution

root_cause: subprocess.run() in run_full_bt_roi_check() at line 529 lacks encoding parameter. Windows defaults to cp932 which cannot decode UTF-8 output from run_backtest.py.
fix: Add encoding='utf-8' and errors='replace' to the subprocess.run() call.
verification: 8/8 tests pass. Dry-run completes successfully without errors.
files_changed: [scripts/prune_noise_features.py]
