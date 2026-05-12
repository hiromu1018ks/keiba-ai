# GSD Debug Knowledge Base

Resolved debug sessions. Used by `gsd-debugger` to surface known-pattern hypotheses at the start of new investigations.

---

## prune-subprocess-encoding — subprocess.run() on Windows crashes with cp932 UnicodeDecodeError
- **Date:** 2026-05-12
- **Error patterns:** UnicodeDecodeError, cp932, codec, decode, byte, illegal multibyte sequence, subprocess, encoding, Windows
- **Root cause:** subprocess.run() with text=True but no encoding parameter defaults to locale encoding (cp932 on Japanese Windows), which cannot decode UTF-8 output from child process
- **Fix:** Add encoding='utf-8' and errors='replace' to subprocess.run() call
- **Files changed:** scripts/prune_noise_features.py
---

