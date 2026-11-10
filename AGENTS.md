# Working on this research repository

- Read `README.md`, `docs/research_protocol_v2.md`, and the relevant version's protocol before changing research behavior. User instructions take precedence.
- Keep published signals, historical reports, frozen NPZ weights and model cards unchanged. Put corrected research in a new version/output directory; identify it as a historical replay or exploration when appropriate.
- The main 555 comparison uses the same 444 forecasts plus allocation/news/risk. Preserve the explicit legacy notebook entry points.
- A signal made after close cannot fill at that same close. Five-session targets enter at `i+1` and end at `i+6`; purge on the full session calendar. Fit scalers only on training input rows.
- V3's current benchmark is 0050. All previously examined history is exploratory. Test returns never choose a candidate, and no result is assumed to beat the benchmark.
- V3 is isolated from daily production. Its default CLI is read-only. Use a fresh run directory for training; never auto-refresh a frozen bundle to make a failed check disappear.
- FinMind's current terms prohibit redistributing provider raw data. Keep new raw responses and full normalized price caches outside Git, public ZIPs and CI artifacts. Publish derived summaries, model artifacts, source hashes and acquisition code instead. See `research/v3/data_contract.md`.
- Run `python -m pytest -q` for code changes. Run `python -m scripts.verify_models` after reviewed inference-source changes; verify old predictions before explicitly maintaining the bundle manifest.
- Record actual validation evidence and remaining limitations. Passing unit tests does not certify missing historical source or publication timestamps.
