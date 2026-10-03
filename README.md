# Pitch Deck Math Benchmark

Code for a two-part Kaggle Benchmarking Challenge submission testing whether LLMs catch planted math errors in realistic startup pitch language, instead of validating them.

Full write-up: [paste your DEV.to post URL here]

## Files

- `cac_ltv_check.py` — Tests whether models catch a wrong LTV:CAC ratio and an impossible growth-compounding claim, plus two calibration controls confirming the harness isn't noise.
- `threshold_probe.py` — Tests how small a lie a model still catches, across four sizes of overstatement (2x, 1.5x, 1.2x, 1.1x the truth), scored by extracting the actual numbers from the response rather than an LLM judge.

## Kaggle Tasks

- [cac_ltv_check](https://www.kaggle.com/benchmarks/tasks/ritamgit/cac-ltv-check)
- [threshold_probe](https://www.kaggle.com/benchmarks/tasks/ritamgit/threshold-probe)
