# H5 full-history comparison

Matched rolling OOT experiment comparing the frozen 120-month H5 TabM recipe with an expanding-history recipe beginning on 2010-01-01. Both use the causal `rare65` policy.

- [Full report](REPORT.md)
- [Selection](selection.json)
- [All reporting metrics](REPORTING_ALL_METRICS.csv)
- [Verification](verification.json)
- [Selected final-fit source bundle](final_fit/bundle.json)

Reproduce with the project environment containing PyTorch, TabM, `rtdl-num-embeddings`, and PyArrow:

```bash
python research_v4/h5_fullhistory/experiment.py
python research_v4/h5_fullhistory/evaluate.py
python research_v4/h5_fullhistory/verify.py
python research_v4/h5_fullhistory/final_fit.py
python research_v4/h5_fullhistory/report.py
```

The final refit is a deployment artifact. Reported quality comes from the annual 2024–2026 rolling OOT recipe, not from evaluating that refit on its own training or calibration data.
