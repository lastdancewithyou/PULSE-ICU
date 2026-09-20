# Baselines

Baselines compared against PULSE-ICU. This folder contains our own code only; third-party
code (TSMixer, iTransformer) is **not** included and has to be fetched from upstream.

## How the baselines were run

- **CatBoost, XGBoost, LSTM, TSMixer, iTransformer** were run interactively in a single
  notebook, [`eicu_baselines_experiment.ipynb`](eicu_baselines_experiment.ipynb) (eICU).
- **DuETT** was run as standalone Python scripts ([`duett/`](duett/)).
  It covers MIMIC pretraining and eICU fine-tuning.

| Baseline | Code | Upstream | Changes to upstream |
|---|---|---|---|
| CatBoost, XGBoost, LSTM | notebook | `pip` packages | - |
| TSMixer | fetch to `baselines/tsmixer/` | [ditschuk/pytorch-tsmixer](https://github.com/ditschuk/pytorch-tsmixer) | none |
| iTransformer | fetch to `baselines/iTransformer/` | [thuml/iTransformer](https://github.com/thuml/iTransformer)|
| DuETT | `duett/` | [layer6ai-labs/DuETT](https://github.com/layer6ai-labs/DuETT)| `duett/duett.py`

## Layout

```
baselines/
  eicu_baselines_experiment.ipynb   # CatBoost / XGBoost / LSTM / TSMixer / iTransformer on eICU
  duett/                            # DuETT (our scripts + duett.py)
    pretrain_duett.py  finetune_duett.py  multitask_duett.py  hourly_datamodule.py
    warmup.py  aggregate_duett.py  run_duett_seeds.sh  duett.py
    duett_preprocessing/            # MIMIC / eICU -> hourly-binned DuETT inputs
  tsmixer/                          # NOT in the repo: fetch as below
  iTransformer/                     # NOT in the repo: fetch as below
```

**DuETT**: `duett/duett.py` is derived from [layer6ai-labs/DuETT](https://github.com/layer6ai-labs/DuETT).
The other upstream files (`train.py`, `physionet.py`, ...) are not needed and not included.

Tested with: torch 2.4.0, pytorch-lightning 2.4.0, torchmetrics 1.4.1, x-transformers 1.5.3,
catboost 1.2.10, xgboost 2.1.4.

## Data

No data is included. MIMIC-IV and eICU require credentialed PhysioNet access.
The scripts read the outputs of the PULSE-ICU preprocessing pipeline (see the repository root).