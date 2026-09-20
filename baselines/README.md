# Baselines

Baselines compared against PULSE-ICU. This folder contains our own code only; third-party
code (TSMixer, iTransformer) is **not** included and has to be fetched from upstream (see below).

## How the baselines were run

- **CatBoost, XGBoost, LSTM, TSMixer, iTransformer** were run interactively in a single
  notebook, [`eicu_baselines_experiment.ipynb`](eicu_baselines_experiment.ipynb) (eICU).
- **DuETT** was run as standalone Python scripts ([`duett/`](duett/)) rather than in a notebook,
  because it uses its own Lightning-based pipeline and data format (hourly-binned inputs).
  It covers MIMIC pretraining and eICU fine-tuning.

| Baseline | Code | Upstream (commit we used) | Changes to upstream |
|---|---|---|---|
| CatBoost, XGBoost, LSTM | notebook | `pip` packages | - |
| TSMixer | fetch to `baselines/tsmixer/` | [ditschuk/pytorch-tsmixer](https://github.com/ditschuk/pytorch-tsmixer) `eefb040` (MIT) | none |
| iTransformer | fetch to `baselines/iTransformer/` | [thuml/iTransformer](https://github.com/thuml/iTransformer) `c2426e6` (MIT) | one unused import removed (see below) |
| DuETT | `duett/` | [layer6ai-labs/DuETT](https://github.com/layer6ai-labs/DuETT) `a99282c` | `duett/duett.py` (see below) |

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

## Third-party code

**TSMixer** (used unmodified). The notebook imports `from tsmixer.tsmixer import TSMixer`, so the upstream
package directory `torchtsmixer/` is used under the name `tsmixer/`:

```bash
git clone https://github.com/ditschuk/pytorch-tsmixer.git /tmp/pytorch-tsmixer
git -C /tmp/pytorch-tsmixer checkout eefb040
cp -r /tmp/pytorch-tsmixer/torchtsmixer baselines/tsmixer
```

**iTransformer**:

```bash
git clone https://github.com/thuml/iTransformer.git baselines/iTransformer
git -C baselines/iTransformer checkout c2426e6
```

We removed one line, `from reformer_pytorch import LSHSelfAttention`, from `layers/SelfAttention_Family.py`.
It is only used by the Reformer variants (`ReformerLayer`), not by iTransformer itself, so we avoided
installing `reformer_pytorch`. Installing it instead also works and leaves upstream fully unmodified.

**DuETT**: `duett/duett.py` is derived from [layer6ai-labs/DuETT](https://github.com/layer6ai-labs/DuETT)
(commit `a99282c`) with minor compatibility edits for PyTorch Lightning 2.x / torchmetrics 1.x.
The other upstream files (`train.py`, `physionet.py`, ...) are not needed and not included.

In the notebook, point the two `sys.path.insert(...)` lines to your checkout: the `baselines/` directory
for `tsmixer`, and `baselines/iTransformer` for iTransformer.

Tested with: torch 2.4.0, pytorch-lightning 2.4.0, torchmetrics 1.4.1, x-transformers 1.5.3,
catboost 1.2.10, xgboost 2.1.4.

## Data

No data is included. MIMIC-IV and eICU require credentialed PhysioNet access.
The scripts read the outputs of the PULSE-ICU preprocessing pipeline (see the repository root).

## Running

**Notebook baselines:** open `eicu_baselines_experiment.ipynb` and run top to bottom.

**DuETT** (run from the repository root unless noted):
1. `bash baselines/duett/duett_preprocessing/run_duett_preprocessing_mimic.sh` and `..._eicu.sh`:
   build DuETT's hourly-binned inputs. Needs `unit_map.csv` and `YAIB_label_map.xlsx` (included);
   outputs go to `baselines/duett/duett_preprocessing/{mimic,eicu}_splits/`.
2. `cd baselines/duett && python pretrain_duett.py`: self-supervised pretraining on MIMIC
   (checkpoints in `checkpoints/mimic_pretrain/`).
3. `bash run_duett_seeds.sh` (inside `baselines/duett`): multi-task fine-tuning on eICU for seeds 41/42/43,
   then `aggregate_duett.py`. `RANDOM_INIT=1 bash run_duett_seeds.sh` trains the same architecture from scratch.

## Attribution

Please cite the original papers: iTransformer, TSMixer, DuETT.
