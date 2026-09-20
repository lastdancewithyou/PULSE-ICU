from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from hourly_datamodule import (
    BINARY_LABELS, D_TARGET, PHENOTYPE_LABELS, SOFA_LABELS, EicuFinetuneDataModule,
)
from multitask_duett import finetune_multitask_model, random_init_multitask_model
from warmup import WarmUpCallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finetune DuETT on eICU (multi-task).")
    parser.add_argument(
        "--pretrain-ckpt",
        default="checkpoints/mimic_pretrain/epoch=139-step=79240.ckpt",
    )
    parser.add_argument("--random-init", action="store_true", help="Ignore --pretrain-ckpt and train from scratch.")
    parser.add_argument(
        "--train-pkl",
        default="duett_preprocessing/eicu_splits/finetune_train_24_fold0_eicu_duett.pkl",
    )
    parser.add_argument(
        "--val-pkl",
        default="duett_preprocessing/eicu_splits/finetune_val_24_fold0_eicu_duett.pkl",
    )
    parser.add_argument(
        "--test-pkl",
        default="duett_preprocessing/eicu_splits/finetune_test_24_eicu_fold0_duett.pkl",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Stop if val_auroc doesn't improve for this many epochs. 0 disables early stopping.")
    parser.add_argument("--exp-name", default=None, help="Defaults to eicu_finetune_{pretrain|randominit}_seed<seed>.")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; re-evaluate the best checkpoint already in <checkpoint-dir> and rewrite test_metrics.json.",
    )
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", type=int, nargs="+", default=[1], help="GPU ids, e.g. --devices 1")
    return parser


class CollectTestOutputs(pl.Callback):
    """MultiTaskDuett.test_step returns per-batch probs/labels; Lightning does not keep them, so gather here."""

    def __init__(self):
        self.outputs: dict[str, list[torch.Tensor]] = {}

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        for key, value in outputs.items():
            self.outputs.setdefault(key, []).append(value)

    def stacked(self) -> dict[str, np.ndarray]:
        return {key: torch.cat(values).numpy() for key, values in self.outputs.items()}


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float] | None:
    if len(np.unique(y_true)) < 2:
        return None
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "prevalence": float(y_true.mean()),
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def compute_test_metrics(out: dict[str, np.ndarray]) -> dict:
    binary = {}
    for i, name in enumerate(BINARY_LABELS):
        m = _binary_metrics(out["y_binary"][:, i], out["binary_probs"][:, i])
        if m is not None:
            binary[name] = m

    sofa = {}
    for i, name in enumerate(SOFA_LABELS):
        y, probs = out["y_sofa"][:, i], out["sofa_probs"][:, i, :]
        # Macro one-vs-rest over the classes that have both positives and negatives
        # (same convention as auroc_macro/auprc_macro in eicu_baseline_new.ipynb).
        classes = [c for c in range(probs.shape[1]) if 0 < (y == c).sum() < len(y)]
        if classes:
            sofa[name] = {
                "auroc_macro": float(np.mean([roc_auc_score(y == c, probs[:, c]) for c in classes])),
                "auprc_macro": float(np.mean([average_precision_score(y == c, probs[:, c]) for c in classes])),
                "class_prevalence": [float((y == c).mean()) for c in range(probs.shape[1])],
            }

    pheno = {}
    for i, name in enumerate(PHENOTYPE_LABELS):
        m = _binary_metrics(out["y_pheno"][:, i], out["pheno_probs"][:, i])
        if m is not None:
            pheno[name] = m

    return {
        "n_test": int(out["y_binary"].shape[0]),
        "binary": binary,
        "sofa": sofa,
        "phenotype": pheno,
        "summary": {
            "binary_mean_auroc": _mean([m["auroc"] for m in binary.values()]),
            "binary_mean_auprc": _mean([m["auprc"] for m in binary.values()]),
            "sofa_mean_auroc": _mean([m["auroc_macro"] for m in sofa.values()]),
            "sofa_mean_auprc": _mean([m["auprc_macro"] for m in sofa.values()]),
            "phenotype_mean_auroc": _mean([m["auroc"] for m in pheno.values()]),
            "phenotype_mean_auprc": _mean([m["auprc"] for m in pheno.values()]),
        },
    }


def print_per_task(metrics: dict) -> None:
    print(f"\n[Test per-task results] n={metrics['n_test']:,}")
    print(f"{'task':<62}{'AUROC':>8}{'AUPRC':>8}{'prev':>8}   (sofa prev = class 0/1/2/3 fractions)")
    for group in ("binary", "phenotype"):
        for name, m in metrics[group].items():
            print(f"{group[:5] + ' | ' + name:<62}{m['auroc']:>8.4f}{m['auprc']:>8.4f}{m['prevalence']:>8.3f}")
    for name, m in metrics["sofa"].items():
        prev = "/".join(f"{p:.3f}" for p in m["class_prevalence"])
        print(f"{'sofa  | ' + name:<62}{m['auroc_macro']:>8.4f}{m['auprc_macro']:>8.4f}   {prev}")


def main() -> None:
    args = build_parser().parse_args()
    pl.seed_everything(args.seed)

    init_tag = "randominit" if args.random_init else "pretrain"
    exp_name = args.exp_name or f"eicu_finetune_{init_tag}_seed{args.seed}"
    checkpoint_dir = Path(args.checkpoint_root) / exp_name

    dm = EicuFinetuneDataModule(
        args.train_pkl, args.val_pkl, args.test_pkl,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    dm.setup()
    print(f"eICU finetune ({init_tag}): {len(dm.train_ds):,} train, {len(dm.val_ds):,} val, "
          f"{len(dm.test_ds):,} test stays, d_static_num={dm.d_static_num()}, "
          f"d_time_series_num={dm.d_time_series_num()}")

    model_kwargs = dict(
        d_static_num=dm.d_static_num(),
        d_time_series_num=dm.d_time_series_num(),
        d_target=D_TARGET,
        pos_frac=None,
        seed=args.seed,
    )
    if args.random_init:
        model = random_init_multitask_model(**model_kwargs)
    else:
        ckpt_path = Path(args.pretrain_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Pretraining checkpoint not found: {ckpt_path}")
        model = finetune_multitask_model(str(ckpt_path), **model_kwargs)
        print(f"Loaded pretrained weights: {ckpt_path}")

    checkpoint = pl.callbacks.ModelCheckpoint(
        save_last=True, monitor="val_auroc", mode="max", save_top_k=1, dirpath=checkpoint_dir,
    )
    callbacks = [WarmUpCallback(steps=args.warmup_steps), checkpoint]
    if args.early_stop_patience > 0:
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor="val_auroc", mode="max", patience=args.early_stop_patience,
        ))

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False,
        max_epochs=args.max_epochs,
        gradient_clip_val=1.0,
        callbacks=callbacks,
    )
    if args.eval_only:
        best_ckpts = [p for p in checkpoint_dir.glob("*.ckpt") if not p.name.startswith("last")]
        if len(best_ckpts) != 1:
            raise FileNotFoundError(f"Expected exactly one best checkpoint in {checkpoint_dir}, found {best_ckpts}")
        best_path = str(best_ckpts[0])
    else:
        trainer.fit(model, dm)
        best_path = checkpoint.best_model_path
    print("Best checkpoint:", best_path)

    collector = CollectTestOutputs()
    trainer.callbacks.append(collector)
    trainer.test(model, dm, ckpt_path=best_path)

    metrics = compute_test_metrics(collector.stacked())
    metrics["args"] = vars(args)
    metrics["best_checkpoint"] = best_path
    out_path = checkpoint_dir / "test_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print_per_task(metrics)
    print("\n[Summary]")
    print(json.dumps(metrics["summary"], indent=2))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
