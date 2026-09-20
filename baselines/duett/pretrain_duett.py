"""
- Self-supervised DuETT pretraining on MIMIC.
- Test는 혹시 별도 평가 활용될 수 있으므로 남겨둠.
"""

from __future__ import annotations

import argparse

import pytorch_lightning as pl

import duett
from hourly_datamodule import D_TARGET, MimicPretrainDataModule
from warmup import WarmUpCallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain DuETT on MIMIC (self-supervised).")
    parser.add_argument(
        "--train-pkl",
        default="duett_preprocessing/mimic_splits/finetune_train_24_fold0_final_duett.pkl",
    )
    parser.add_argument(
        "--val-pkl",
        default="duett_preprocessing/mimic_splits/finetune_val_24_fold0_final_duett.pkl",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--early-stop-patience", type=int, default=20, help="Stop if val_loss doesn't improve for this many epochs. 0 disables early stopping.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/mimic_pretrain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pl.seed_everything(args.seed)

    dm = MimicPretrainDataModule(
        args.train_pkl, args.val_pkl, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    dm.setup()
    print(f"MIMIC pretrain: {len(dm.train_ds):,} train stays, {len(dm.val_ds):,} val stays, "
          f"d_static_num={dm.d_static_num()}, d_time_series_num={dm.d_time_series_num()}")

    model = duett.pretrain_model(
        d_static_num=dm.d_static_num(),
        d_time_series_num=dm.d_time_series_num(),
        d_target=D_TARGET,  # unused by the pretrain loss
        pos_frac=None,
        seed=args.seed,
    )

    checkpoint = pl.callbacks.ModelCheckpoint(
        save_last=True, monitor="val_loss", mode="min", save_top_k=1, dirpath=args.checkpoint_dir,
    )
    warmup = WarmUpCallback(steps=args.warmup_steps)
    callbacks = [warmup, checkpoint]
    if args.early_stop_patience > 0:
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=args.early_stop_patience,
        ))

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False,
        num_sanity_val_steps=2,
        max_epochs=args.max_epochs,
        gradient_clip_val=1.0,
        callbacks=callbacks,
    )
    trainer.fit(model, dm)
    print("Best checkpoint:", checkpoint.best_model_path)


if __name__ == "__main__":
    main()
