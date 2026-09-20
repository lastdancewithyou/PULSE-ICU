from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize_seeds(df: pd.DataFrame, metrics: list[str], decimals: int = 4) -> pd.DataFrame:
    grouped = df.groupby("label", sort=False)[metrics].agg(["mean", "std"])
    out = pd.DataFrame(index=grouped.index)
    for m in metrics:
        out[m] = grouped[(m, "mean")].round(decimals).astype(str) + " ± " + grouped[(m, "std")].round(decimals).astype(str)
    return out.reset_index()


def load_rows(paths: list[Path], seeds: list[int]) -> dict[str, pd.DataFrame]:
    binary, sofa, pheno, overall = [], [], [], []
    for path, seed in zip(paths, seeds):
        metrics = json.loads(path.read_text())
        for name, m in metrics["binary"].items():
            binary.append({"label": name, "seed": seed, "auroc": m["auroc"], "auprc": m["auprc"]})
        for name, m in metrics["sofa"].items():
            if "auprc_macro" not in m:
                raise ValueError(
                    f"{path} has no SOFA AUPRC (written by the old finetune_duett.py). "
                    f"Re-evaluate it with: python finetune_duett.py --seed {seed} --eval-only"
                )
            sofa.append({"label": name, "seed": seed, "auroc_macro": m["auroc_macro"], "auprc_macro": m["auprc_macro"]})
        for name, m in metrics["phenotype"].items():
            pheno.append({"label": name, "seed": seed, "auroc": m["auroc"], "auprc": m["auprc"]})
        overall.append({"label": "seed_mean", "seed": seed, **metrics["summary"]})
    return {
        "binary": pd.DataFrame(binary),
        "sofa": pd.DataFrame(sofa),
        "phenotype": pd.DataFrame(pheno),
        "overall": pd.DataFrame(overall),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate DuETT finetune results over seeds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    args = parser.parse_args()

    tag = "randominit" if args.random_init else "pretrain"
    root = Path(args.checkpoint_root)
    paths = [root / f"eicu_finetune_{tag}_seed{seed}" / "test_metrics.json" for seed in args.seeds]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing test_metrics.json: {[str(p) for p in missing]}")

    rows = load_rows(paths, args.seeds)
    tables = {
        "binary": summarize_seeds(rows["binary"], ["auroc", "auprc"]),
        "sofa": summarize_seeds(rows["sofa"], ["auroc_macro", "auprc_macro"]),
        "phenotype": summarize_seeds(rows["phenotype"], ["auroc", "auprc"]),
        "overall": summarize_seeds(
            rows["overall"],
            [
                "binary_mean_auroc", "binary_mean_auprc", "sofa_mean_auroc", "sofa_mean_auprc",
                "phenotype_mean_auroc", "phenotype_mean_auprc",
            ],
        ),
    }

    pd.set_option("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 200)
    for name, table in tables.items():
        print(f"\n[{name}] {tag}, seeds={args.seeds}")
        print(table.to_string(index=False))

    out_path = root / f"eicu_finetune_{tag}_seeds{'-'.join(map(str, args.seeds))}_summary.json"
    out_path.write_text(json.dumps(
        {"seeds": args.seeds, "tag": tag, **{k: v.to_dict(orient="records") for k, v in tables.items()}},
        indent=2, ensure_ascii=False,
    ))
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
