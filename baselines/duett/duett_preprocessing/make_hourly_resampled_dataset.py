#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
import re
import time
from pathlib import Path
from typing import Iterable

_STAGE_START = time.time()


def stage(msg: str) -> None:
    print(f"[{time.time() - _STAGE_START:7.1f}s] {msg}", flush=True)


DEFAULT_LABEL_COLS = [
    "mortality_48hr",
    "mortality_30",
    "mortality_inicu",
    "mortality_inhospital",
    "los_3days",
    "los_7days",
    "readmission_30",
    "transfusion_12hr",
    "shock_8hr",
    "vasopressor_need_12hr",
    "ventilator_need_12hr",
    "SOFA_centralnervous_24hr",
    "SOFA_cardiovascular_24hr",
    "SOFA_respiratory_24hr",
    "SOFA_coagulation_24hr",
    "SOFA_liver_24hr",
    "SOFA_renal_24hr",
    "Acute and unspecified renal failure",
    "Acute cerebrovascular disease",
    "Acute myocardial infarction",
    "Cardiac dysrhythmias",
    "Chronic kidney disease",
    "Chronic obstructive pulmonary disease and bronchiectasis",
    "Complications of surgical procedures or medical care",
    "Conduction disorders",
    "Congestive heart failure; nonhypertensive",
    "Coronary atherosclerosis and other heart disease",
    "Diabetes mellitus with complications",
    "Diabetes mellitus without complication",
    "Disorders of lipid metabolism",
    "Essential hypertension",
    "Fluid and electrolyte disorders",
    "Gastrointestinal hemorrhage",
    "Hypertension with complications and secondary hypertension",
    "Other liver diseases",
    "Other lower respiratory disease",
    "Other upper respiratory disease",
    "Pleurisy; pneumothorax; pulmonary collapse",
    "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
    "Respiratory failure; insufficiency; arrest (adult)",
    "Septicemia (except in labor)",
    "Shock",
]


def load_runtime_dependencies() -> None:
    global np, pd

    import numpy as np
    import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create hourly-resampled MIMIC/eICU datasets from preprocessed events."
    )
    parser.add_argument(
        "--sequence-path",
        default="datasets/mimic_preprocessed_24_final.parquet",
        help="Preprocessed irregular event parquet.",
    )
    parser.add_argument(
        "--cohort-path",
        default="cohort/mimic_cohort_24_final.parquet",
        help="Cohort/label parquet with one row per stay_id.",
    )
    parser.add_argument("--output-dir", default="hourly_resampled")
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument(
        "--num-time-bins",
        type=int,
        default=None,
        help=(
            "Number of equal-width bins across the observation window. "
            "Defaults to 32 with --duett and to --window-hours otherwise."
        ),
    )
    parser.add_argument(
        "--feature-col",
        default="label_idx",
        choices=["label_idx", "itemid", "label"],
        help="Variable identity used as hourly feature columns.",
    )
    parser.add_argument(
        "--value-col",
        default="value",
        help="Numeric event value to aggregate in each hour.",
    )
    parser.add_argument(
        "--source-col",
        default="source_table",
        help="Column identifying chart/procedure/medication source tables.",
    )
    parser.add_argument(
        "--unit-col",
        default=None,
        help="Optional unit column name. If omitted, valueuom_idx or unit_idx is used when present.",
    )
    parser.add_argument(
        "--source-aware",
        action="store_true",
        default=True,
        help="Use source-specific hourly rules: chart=last, medication=sum, procedure=presence.",
    )
    parser.add_argument(
        "--no-source-aware",
        action="store_false",
        dest="source_aware",
        help="Use one aggregation rule for every source.",
    )
    parser.add_argument("--chart-agg", default="last", choices=["mean", "median", "min", "max", "first", "last"])
    parser.add_argument("--medication-agg", default="sum", choices=["sum", "mean", "median", "min", "max", "first", "last"])
    parser.add_argument(
        "--agg",
        default="mean",
        choices=["mean", "median", "min", "max", "first", "last"],
        help="Aggregation for repeated measurements in the same stay-hour-feature.",
    )
    parser.add_argument(
        "--fill",
        default="zero",
        choices=["none", "zero", "ffill", "ffill_zero"],
        help="Missing-value fill after hourly pivot. Mask columns always mark observed bins before fill.",
    )
    parser.add_argument(
        "--ffill-limit",
        type=int,
        default=None,
        help="Optional maximum number of hours to forward-fill per stay.",
    )
    parser.add_argument(
        "--min-observed-hours",
        type=int,
        default=1,
        help="Drop stays with fewer observed hours before grid expansion.",
    )
    parser.add_argument(
        "--sample-stays",
        type=int,
        default=None,
        help="Use only N stay_id values for a quick local smoke test.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Use a fraction of stay_id values for a quick local smoke test.",
    )
    parser.add_argument(
        "--sample-random-state",
        type=int,
        default=42,
        help="Random seed for --sample-stays/--sample-frac.",
    )
    parser.add_argument(
        "--exclude-invalid-mortality48",
        action="store_true",
        help="Drop cohort rows where mortality_48hr == -1.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Label columns to include in the pickle. Default: known task labels present in cohort.",
    )
    parser.add_argument(
        "--save-format",
        choices=["parquet", "pickle", "both"],
        default="both",
    )
    parser.add_argument(
        "--split-pkl-dir",
        default=None,
        help=(
            "Optional irregular split pickle directory. When set, save hourly "
            "pickles with the same train/val/test stay_ids as files such as "
            "finetune_train_24_fold0_final.pkl."
        ),
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=[0],
        default=0,
        help="Use fold 0 only (fixed for the DuETT comparison).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Use train-only feature mean/std normalization for split pickles.",
    )
    parser.add_argument(
        "--add-time-since",
        action="store_true",
        default=True,
        help="Include hours since each variable was last observed in pickle items.",
    )
    parser.add_argument(
        "--no-time-since",
        action="store_false",
        dest="add_time_since",
        help="Do not include time-since-observed arrays.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Default is derived from window and feature column.",
    )
    parser.add_argument(
        "--duett",
        action="store_true",
        help=(
            "Save DuETT-ready fields: x_ts=[normalized values || observation counts], "
            "x_static, times in fractional days, and y. Output names use a _duett suffix."
        ),
    )
    parser.add_argument(
        "--duett-feature-map",
        default="YAIB_label_map.xlsx",
        help="YAIB label map defining the shared MIMIC/eICU DuETT feature vocabulary.",
    )
    return parser


def require_columns(df, columns: Iterable[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def feature_name(value: object) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    return f"f_{safe}"


def normalize_source_name(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "chartevents", "labevents", "vitalperiodic", "vitalaperiodic", "chart"}:
        return "chart"
    if text in {"0", "0.0", "inputevents", "medication", "infusiondrug", "input"}:
        return "medication"
    if text in {"2", "2.0", "procedureevents", "procedure", "treatment"}:
        return "procedure"
    return "other"


def load_duett_feature_spec(path: str) -> list[dict]:
    label_map = pd.read_excel(path, sheet_name="label_map")
    require_columns(
        label_map,
        ["label_idx", "mimic_label", "eicu_label", "source_table_idx", "use_input"],
        f"{path}:label_map",
    )
    selected = label_map[
        label_map["eicu_label"].notna() & label_map["use_input"].eq("y")
    ]
    spec = []
    seen = set()
    for row in selected.itertuples(index=False):
        label_idx = int(row.label_idx)
        source = normalize_source_name(row.source_table_idx)
        key = (source, label_idx)
        if key in seen:
            continue
        seen.add(key)
        spec.append(
            {
                "label_idx": label_idx,
                "source": source,
                "mimic_label": str(row.mimic_label),
                "feature_col": f"{source}__{feature_name(label_idx)}",
            }
        )
    if not spec:
        raise ValueError(f"No shared eICU input variables found in {path}:label_map.")
    return spec


def filter_duett_sequence(sequence, args: argparse.Namespace, spec: list[dict]):
    allowed = {(item["source"], item["label_idx"]) for item in spec}
    source = sequence[args.source_col].map(normalize_source_name)
    feature_ids = pd.to_numeric(sequence[args.feature_col], errors="coerce")
    keep = pd.Series(
        [(src, int(fid)) in allowed if pd.notna(fid) else False for src, fid in zip(source, feature_ids)],
        index=sequence.index,
    )
    return sequence[keep].copy()


def align_duett_columns(wide, mask, counts, spec: list[dict]):
    feature_cols = [item["feature_col"] for item in spec]
    wide = wide.reindex(columns=feature_cols)
    mask = wide.notna().astype("int8")
    mask = mask.rename(columns={col: f"mask_{col}" for col in feature_cols})
    count_cols = [f"count_{col}" for col in feature_cols]
    counts = counts.reindex(columns=count_cols, fill_value=0)
    return wide, mask, counts


def load_inputs(args: argparse.Namespace):
    sequence = pd.read_parquet(args.sequence_path)
    cohort = pd.read_parquet(args.cohort_path)

    if args.source_aware and args.source_col not in sequence.columns and "source_table_idx" in sequence.columns:
        args.source_col = "source_table_idx"
    if args.unit_col is None:
        if "valueuom_idx" in sequence.columns:
            args.unit_col = "valueuom_idx"
        elif "unit_idx" in sequence.columns:
            args.unit_col = "unit_idx"

    require_columns(
        sequence,
        ["stay_id", "intime_offset_hr", args.feature_col],
        args.sequence_path,
    )
    if args.source_aware:
        require_columns(sequence, [args.source_col], args.sequence_path)
    else:
        require_columns(sequence, [args.value_col], args.sequence_path)
    require_columns(cohort, ["stay_id", "age", "gender"], args.cohort_path)

    sequence = sequence.copy()
    if args.value_col in sequence.columns:
        sequence[args.value_col] = pd.to_numeric(sequence[args.value_col], errors="coerce")
    if args.source_aware:
        source = sequence[args.source_col].map(normalize_source_name)
        procedure_mask = source.eq("procedure")
        sequence = sequence.dropna(subset=["intime_offset_hr", args.feature_col])
        if args.value_col in sequence.columns:
            sequence = sequence[procedure_mask | sequence[args.value_col].notna()].copy()
    else:
        sequence = sequence.dropna(subset=[args.value_col, "intime_offset_hr", args.feature_col])
    sequence = sequence[
        (sequence["intime_offset_hr"] >= 0)
        & (sequence["intime_offset_hr"] < args.window_hours)
    ].copy()
    bin_width_hours = args.window_hours / args.num_time_bins
    sequence["hour"] = np.floor(
        sequence["intime_offset_hr"] / bin_width_hours
    ).astype(int)
    sequence["hour"] = sequence["hour"].clip(0, args.num_time_bins - 1)

    if args.exclude_invalid_mortality48 and "mortality_48hr" in cohort.columns:
        cohort = cohort[cohort["mortality_48hr"] != -1].copy()
        sequence = sequence[sequence["stay_id"].isin(cohort["stay_id"])].copy()

    observed_hours = sequence.groupby("stay_id")["hour"].nunique()
    keep_stays = observed_hours[observed_hours >= args.min_observed_hours].index
    sequence = sequence[sequence["stay_id"].isin(keep_stays)].copy()
    cohort = cohort[cohort["stay_id"].isin(sequence["stay_id"].unique())].copy()

    if args.sample_stays is not None or args.sample_frac is not None:
        stay_ids = pd.Series(cohort["stay_id"].drop_duplicates().to_numpy())
        if args.split_pkl_dir is not None:
            split_ids = []
            split_files = restrict_split_files_to_fold(
                find_split_files(Path(args.split_pkl_dir), args.window_hours),
                args.fold,
            )
            for split_file in split_files:
                split_ids.extend(load_split_ids(split_file))
            if split_ids:
                stay_ids = stay_ids[stay_ids.isin(set(split_ids))]
            if stay_ids.empty:
                raise ValueError(
                    "No overlap between cohort stay_id values and split pickle stay_id values."
                )
        if args.sample_frac is not None:
            if not 0 < args.sample_frac <= 1:
                raise ValueError("--sample-frac must be in the range (0, 1].")
            sample_n = max(1, int(len(stay_ids) * args.sample_frac))
        else:
            sample_n = min(int(args.sample_stays), len(stay_ids))
        sampled_stays = stay_ids.sample(
            n=sample_n, random_state=args.sample_random_state
        ).to_numpy()
        sequence = sequence[sequence["stay_id"].isin(sampled_stays)].copy()
        cohort = cohort[cohort["stay_id"].isin(sampled_stays)].copy()
        print(f"Smoke-test sample: {len(sampled_stays)} stay_id values")

    return sequence, cohort


def aggregate_one_source(df, args: argparse.Namespace, source_name: str, agg: str):
    data = df.copy()
    data["_hourly_feature"] = (
        source_name + "__" + data[args.feature_col].map(feature_name)
    )
    group_cols = ["stay_id", "hour", "_hourly_feature"]
    grouped = data.groupby(group_cols, sort=False)[args.value_col].agg(agg).reset_index()
    counts = data.groupby(group_cols, sort=False).size().reset_index(name="_observation_count")
    return grouped, counts


def add_interval_offsets(df):
    data = df.copy()
    if {"starttime", "endtime", "intime"}.issubset(data.columns):
        data["start_hr"] = (
            pd.to_datetime(data["starttime"]) - pd.to_datetime(data["intime"])
        ).dt.total_seconds() / 3600.0
        data["end_hr"] = (
            pd.to_datetime(data["endtime"]) - pd.to_datetime(data["intime"])
        ).dt.total_seconds() / 3600.0
    elif {"start_hr", "end_hr"}.issubset(data.columns):
        pass
    else:
        data["start_hr"] = data["intime_offset_hr"]
        data["end_hr"] = data["intime_offset_hr"]
    data["start_hr"] = data["start_hr"].fillna(data["intime_offset_hr"])
    data["end_hr"] = data["end_hr"].fillna(data["start_hr"])
    data["end_hr"] = data[["start_hr", "end_hr"]].max(axis=1)
    return data


def expand_medication_intervals(df, args: argparse.Namespace):
    """Allocate total medication amount into overlapped hourly bins."""
    if df.empty:
        return df.iloc[0:0].copy()

    rows = []
    data = add_interval_offsets(df)
    bin_width_hours = args.window_hours / args.num_time_bins
    for row in data.itertuples(index=False):
        start = max(0.0, float(row.start_hr))
        end = min(float(args.window_hours), float(row.end_hr))
        value = getattr(row, args.value_col)
        if not np.isfinite(value):
            continue

        if end <= start:
            hour = int(np.floor(start / bin_width_hours))
            if 0 <= hour < args.num_time_bins:
                record = row._asdict()
                record["hour"] = hour
                record[args.value_col] = float(value)
                rows.append(record)
            continue

        duration = end - start
        first_bin = int(np.floor(start / bin_width_hours))
        last_bin_exclusive = int(np.ceil(end / bin_width_hours))
        for hour in range(first_bin, last_bin_exclusive):
            if hour < 0 or hour >= args.num_time_bins:
                continue
            bin_start = hour * bin_width_hours
            bin_end = (hour + 1) * bin_width_hours
            overlap = max(0.0, min(end, bin_end) - max(start, bin_start))
            if overlap <= 0:
                continue
            record = row._asdict()
            record["hour"] = hour
            record[args.value_col] = float(value) * (overlap / duration)
            rows.append(record)

    return pd.DataFrame(rows)


def expand_procedure_intervals(df, args: argparse.Namespace):
    """Mark every hourly bin overlapped by a procedure interval as present."""
    if df.empty:
        return df.iloc[0:0].copy()

    rows = []
    data = add_interval_offsets(df)
    bin_width_hours = args.window_hours / args.num_time_bins
    for row in data.itertuples(index=False):
        start = max(0.0, float(row.start_hr))
        end = min(float(args.window_hours), float(row.end_hr))
        if end <= start:
            hours = [int(np.floor(start / bin_width_hours))]
        else:
            hours = range(
                int(np.floor(start / bin_width_hours)),
                int(np.ceil(end / bin_width_hours)),
            )

        for hour in hours:
            if hour < 0 or hour >= args.num_time_bins:
                continue
            if end > start:
                bin_start = hour * bin_width_hours
                bin_end = (hour + 1) * bin_width_hours
                overlap = max(0.0, min(end, bin_end) - max(start, bin_start))
                if overlap <= 0:
                    continue
            record = row._asdict()
            record["hour"] = hour
            record[args.value_col] = 1
            rows.append(record)

    return pd.DataFrame(rows)


def aggregate_hourly_source_aware(sequence, args: argparse.Namespace):
    source = sequence[args.source_col].map(normalize_source_name)
    chart = sequence[source.eq("chart")]
    medication = sequence[source.eq("medication")]
    procedure = sequence[source.eq("procedure")]
    other = sequence[~sequence.index.isin(chart.index.union(medication.index).union(procedure.index))]

    parts = []
    count_parts = []
    feature_types = {}

    if not chart.empty:
        chart_grouped, chart_counts = aggregate_one_source(chart, args, "chart", args.chart_agg)
        parts.append(chart_grouped)
        count_parts.append(chart_counts)
        feature_types.update({col: "chart" for col in chart_grouped["_hourly_feature"].unique()})

    if not medication.empty:
        medication = expand_medication_intervals(medication, args)
        if not medication.empty:
            med_grouped, med_counts = aggregate_one_source(
                medication, args, "medication", args.medication_agg
            )
            parts.append(med_grouped)
            count_parts.append(med_counts)
            feature_types.update({col: "medication" for col in med_grouped["_hourly_feature"].unique()})

    if not procedure.empty:
        proc = expand_procedure_intervals(procedure, args)
        if not proc.empty:
            proc["_hourly_feature"] = "procedure__" + proc[args.feature_col].map(feature_name)
            proc_counts = (
                proc.groupby(["stay_id", "hour", "_hourly_feature"], sort=False)
                .size()
                .reset_index(name="_observation_count")
            )
            proc_grouped = proc_counts.rename(columns={"_observation_count": args.value_col}).copy()
            proc_grouped[args.value_col] = (proc_grouped[args.value_col] > 0).astype("int8")
            parts.append(proc_grouped)
            count_parts.append(proc_counts)
            feature_types.update({col: "procedure" for col in proc_grouped["_hourly_feature"].unique()})

    if not other.empty:
        other_grouped, other_counts = aggregate_one_source(other, args, "other", args.agg)
        parts.append(other_grouped)
        count_parts.append(other_counts)
        feature_types.update({col: "other" for col in other_grouped["_hourly_feature"].unique()})

    if not parts:
        raise ValueError("No events remained after source-aware hourly aggregation.")

    grouped = pd.concat(parts, ignore_index=True)
    wide = grouped.pivot_table(
        index=["stay_id", "hour"],
        columns="_hourly_feature",
        values=args.value_col,
        aggfunc="first",
    )
    wide = wide.sort_index(axis=1)
    mask = wide.notna().astype("int8")
    mask = mask.rename(columns={col: f"mask_{col}" for col in mask.columns})
    counts_long = pd.concat(count_parts, ignore_index=True)
    counts = counts_long.pivot_table(
        index=["stay_id", "hour"],
        columns="_hourly_feature",
        values="_observation_count",
        aggfunc="sum",
        fill_value=0,
    )
    counts = counts.reindex(columns=wide.columns, fill_value=0)
    counts = counts.rename(columns={col: f"count_{col}" for col in counts.columns})
    return wide, mask, counts, feature_types


def aggregate_hourly(sequence, args: argparse.Namespace):
    group_cols = ["stay_id", "hour", args.feature_col]
    grouped = sequence.groupby(group_cols, sort=False)[args.value_col].agg(args.agg).reset_index()
    counts_long = sequence.groupby(group_cols, sort=False).size().reset_index(name="_observation_count")
    wide = grouped.pivot_table(
        index=["stay_id", "hour"],
        columns=args.feature_col,
        values=args.value_col,
        aggfunc="first",
    )
    wide = wide.rename(columns={col: feature_name(col) for col in wide.columns})
    wide = wide.sort_index(axis=1)

    mask = wide.notna().astype("int8")
    mask = mask.rename(columns={col: f"mask_{col}" for col in mask.columns})
    counts = counts_long.pivot_table(
        index=["stay_id", "hour"],
        columns=args.feature_col,
        values="_observation_count",
        aggfunc="sum",
        fill_value=0,
    )
    counts = counts.rename(columns={col: f"count_{feature_name(col)}" for col in counts.columns})
    counts = counts.reindex(columns=[f"count_{col}" for col in wide.columns], fill_value=0)
    return wide, mask, counts, {col: "event" for col in wide.columns}


def expand_hourly_grid(wide, mask, counts, cohort, num_time_bins: int):
    stay_ids = cohort["stay_id"].drop_duplicates().sort_values()
    full_index = pd.MultiIndex.from_product(
        [stay_ids, range(num_time_bins)], names=["stay_id", "hour"]
    )
    wide = wide.reindex(full_index)
    mask = mask.reindex(full_index).fillna(0).astype("int8")
    counts = counts.reindex(full_index).fillna(0).astype("int32")
    return wide, mask, counts


def apply_fill(wide, fill: str, ffill_limit: int | None):
    if fill == "none":
        return wide
    if fill in {"ffill", "ffill_zero"}:
        wide = wide.groupby(level="stay_id").ffill(limit=ffill_limit)
    if fill in {"zero", "ffill_zero"}:
        wide = wide.fillna(0)
    return wide


def apply_structural_zeros(wide, feature_types: dict):
    """Medication/procedure absence is a true zero; chart absence stays missing."""
    zero_cols = [
        col
        for col, feature_type in feature_types.items()
        if feature_type in {"medication", "procedure"}
    ]
    zero_cols = [col for col in zero_cols if col in wide.columns]
    if zero_cols:
        wide = wide.copy()
        wide[zero_cols] = wide[zero_cols].fillna(0)
    return wide


def compute_time_since(mask, bin_width_hours: float = 1.0):
    """Return per-feature hours since last observation."""
    result = mask.copy().astype("float32")
    num_time_bins = int(mask.index.get_level_values("hour").max()) + 1
    cap = float(num_time_bins) * bin_width_hours
    for stay_id, stay_mask in mask.groupby(level="stay_id", sort=False):
        arr = stay_mask.to_numpy(dtype="int8")
        out = np.zeros(arr.shape, dtype="float32")
        last_seen = np.full(arr.shape[1], np.nan, dtype="float32")
        for hour in range(arr.shape[0]):
            observed = arr[hour] == 1
            last_seen[observed] = hour
            out[hour] = np.where(
                np.isnan(last_seen), cap, (hour - last_seen) * bin_width_hours
            )
        result.loc[stay_id] = out
    result = result.rename(columns={col: col.replace("mask_", "delta_") for col in result.columns})
    return result


def load_split_ids(path: Path) -> list:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} does not contain a dict.")
    ids = []
    for key, item in data.items():
        if key == "_meta":
            continue
        if isinstance(item, dict):
            ids.append(item.get("original_stay_id", item.get("stay_id", key)))
        else:
            ids.append(key)
    return ids


def find_split_files(split_dir: Path, window_hours: int) -> list[Path]:
    patterns = [
        f"finetune_train_{window_hours}_fold*_final*.pkl",
        f"finetune_val_{window_hours}_fold*_final*.pkl",
        f"finetune_test_{window_hours}_final*.pkl",
    ]
    files = []
    for pattern in patterns:
        files.extend(sorted(split_dir.glob(pattern)))
    return files


def restrict_split_files_to_fold(files: list[Path], fold: int | None) -> list[Path]:
    if fold is None:
        return files
    fold_tag = f"_fold{fold}"
    selected = []
    for path in files:
        name = path.name
        if "finetune_test" in name:
            match = re.search(r"_fold(\d+)", name)
            if match is None or int(match.group(1)) == fold:
                selected.append(path)
        elif fold_tag in name:
            selected.append(path)
    return selected


def compute_train_stats(wide, train_ids: Iterable, feature_types: dict | None = None) -> dict:
    train_ids = set(train_ids)
    train_values = wide[wide.index.get_level_values("stay_id").isin(train_ids)]
    medians = train_values.median(axis=0, skipna=True).fillna(0.0)
    mads = train_values.sub(medians, axis=1).abs().median(axis=0, skipna=True).fillna(0.0)
    lower = medians - 3.0 * mads
    upper = medians + 3.0 * mads
    zero_mad = mads == 0
    lower.loc[zero_mad] = -np.inf
    upper.loc[zero_mad] = np.inf
    clipped = train_values.clip(lower=lower, upper=upper, axis=1)
    means = clipped.mean(axis=0, skipna=True).fillna(0.0)
    stds = clipped.std(axis=0, skipna=True).replace(0, np.nan).fillna(1.0)
    if feature_types is not None:
        binary_cols = [
            col for col, feature_type in feature_types.items() if feature_type == "procedure"
        ]
        binary_cols = [col for col in binary_cols if col in means.index]
        means.loc[binary_cols] = 0.0
        stds.loc[binary_cols] = 1.0
        lower.loc[binary_cols] = -np.inf
        upper.loc[binary_cols] = np.inf
    return {
        "mean": means.astype("float32"),
        "std": stds.astype("float32"),
        "clip_lower": lower.astype("float32"),
        "clip_upper": upper.astype("float32"),
    }


def compute_static_stats(cohort, train_ids: Iterable) -> dict:
    """Compute train-only age statistics for the DuETT static vector."""
    train_ids = set(train_ids)
    train_rows = cohort[cohort["stay_id"].isin(train_ids)].drop_duplicates("stay_id")
    ages = pd.to_numeric(train_rows["age"], errors="coerce")
    age_mean = float(ages.mean()) if ages.notna().any() else 0.0
    age_std = float(ages.std()) if ages.notna().sum() > 1 else 1.0
    if not np.isfinite(age_std) or age_std == 0:
        age_std = 1.0
    return {"age_mean": age_mean, "age_std": age_std}


def encode_duett_static(cohort_row, stats: dict | None) -> list[float]:
    """Encode age and gender as a stable numeric DuETT static vector."""
    age = pd.to_numeric(pd.Series([cohort_row["age"]]), errors="coerce").iloc[0]
    if pd.isna(age):
        age = stats["age_mean"] if stats is not None else 0.0
    if stats is not None:
        age = (float(age) - stats["age_mean"]) / stats["age_std"]

    gender = str(cohort_row["gender"]).strip().lower()
    is_male = float(gender in {"m", "male", "1", "1.0"})
    is_female = float(gender in {"f", "female", "0", "0.0"})
    is_unknown = float(not (is_male or is_female))
    return [float(age), is_male, is_female, is_unknown]


def apply_normalization(wide, stats: dict | None):
    if stats is None:
        return wide
    clipped = wide.clip(
        lower=stats.get("clip_lower"), upper=stats.get("clip_upper"), axis=1
    )
    return (clipped - stats["mean"]) / stats["std"]


def make_output_prefix(args: argparse.Namespace) -> str:
    if args.prefix:
        return args.prefix
    prefix = f"hourly_{args.window_hours}h_{args.feature_col}_{args.agg}_{args.fill}"
    return f"{prefix}_{args.num_time_bins}bins_duett" if args.duett else prefix


def save_wide_parquet(hourly, cohort, output_path: Path) -> None:
    cohort_cols = [col for col in ["stay_id", "age", "gender"] if col in cohort.columns]
    hourly = hourly.reset_index().merge(
        cohort[cohort_cols].drop_duplicates("stay_id"),
        on="stay_id",
        how="left",
        validate="many_to_one",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(output_path, index=False)
    print(f"Saved: {output_path}")


def save_pickle_dataset(
    hourly,
    mask,
    counts,
    cohort,
    labels: list[str],
    output_path: Path,
    stay_ids: Iterable | None = None,
    time_since=None,
    stats: dict | None = None,
    static_stats: dict | None = None,
    duett: bool = False,
    bin_width_hours: float = 1.0,
) -> None:
    feature_cols = [
        col
        for col in hourly.columns
        if not col.startswith(("mask_", "count_", "delta_"))
    ]
    mask_cols = [f"mask_{col}" for col in feature_cols]
    cohort_index = cohort.drop_duplicates("stay_id").set_index("stay_id")
    if stay_ids is not None:
        stay_ids = list(dict.fromkeys(stay_ids))
        index_mask = hourly.index.get_level_values("stay_id").isin(stay_ids)
        hourly = hourly[index_mask]
        mask = mask[index_mask]
        if time_since is not None:
            time_since = time_since[index_mask]
        counts = counts[index_mask]
    dataset = {
        "_meta": {
            "feature_cols": feature_cols,
            "mask_cols": mask_cols,
            "label_cols": labels,
            "normalized": stats is not None,
            "representation": "hourly_value_mask_time_since",
        }
    }
    if duett:
        dataset["_meta"].update(
            {
                "representation": "duett_value_count_static_time",
                "x_ts_layout": "values_then_observation_counts",
                "x_ts_feature_dim": len(feature_cols) * 2,
                "x_static_names": ["age", "gender_male", "gender_female", "gender_unknown"],
                "times_unit": "fractional_days_from_icu_admission",
                "bin_width_hours": bin_width_hours,
            }
        )
    if stats is not None:
        dataset["_meta"]["normalization"] = {
            "mean": stats["mean"].to_dict(),
            "std": stats["std"].to_dict(),
            "clip_lower": stats["clip_lower"].to_dict(),
            "clip_upper": stats["clip_upper"].to_dict(),
        }
    if static_stats is not None:
        dataset["_meta"]["static_normalization"] = static_stats

    values = hourly[feature_cols]
    masks = mask[mask_cols]
    count_cols = [f"count_{col}" for col in feature_cols]
    observation_counts = counts[count_cols]
    for stay_id, stay_values in values.groupby(level="stay_id", sort=False):
        if stay_id not in cohort_index.index:
            continue
        cohort_row = cohort_index.loc[stay_id]
        stay_masks = masks.loc[stay_id]
        if duett:
            stay_counts = observation_counts.loc[stay_id]
            value_array = stay_values.to_numpy(dtype="float32")
            count_array = stay_counts.to_numpy(dtype="float32")
            x_ts = np.concatenate([value_array, count_array], axis=1)
            x_static = encode_duett_static(cohort_row, static_stats)
            times = (
                (np.arange(len(stay_values), dtype="float32") + 1.0)
                * bin_width_hours
                / 24.0
            ).tolist()
            item = {
                "stay_id": stay_id,
                "x_ts": x_ts.tolist(),
                "x_static": x_static,
                "times": times,
                "x": (x_ts.tolist(), x_static, times),
                "y": [cohort_row[label] for label in labels],
            }
        else:
            item = {
                "stay_id": stay_id,
                "age": cohort_row["age"],
                "gender": cohort_row["gender"],
                "time_index": list(range(len(stay_values))),
                "x": stay_values.to_numpy(dtype="float32").tolist(),
                "mask": stay_masks.to_numpy(dtype="int8").tolist(),
            }
            if time_since is not None:
                item["time_since"] = time_since.loc[stay_id].to_numpy(dtype="float32").tolist()
        for label in labels:
            item[label] = cohort_row[label]
        dataset[stay_id] = item

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {output_path}")


def run(args: argparse.Namespace) -> None:
    load_runtime_dependencies()
    if args.window_hours <= 0:
        raise ValueError("--window-hours must be positive.")
    if args.num_time_bins is None:
        args.num_time_bins = 32 if args.duett else args.window_hours
    if args.num_time_bins <= 0:
        raise ValueError("--num-time-bins must be positive.")
    bin_width_hours = args.window_hours / args.num_time_bins
    sequence, cohort = load_inputs(args)
    stage(f"Loaded sequence ({len(sequence):,} rows) and cohort ({cohort['stay_id'].nunique():,} stays)")
    duett_spec = None
    if args.duett:
        if args.feature_col != "label_idx":
            raise ValueError("--duett requires --feature-col label_idx for MIMIC/eICU alignment.")
        duett_spec = load_duett_feature_spec(args.duett_feature_map)
        sequence = filter_duett_sequence(sequence, args, duett_spec)
        if sequence.empty:
            raise ValueError("No MIMIC events matched the shared DuETT feature vocabulary.")
    labels = args.labels or [col for col in DEFAULT_LABEL_COLS if col in cohort.columns]

    if args.source_aware:
        wide_raw, mask, counts, feature_types = aggregate_hourly_source_aware(sequence, args)
    else:
        wide_raw, mask, counts, feature_types = aggregate_hourly(sequence, args)
    stage("Hourly aggregation complete")
    if duett_spec is not None:
        wide_raw, mask, counts = align_duett_columns(
            wide_raw, mask, counts, duett_spec
        )
        feature_types = {
            item["feature_col"]: item["source"] for item in duett_spec
        }
    wide_raw, mask, counts = expand_hourly_grid(
        wide_raw, mask, counts, cohort, args.num_time_bins
    )
    wide_raw = apply_structural_zeros(wide_raw, feature_types)
    time_since = (
        compute_time_since(mask, bin_width_hours) if args.add_time_since else None
    )
    stage("Grid expansion + time-since computation complete")
    wide = apply_fill(wide_raw, args.fill, args.ffill_limit)

    hourly = pd.concat([wide, mask], axis=1)
    if args.duett:
        hourly = pd.concat([hourly, counts], axis=1)
    if time_since is not None:
        hourly = pd.concat([hourly, time_since], axis=1)
    output_dir = Path(args.output_dir)
    prefix = make_output_prefix(args)

    print(
        "Hourly dataset:",
        f"stays={cohort['stay_id'].nunique()}",
        f"hours={args.window_hours}",
        f"bins={args.num_time_bins}",
        f"bin_width_hours={bin_width_hours:g}",
        f"features={wide.shape[1]}",
        f"rows={hourly.shape[0]}",
    )
    print("Feature groups:", dict(pd.Series(feature_types).value_counts()))

    if args.save_format in {"parquet", "both"}:
        save_wide_parquet(hourly, cohort, output_dir / f"{prefix}.parquet")
    if args.save_format in {"pickle", "both"} and args.split_pkl_dir is None:
        save_pickle_dataset(
            hourly,
            mask,
            counts,
            cohort,
            labels,
            output_dir / f"{prefix}.pkl",
            duett=args.duett,
            bin_width_hours=bin_width_hours,
        )
    if args.split_pkl_dir is not None:
        split_dir = Path(args.split_pkl_dir)
        split_files = restrict_split_files_to_fold(
            find_split_files(split_dir, args.window_hours),
            args.fold,
        )
        if not split_files:
            raise FileNotFoundError(
                f"No finetune split pickle files for {args.window_hours}h"
                f" and fold={args.fold} were found in {split_dir}."
            )

        fold_train_ids = {}
        for split_file in split_files:
            if "finetune_train" in split_file.name and "_fold" in split_file.name:
                fold = split_file.name.split("_fold", 1)[1].split("_", 1)[0].split(".", 1)[0]
                fold_train_ids[fold] = load_split_ids(split_file)

        stats_by_fold = {
            fold: compute_train_stats(wide_raw, ids, feature_types)
            for fold, ids in fold_train_ids.items()
        }
        static_stats_by_fold = {
            fold: compute_static_stats(cohort, ids)
            for fold, ids in fold_train_ids.items()
        }

        for split_file in split_files:
            split_ids = load_split_ids(split_file)
            fold = None
            if "_fold" in split_file.name:
                fold = split_file.name.split("_fold", 1)[1].split("_", 1)[0].split(".", 1)[0]

            if args.normalize and "finetune_test" in split_file.name and stats_by_fold:
                for test_fold, test_stats in sorted(stats_by_fold.items()):
                    split_wide = apply_normalization(wide_raw, test_stats)
                    split_wide = apply_fill(split_wide, args.fill, args.ffill_limit) # 이거 빼야할 것 같은데
                    out_name = split_file.name.replace(
                        ".pkl",
                        f"_fold{test_fold}_{'duett' if args.duett else 'hourly'}.pkl",
                    )
                    save_pickle_dataset(
                        split_wide,
                        mask,
                        counts,
                        cohort,
                        labels,
                        output_dir / out_name,
                        stay_ids=split_ids,
                        time_since=time_since,
                        stats=test_stats,
                        static_stats=static_stats_by_fold[test_fold],
                        duett=args.duett,
                        bin_width_hours=bin_width_hours,
                    )
                continue

            stats = stats_by_fold.get(fold) if args.normalize else None
            split_wide = apply_normalization(wide_raw, stats)
            split_wide = apply_fill(split_wide, args.fill, args.ffill_limit)
            out_name = split_file.name.replace(
                ".pkl", f"_{'duett' if args.duett else 'hourly'}.pkl"
            )
            save_pickle_dataset(
                split_wide,
                mask,
                counts,
                cohort,
                labels,
                output_dir / out_name,
                stay_ids=split_ids,
                time_since=time_since,
                stats=stats,
                static_stats=static_stats_by_fold.get(fold) if args.normalize else None,
                duett=args.duett,
                bin_width_hours=bin_width_hours,
            )


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
