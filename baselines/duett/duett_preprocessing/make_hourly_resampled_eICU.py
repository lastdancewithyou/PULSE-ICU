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
    "mortality_inicu",
    "mortality_48hr",
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
    global np, pd, KFold, train_test_split

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold, train_test_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create hourly-resampled eICU-YAIB datasets.")
    parser.add_argument("--sequence-path", default="datasets/eicu_mapped.parquet")
    parser.add_argument("--timeline-path", default="datasets/eicu_timeline.parquet")
    parser.add_argument("--cohort-path", default="data_statistic/finetune_eicu_YAIB_cohort.parquet")
    parser.add_argument("--raw-cohort-path", default="cohort/eicu_cohort_labeled.parquet")
    parser.add_argument("--yaib-map", default="YAIB_label_map.xlsx")
    parser.add_argument("--units-map", default="units_map.csv")
    parser.add_argument("--split-pkl-dir", default="new_data_preparation_eicu")
    parser.add_argument(
        "--fold",
        type=int,
        choices=[0],
        default=0,
        help="Use fold 0 only (fixed for the DuETT comparison).",
    )
    parser.add_argument(
        "--split-file-glob",
        nargs="*",
        default=None,
        help="Optional pickle glob(s) inside --split-pkl-dir. Use this to reuse an existing final split naming scheme.",
    )
    parser.add_argument("--output-dir", default="hourly_resampled_eicu")
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument(
        "--num-time-bins",
        type=int,
        default=None,
        help="Equal-width bins across the window. Defaults to 32 with --duett.",
    )
    parser.add_argument("--feature-col", default="label_idx", choices=["label_idx", "label", "eicu_label"])
    parser.add_argument("--value-col", default="value")
    parser.add_argument("--source-col", default="source_table_idx")
    parser.add_argument("--chart-agg", default="last", choices=["last", "mean", "median", "min", "max", "first"])
    parser.add_argument("--medication-agg", default="sum", choices=["sum", "mean", "median", "min", "max", "first", "last"])
    parser.add_argument("--fill", default="zero", choices=["none", "zero", "ffill", "ffill_zero"])
    parser.add_argument("--ffill-limit", type=int, default=None)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", action="store_false", dest="normalize")
    parser.add_argument("--add-time-since", action="store_true", default=True)
    parser.add_argument("--no-time-since", action="store_false", dest="add_time_since")
    parser.add_argument("--save-format", choices=["pickle", "parquet", "both"], default="both")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--min-observed-hours", type=int, default=1)
    parser.add_argument("--sample-stays", type=int, default=None)
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--sample-random-state", type=int, default=42)
    parser.add_argument("--prefix", default="eicu_hourly_24h_label_idx_last_zero")
    parser.add_argument(
        "--duett",
        action="store_true",
        help="Save DuETT x=(x_ts, x_static, times) with value/count channels.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--make-splits-if-missing",
        action="store_true",
        default=False,
        help="Create new hourly train/test/CV splits when split-pkl-dir has no eICU split pickles. Do not use for matched irregular/hourly comparisons.",
    )
    parser.add_argument(
        "--auto-map",
        action="store_true",
        default=True,
        help="Build --sequence-path from --timeline-path when the mapped file is missing.",
    )
    parser.add_argument("--no-auto-map", action="store_false", dest="auto_map")
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
    if text in {"1", "1.0", "chartevents", "labevents", "chart"}:
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


def eicu_unit_preprocess(timeline):
    timeline = timeline.copy()
    timeline["valueuom"] = timeline["valueuom"].fillna("no uom")
    timeline.loc[timeline["valueuom"] == "K/mcL", "valueuom"] = "K/uL"
    timeline.loc[timeline["valueuom"] == "mm Hg", "valueuom"] = "mmHg"
    lower = timeline["valueuom"].astype(str).str.lower()
    timeline.loc[lower.isin(["unit(s)", "units", "unit"]), "valueuom"] = "units"
    timeline.loc[lower.isin(["ph units", "[ph]"]), "valueuom"] = "units"
    return timeline


def build_eicu_label_list(label_map) -> list[str]:
    labels = label_map[
        label_map["eicu_label"].notna() & label_map["use_input"].eq("y")
    ]["eicu_label"].tolist()
    exclude = ["verbal", "motor", "eyes", "gcs", "glasgow coma scale"]
    selected = []
    for label in labels:
        if any(keyword in str(label).lower() for keyword in exclude):
            continue
        for sublabel in re.split(r"[,]", str(label)):
            sublabel = sublabel.strip()
            if sublabel and sublabel not in selected:
                selected.append(sublabel)
    return selected


def build_label_map_subset(label_map, eicu_labels):
    required = [
        "eicu_label",
        "label_idx",
        "source_table_idx",
        "ordercategoryname_idx",
        "ordercategorydescription_idx",
        "use_input",
    ]
    require_columns(label_map, required, "YAIB label_map")
    expanded = (
        label_map[label_map["use_input"].eq("y")]
        .assign(eicu_label=lambda df: df["eicu_label"].astype(str).str.split(r"[,]"))
        .explode("eicu_label")
    )
    expanded["eicu_label"] = expanded["eicu_label"].str.strip()
    expanded = expanded[expanded["eicu_label"].isin(eicu_labels)]
    cols = [
        "eicu_label",
        "label_idx",
        "source_table_idx",
        "ordercategoryname_idx",
        "ordercategorydescription_idx",
    ]
    subset = expanded[cols].drop_duplicates().reset_index(drop=True)
    duplicated = subset[subset.duplicated("eicu_label", keep=False)]
    if not duplicated.empty:
        conflicts = duplicated.groupby("eicu_label")[cols[1:]].nunique()
        conflicts = conflicts[(conflicts > 1).any(axis=1)]
        if not conflicts.empty:
            raise ValueError(f"Conflicting YAIB mapping rows: {conflicts.index.tolist()[:20]}")
        subset = subset.drop_duplicates(subset=["eicu_label"], keep="first")
    return subset.reset_index(drop=True)


def build_mapped_sequence(args: argparse.Namespace):
    timeline = pd.read_parquet(args.timeline_path)
    require_columns(timeline, ["stay_id", "label", "value", "intime_offset"], args.timeline_path)
    if "valueuom" not in timeline.columns:
        timeline["valueuom"] = "no uom"

    label_map = pd.read_excel(args.yaib_map, sheet_name="label_map")
    eicu_labels = build_eicu_label_list(label_map)
    timeline = timeline[timeline["label"].isin(eicu_labels)].copy()
    timeline = eicu_unit_preprocess(timeline)
    label_map_subset = build_label_map_subset(label_map, eicu_labels)
    mapped = timeline.merge(
        label_map_subset,
        left_on="label",
        right_on="eicu_label",
        how="left",
        validate="many_to_one",
    )
    numeric_cols = [
        "label_idx",
        "source_table_idx",
        "ordercategoryname_idx",
        "ordercategorydescription_idx",
    ]
    if mapped[numeric_cols].isna().any().any():
        missing = mapped.loc[mapped[numeric_cols].isna().any(axis=1), "label"].drop_duplicates().tolist()
        raise ValueError(f"YAIB mapping is missing for labels: {missing[:20]}")
    mapped[numeric_cols] = mapped[numeric_cols].astype(int)

    unit_map = pd.read_csv(args.units_map).rename(columns={"id": "unit_idx"})
    mapped["unit_idx"] = mapped["valueuom"].map(dict(zip(unit_map["unit"], unit_map["unit_idx"]))).fillna(3).astype(int)
    mapped["intime_offset_hr"] = mapped["intime_offset"].astype(float) * 24.0
    mapped = mapped.sort_values(["stay_id", "intime_offset"]).reset_index(drop=True)

    output = Path(args.sequence_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_parquet(output, index=False)
    print(f"Saved mapped eICU sequence: {output}")
    return mapped


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
            stay_id = item.get("original_stay_id", item.get("stay_id", key))
        else:
            stay_id = key
        if hasattr(stay_id, "item"):
            stay_id = stay_id.item()
        if isinstance(stay_id, str) and stay_id.isdigit():
            stay_id = int(stay_id)
        ids.append(stay_id)
    return ids


def find_split_files(split_dir: Path, window_hours: int) -> list[Path]:
    patterns = getattr(find_split_files, "patterns", None) or [
        f"finetune_train_{window_hours}_fold*_eicu*.pkl",
        f"finetune_val_{window_hours}_fold*_eicu*.pkl",
        f"finetune_test_{window_hours}_eicu*.pkl",
        f"finetune_train_{window_hours}_*_fold*_eicu*.pkl",
        f"finetune_val_{window_hours}_*_fold*_eicu*.pkl",
    ]
    files = []
    for pattern in patterns:
        files.extend(sorted(split_dir.glob(pattern)))
    return sorted(dict.fromkeys(files))


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


def find_split_files_for_args(args: argparse.Namespace) -> list[Path]:
    if args.split_file_glob:
        find_split_files.patterns = args.split_file_glob
        try:
            files = find_split_files(Path(args.split_pkl_dir), args.window_hours)
        finally:
            delattr(find_split_files, "patterns")
    else:
        files = find_split_files(Path(args.split_pkl_dir), args.window_hours)
    return restrict_split_files_to_fold(files, args.fold)


def validate_split_overlap(split_files: list[Path], cohort) -> None:
    cohort_ids = set(cohort["stay_id"].drop_duplicates().tolist())
    bad = []
    for split_file in split_files:
        split_ids = set(load_split_ids(split_file))
        overlap = len(split_ids & cohort_ids)
        if overlap == 0:
            bad.append(split_file.name)
        print(f"Split overlap: {split_file.name} -> {overlap}/{len(split_ids)} stays")
    if bad:
        raise ValueError(
            "Some split files have zero overlap with the eICU cohort. "
            "They are probably from another dataset, such as MIMIC. "
            f"Bad files: {bad[:10]}"
        )


def extract_fold(name: str) -> str | None:
    match = re.search(r"_fold(\d+)", name)
    return match.group(1) if match else None


def is_full_train_split(name: str, window_hours: int) -> bool:
    return re.search(rf"^finetune_train_{window_hours}_fold\d+", name) is not None


def load_inputs(args: argparse.Namespace):
    sequence_path = Path(args.sequence_path)
    if sequence_path.exists():
        sequence = pd.read_parquet(sequence_path)
    elif args.auto_map:
        sequence = build_mapped_sequence(args)
    else:
        raise FileNotFoundError(f"Mapped sequence not found: {sequence_path}")

    cohort_path = Path(args.cohort_path)
    if cohort_path.exists():
        cohort = pd.read_parquet(cohort_path)
    elif Path(args.raw_cohort_path).exists():
        cohort = pd.read_parquet(args.raw_cohort_path)
        print(f"Using raw labeled cohort because --cohort-path was not found: {args.raw_cohort_path}")
    else:
        raise FileNotFoundError(f"Cohort not found: {args.cohort_path}")
    stage(f"Loaded sequence ({len(sequence):,} rows) and cohort ({len(cohort):,} stays)")

    require_columns(sequence, ["stay_id", "intime_offset", args.feature_col, args.value_col, args.source_col], args.sequence_path)
    require_columns(cohort, ["stay_id", "age", "gender"], args.cohort_path)
    missing_labels = [label for label in DEFAULT_LABEL_COLS if label not in cohort.columns]
    if missing_labels:
        raise ValueError(
            "cohort is missing hourly labels. Run `python3 preprocessing_eICU.py all` first. "
            f"Missing columns: {missing_labels[:20]}"
        )

    sequence = sequence.copy()
    if "intime_offset_hr" not in sequence.columns:
        sequence["intime_offset_hr"] = sequence["intime_offset"].astype(float) * 24.0
    sequence[args.value_col] = pd.to_numeric(sequence[args.value_col], errors="coerce")

    source = sequence[args.source_col].map(normalize_source_name)
    procedure_mask = source.eq("procedure")
    sequence = sequence.dropna(subset=["intime_offset_hr", args.feature_col])
    sequence = sequence[procedure_mask | sequence[args.value_col].notna()].copy()
    sequence = sequence[(sequence["intime_offset_hr"] >= 0) & (sequence["intime_offset_hr"] < args.window_hours)].copy()
    bin_width_hours = args.window_hours / args.num_time_bins
    sequence["hour"] = np.floor(
        sequence["intime_offset_hr"] / bin_width_hours
    ).astype(int)
    sequence["hour"] = sequence["hour"].clip(0, args.num_time_bins - 1)

    observed_hours = sequence.groupby("stay_id")["hour"].nunique()
    keep_stays = observed_hours[observed_hours >= args.min_observed_hours].index
    sequence = sequence[sequence["stay_id"].isin(keep_stays)].copy()
    cohort = cohort[cohort["stay_id"].isin(sequence["stay_id"].unique())].copy()

    if args.sample_stays is not None or args.sample_frac is not None:
        stay_ids = pd.Series(cohort["stay_id"].drop_duplicates().to_numpy())
        split_files = find_split_files_for_args(args)
        split_ids = []
        for split_file in split_files:
            split_ids.extend(load_split_ids(split_file))
        if split_ids:
            stay_ids = stay_ids[stay_ids.isin(set(split_ids))]
        if stay_ids.empty:
            raise ValueError("No candidate stay_id values remain for sampling.")
        if args.sample_frac is not None:
            if not 0 < args.sample_frac <= 1:
                raise ValueError("--sample-frac must be in the range (0, 1].")
            n = max(1, int(len(stay_ids) * args.sample_frac))
        else:
            n = min(args.sample_stays, len(stay_ids))
        sampled = stay_ids.sample(n=n, random_state=args.sample_random_state).to_numpy()
        sequence = sequence[sequence["stay_id"].isin(sampled)].copy()
        cohort = cohort[cohort["stay_id"].isin(sampled)].copy()
        print(f"Smoke-test sample: {n} stay_id values")

    return sequence, cohort


def add_interval_offsets(df):
    data = df.copy()
    if {"starttime", "endtime", "intime"}.issubset(data.columns):
        data["start_hr"] = (pd.to_datetime(data["starttime"]) - pd.to_datetime(data["intime"])).dt.total_seconds() / 3600.0
        data["end_hr"] = (pd.to_datetime(data["endtime"]) - pd.to_datetime(data["intime"])).dt.total_seconds() / 3600.0
    else:
        data["start_hr"] = data["intime_offset_hr"]
        data["end_hr"] = data["intime_offset_hr"]
    data["start_hr"] = data["start_hr"].fillna(data["intime_offset_hr"])
    data["end_hr"] = data["end_hr"].fillna(data["start_hr"])
    data["end_hr"] = data[["start_hr", "end_hr"]].max(axis=1)
    return data


def expand_medication_intervals(df, args: argparse.Namespace):
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
            hours = [int(np.floor(start / bin_width_hours))]
            duration = 1.0
        else:
            hours = range(
                int(np.floor(start / bin_width_hours)),
                int(np.ceil(end / bin_width_hours)),
            )
            duration = end - start
        for hour in hours:
            if hour < 0 or hour >= args.num_time_bins:
                continue
            bin_start = hour * bin_width_hours
            bin_end = (hour + 1) * bin_width_hours
            overlap = 1.0 if end <= start else max(
                0.0, min(end, bin_end) - max(start, bin_start)
            )
            if overlap <= 0:
                continue
            record = row._asdict()
            record["hour"] = hour
            record[args.value_col] = float(value) * (overlap / duration)
            rows.append(record)
    return pd.DataFrame(rows)


def expand_procedure_intervals(df, args: argparse.Namespace):
    rows = []
    data = add_interval_offsets(df)
    bin_width_hours = args.window_hours / args.num_time_bins
    for row in data.itertuples(index=False):
        start = max(0.0, float(row.start_hr))
        end = min(float(args.window_hours), float(row.end_hr))
        hours = (
            [int(np.floor(start / bin_width_hours))]
            if end <= start
            else range(
                int(np.floor(start / bin_width_hours)),
                int(np.ceil(end / bin_width_hours)),
            )
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


def aggregate_one_source(df, args: argparse.Namespace, source_name: str, agg: str):
    data = df.copy()
    data["_hourly_feature"] = source_name + "__" + data[args.feature_col].map(feature_name)
    group_cols = ["stay_id", "hour", "_hourly_feature"]
    values = data.groupby(group_cols, sort=False)[args.value_col].agg(agg).reset_index()
    counts = data.groupby(group_cols, sort=False).size().reset_index(name="_observation_count")
    return values, counts


def aggregate_hourly(sequence, args: argparse.Namespace):
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
            proc_counts = proc.groupby(
                ["stay_id", "hour", "_hourly_feature"], sort=False
            ).size().reset_index(name="_observation_count")
            proc_grouped = proc_counts.rename(
                columns={"_observation_count": args.value_col}
            ).copy()
            proc_grouped[args.value_col] = (proc_grouped[args.value_col] > 0).astype("int8")
            parts.append(proc_grouped)
            count_parts.append(proc_counts)
            feature_types.update({col: "procedure" for col in proc_grouped["_hourly_feature"].unique()})
    if not other.empty:
        other_grouped, other_counts = aggregate_one_source(other, args, "other", "last")
        parts.append(other_grouped)
        count_parts.append(other_counts)
        feature_types.update({col: "other" for col in other_grouped["_hourly_feature"].unique()})
    if not parts:
        raise ValueError("No events remained after hourly aggregation.")

    grouped = pd.concat(parts, ignore_index=True)
    wide = grouped.pivot_table(index=["stay_id", "hour"], columns="_hourly_feature", values=args.value_col, aggfunc="first")
    wide = wide.sort_index(axis=1)
    mask = wide.notna().astype("int8").rename(columns={col: f"mask_{col}" for col in wide.columns})
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


def expand_grid(wide, mask, counts, cohort, num_time_bins: int):
    full_index = pd.MultiIndex.from_product(
        [cohort["stay_id"].drop_duplicates().sort_values(), range(num_time_bins)],
        names=["stay_id", "hour"],
    )
    return (
        wide.reindex(full_index),
        mask.reindex(full_index).fillna(0).astype("int8"),
        counts.reindex(full_index).fillna(0).astype("int32"),
    )


def apply_structural_zeros(wide, feature_types: dict):
    zero_cols = [col for col, typ in feature_types.items() if typ in {"medication", "procedure"} and col in wide.columns]
    if zero_cols:
        wide = wide.copy()
        wide[zero_cols] = wide[zero_cols].fillna(0)
    return wide


def apply_fill(wide, fill: str, ffill_limit: int | None):
    if fill == "none":
        return wide
    if fill in {"ffill", "ffill_zero"}:
        wide = wide.groupby(level="stay_id").ffill(limit=ffill_limit)
    if fill in {"zero", "ffill_zero"}:
        wide = wide.fillna(0)
    return wide


def compute_time_since(mask, bin_width_hours: float = 1.0):
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
    return result.rename(columns={col: col.replace("mask_", "delta_") for col in result.columns})


def compute_train_stats(wide, train_ids: Iterable, feature_types: dict) -> dict:
    train_values = wide[wide.index.get_level_values("stay_id").isin(set(train_ids))]
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
    procedure_cols = [col for col, typ in feature_types.items() if typ == "procedure" and col in means.index]
    means.loc[procedure_cols] = 0.0
    stds.loc[procedure_cols] = 1.0
    lower.loc[procedure_cols] = -np.inf
    upper.loc[procedure_cols] = np.inf
    return {
        "mean": means.astype("float32"),
        "std": stds.astype("float32"),
        "clip_lower": lower.astype("float32"),
        "clip_upper": upper.astype("float32"),
    }


def compute_static_stats(cohort, train_ids: Iterable) -> dict:
    train_rows = cohort[
        cohort["stay_id"].isin(set(train_ids))
    ].drop_duplicates("stay_id")
    ages = pd.to_numeric(train_rows["age"], errors="coerce")
    age_mean = float(ages.mean()) if ages.notna().any() else 0.0
    age_std = float(ages.std()) if ages.notna().sum() > 1 else 1.0
    if not np.isfinite(age_std) or age_std == 0:
        age_std = 1.0
    return {"age_mean": age_mean, "age_std": age_std}


def encode_duett_static(row, stats: dict | None) -> list[float]:
    age = pd.to_numeric(pd.Series([row["age"]]), errors="coerce").iloc[0]
    if pd.isna(age):
        age = stats["age_mean"] if stats is not None else 0.0
    if stats is not None:
        age = (float(age) - stats["age_mean"]) / stats["age_std"]
    gender = str(row["gender"]).strip().lower()
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


def save_wide_parquet(hourly, cohort, output_path: Path) -> None:
    hourly = hourly.reset_index().merge(
        cohort[["stay_id", "age", "gender"]].drop_duplicates("stay_id"),
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
    stay_ids=None,
    time_since=None,
    stats=None,
    static_stats=None,
    duett: bool = False,
    bin_width_hours: float = 1.0,
) -> None:
    feature_cols = [
        col for col in hourly.columns
        if not col.startswith(("mask_", "count_", "delta_"))
    ]
    mask_cols = [f"mask_{col}" for col in feature_cols]
    if stay_ids is not None:
        stay_ids = list(dict.fromkeys(stay_ids))
        index_mask = hourly.index.get_level_values("stay_id").isin(stay_ids)
        hourly = hourly[index_mask]
        mask = mask[index_mask]
        counts = counts[index_mask]
        if time_since is not None:
            time_since = time_since[index_mask]
    cohort_index = cohort.drop_duplicates("stay_id").set_index("stay_id")
    dataset = {
        "_meta": {
            "feature_cols": feature_cols,
            "mask_cols": mask_cols,
            "label_cols": labels,
            "normalized": stats is not None,
            "representation": "eicu_yaib_hourly_value_mask_time_since",
        }
    }
    if duett:
        dataset["_meta"].update(
            {
                "representation": "eicu_duett_value_count_static_time",
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
        row = cohort_index.loc[stay_id]
        if duett:
            value_array = stay_values.to_numpy(dtype="float32")
            count_array = observation_counts.loc[stay_id].to_numpy(dtype="float32")
            x_ts = np.concatenate([value_array, count_array], axis=1).tolist()
            x_static = encode_duett_static(row, static_stats)
            times = (
                (np.arange(len(stay_values), dtype="float32") + 1.0)
                * bin_width_hours
                / 24.0
            ).tolist()
            item = {
                "stay_id": stay_id,
                "x_ts": x_ts,
                "x_static": x_static,
                "times": times,
                "x": (x_ts, x_static, times),
                "y": [row[label] for label in labels],
            }
        else:
            item = {
                "stay_id": stay_id,
                "age": row["age"],
                "gender": row["gender"],
                "time_index": list(range(len(stay_values))),
                "x": stay_values.to_numpy(dtype="float32").tolist(),
                "mask": masks.loc[stay_id].to_numpy(dtype="int8").tolist(),
            }
            if time_since is not None:
                item["time_since"] = time_since.loc[stay_id].to_numpy(dtype="float32").tolist()
        for label in labels:
            item[label] = row[label]
        dataset[stay_id] = item

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {output_path}")


def save_generated_hourly_splits(
    wide_raw, mask, counts, cohort, labels, output_dir: Path,
    args, time_since, feature_types, bin_width_hours
) -> None:
    stay_ids = pd.Series(cohort["stay_id"].drop_duplicates().to_numpy())
    if len(stay_ids) < 2:
        raise ValueError("Need at least 2 stays to create hourly train/test splits.")

    train_ids, test_ids = train_test_split(
        stay_ids.to_numpy(),
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_ids = np.array(train_ids)
    test_ids = np.array(test_ids)
    if len(train_ids) < 2:
        raise ValueError("Need at least 2 train stays to create hourly CV folds.")

    n_splits = min(args.n_folds, len(train_ids))
    if n_splits != args.n_folds:
        print(f"Using n_folds={n_splits} because only {len(train_ids)} training stays are available.")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
    print(f"Generated hourly splits: train={len(train_ids)}, test={len(test_ids)}, folds={n_splits}")

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train_ids)):
        if args.fold is not None and fold_idx != args.fold:
            continue
        fold_train_ids = train_ids[tr_idx]
        fold_val_ids = train_ids[val_idx]
        stats = compute_train_stats(wide_raw, fold_train_ids, feature_types) if args.normalize else None
        static_stats = compute_static_stats(cohort, fold_train_ids) if args.normalize else None
        split_wide = apply_fill(apply_normalization(wide_raw, stats), args.fill, args.ffill_limit)
        suffix = "duett" if args.duett else "hourly"
        save_pickle_dataset(
            split_wide,
            mask,
            counts,
            cohort,
            labels,
            output_dir / f"finetune_train_{args.window_hours}_fold{fold_idx}_eicu_{suffix}.pkl",
            fold_train_ids,
            time_since,
            stats,
            static_stats,
            args.duett,
            bin_width_hours,
        )
        save_pickle_dataset(
            split_wide,
            mask,
            counts,
            cohort,
            labels,
            output_dir / f"finetune_val_{args.window_hours}_fold{fold_idx}_eicu_{suffix}.pkl",
            fold_val_ids,
            time_since,
            stats,
            static_stats,
            args.duett,
            bin_width_hours,
        )
        save_pickle_dataset(
            split_wide,
            mask,
            counts,
            cohort,
            labels,
            output_dir / f"finetune_test_{args.window_hours}_eicu_fold{fold_idx}_{suffix}.pkl",
            test_ids,
            time_since,
            stats,
            static_stats,
            args.duett,
            bin_width_hours,
        )


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
    duett_spec = None
    if args.duett:
        if args.feature_col != "label_idx":
            raise ValueError("--duett requires --feature-col label_idx for MIMIC/eICU alignment.")
        duett_spec = load_duett_feature_spec(args.yaib_map)
        sequence = filter_duett_sequence(sequence, args, duett_spec)
        if sequence.empty:
            raise ValueError("No eICU events matched the shared DuETT feature vocabulary.")
    labels = args.labels or [col for col in DEFAULT_LABEL_COLS if col in cohort.columns]
    wide_raw, mask, counts, feature_types = aggregate_hourly(sequence, args)
    stage("Hourly aggregation complete")
    if duett_spec is not None:
        wide_raw, mask, counts = align_duett_columns(
            wide_raw, mask, counts, duett_spec
        )
        feature_types = {
            item["feature_col"]: item["source"] for item in duett_spec
        }
    wide_raw, mask, counts = expand_grid(
        wide_raw, mask, counts, cohort, args.num_time_bins
    )
    wide_raw = apply_structural_zeros(wide_raw, feature_types)
    time_since = compute_time_since(mask, bin_width_hours) if args.add_time_since else None
    stage("Grid expansion + time-since computation complete")
    wide = apply_fill(wide_raw, args.fill, args.ffill_limit)
    hourly = pd.concat([wide, mask], axis=1)
    if args.duett:
        hourly = pd.concat([hourly, counts], axis=1)
    if time_since is not None:
        hourly = pd.concat([hourly, time_since], axis=1)

    output_dir = Path(args.output_dir)
    print(
        "eICU hourly dataset:",
        f"stays={cohort['stay_id'].nunique()}",
        f"hours={args.window_hours}",
        f"bins={args.num_time_bins}",
        f"bin_width_hours={bin_width_hours:g}",
        f"features={wide.shape[1]}",
        f"rows={hourly.shape[0]}",
    )
    print("Feature groups:", dict(pd.Series(feature_types).value_counts()))

    if args.save_format in {"parquet", "both"}:
        prefix = f"{args.prefix}_{args.num_time_bins}bins_duett" if args.duett else args.prefix
        save_wide_parquet(hourly, cohort, output_dir / f"{prefix}.parquet")

    split_files = find_split_files_for_args(args)
    if not split_files:
        if args.save_format in {"pickle", "both"}:
            if args.make_splits_if_missing:
                save_generated_hourly_splits(
                    wide_raw, mask, counts, cohort, labels, output_dir,
                    args, time_since, feature_types, bin_width_hours
                )
            else:
                raise FileNotFoundError(
                    "No eICU split pickle files found. To keep the split identical to irregular eICU, "
                    f"run `python3 make_input_eICU.py --output-dir {args.split_pkl_dir}` first, then rerun this script. "
                    "Use --make-splits-if-missing only when intentionally creating a new hourly-only split."
                )
        return
    validate_split_overlap(split_files, cohort)

    fold_train_ids = {}
    for split_file in split_files:
        if is_full_train_split(split_file.name, args.window_hours):
            fold = extract_fold(split_file.name)
            fold_train_ids[fold] = load_split_ids(split_file)
    if args.normalize and not fold_train_ids:
        raise ValueError(
            "Could not find full train fold files for normalization stats. Expected names like "
            f"`finetune_train_{args.window_hours}_fold0*.pkl` in {args.split_pkl_dir}."
        )
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
        fold = extract_fold(split_file.name)

        if args.normalize and "finetune_test" in split_file.name and stats_by_fold:
            for test_fold, test_stats in sorted(stats_by_fold.items()):
                split_wide = apply_fill(apply_normalization(wide_raw, test_stats), args.fill, args.ffill_limit)
                suffix = "duett" if args.duett else "hourly"
                out_name = split_file.name.replace(".pkl", f"_fold{test_fold}_{suffix}.pkl")
                save_pickle_dataset(
                    split_wide, mask, counts, cohort, labels, output_dir / out_name,
                    split_ids, time_since, test_stats,
                    static_stats_by_fold[test_fold], args.duett, bin_width_hours
                )
            continue

        stats = stats_by_fold.get(fold) if args.normalize else None
        split_wide = apply_fill(apply_normalization(wide_raw, stats), args.fill, args.ffill_limit)
        suffix = "duett" if args.duett else "hourly"
        out_name = split_file.name.replace(".pkl", f"_{suffix}.pkl")
        save_pickle_dataset(
            split_wide, mask, counts, cohort, labels, output_dir / out_name,
            split_ids, time_since, stats,
            static_stats_by_fold.get(fold) if args.normalize else None,
            args.duett, bin_width_hours
        )


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
