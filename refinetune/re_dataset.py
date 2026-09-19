# Pure python
from pathlib import Path
from typing import Optional
import math
import os
import pickle
import tempfile

# PyTorch
import torch
from torch.utils.data import Dataset

# ETC
import pandas as pd


TASK_TO_INDEX = {
    "pretrain": 0,
    "mortality_30days": 1,
    "mortality_inhospital": 2,
    "mortality_icu": 3,
    "mortality48hr": 4,
    "los_3days": 5,
    "los_7days": 6,
    "readmission_30": 7,
    "transfusion_12hr": 8,
    "vasopressor_need_12hr": 9,
    "ventilation_need_12hr": 10,
    "shock_8hr": 11,
    "sofa_centralnervous_24hr": 12,
    "sofa_cardiovascular_24hr": 13,
    "sofa_respiratory_24hr": 14,
    "sofa_coagulation_24hr": 15,
    "sofa_liver_24hr": 16,
    "sofa_renal_24hr": 17,
    "phenotype": 18,
    "multitask": 19,
}


PHENOTYPING_COLS = [
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


VP_VALUE_TARGET_IDS = (7, 11, 12, 33, 41, 342, 142, 161, 2208, 2209, 37)
DEFAULT_VP_STATS_FILENAME = "vp_value_normalization_stats.pkl"


def build_vp_value_stats(
    pretrain_train_path,
    output_path=None,
    target_ids=VP_VALUE_TARGET_IDS,
    eps=1e-6,
):
    """
    Compute variable-specific mean/std for VALUE PREDICTION targets only.

    IMPORTANT
    ---------
    - Statistics are estimated ONLY from pretrain_train.pkl.
    - Raw value_seq is NOT changed.
    - These statistics are used only to standardize VP regression targets:
          z = (x - mean_j) / std_j
    - Value embedding still receives the original raw value.
    """
    pretrain_train_path = Path(pretrain_train_path)
    if output_path is None:
        output_path = pretrain_train_path.parent / DEFAULT_VP_STATS_FILENAME
    output_path = Path(output_path)

    if not pretrain_train_path.exists():
        raise FileNotFoundError(
            f"Pretraining train file not found: {pretrain_train_path}"
        )

    data = pd.read_pickle(pretrain_train_path)
    if not isinstance(data, dict):
        raise TypeError(
            f"{pretrain_train_path} must contain a dict, "
            f"got {type(data).__name__}."
        )

    target_set = {int(x) for x in target_ids}

    # event_id -> [count, sum, sumsq]
    accum = {
        event_id: [0, 0.0, 0.0]
        for event_id in target_set
    }

    for sequence_key, item in data.items():
        ehr_seq = item.get("ehr_seq")
        value_seq = item.get("value_seq")

        if ehr_seq is None or value_seq is None:
            raise KeyError(
                f"Sequence {sequence_key!r} is missing ehr_seq/value_seq."
            )

        if len(ehr_seq) != len(value_seq):
            raise ValueError(
                f"Sequence {sequence_key!r}: "
                f"len(ehr_seq)={len(ehr_seq)} != len(value_seq)={len(value_seq)}"
            )

        for event_id, value in zip(ehr_seq, value_seq):
            if pd.isna(event_id) or pd.isna(value):
                continue

            event_id = int(event_id)
            if event_id not in target_set:
                continue

            value = float(value)
            if not math.isfinite(value):
                continue

            accum[event_id][0] += 1
            accum[event_id][1] += value
            accum[event_id][2] += value * value

    stats = {}
    for event_id in sorted(target_set):
        count, value_sum, value_sumsq = accum[event_id]
        if count == 0:
            continue

        mean = value_sum / count
        variance = max(value_sumsq / count - mean * mean, 0.0)
        std = math.sqrt(variance)

        if count < 2 or (not math.isfinite(std)) or std < eps:
            std = 1.0

        stats[event_id] = {
            "count": int(count),
            "mean": float(mean),
            "std": float(std),
        }

    if not stats:
        raise ValueError(
            "No finite observations were found for any VP target variable."
        )

    payload = {
        "normalization": "vp_target_only_event_wise_zscore",
        "source": str(pretrain_train_path),
        "target_ids": list(map(int, target_ids)),
        "eps": float(eps),
        "stats": stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(output_path.parent),
        prefix=output_path.name + ".tmp.",
        delete=False,
    ) as tmp:
        pickle.dump(payload, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path = tmp.name

    os.replace(tmp_path, output_path)

    print(
        f"[VP stats] saved TRAIN-only statistics for "
        f"{len(stats)}/{len(target_set)} VP variables -> {output_path}"
    )
    return payload


def load_vp_value_stats(stats_path):
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"VP normalization stats not found: {stats_path}")

    with open(stats_path, "rb") as f:
        payload = pickle.load(f)

    if payload.get("normalization") != "vp_target_only_event_wise_zscore":
        raise ValueError(
            f"Unexpected VP stats format in {stats_path}: "
            f"{payload.get('normalization')!r}"
        )

    return payload


class EHR_Longformer_Dataset(Dataset):
    """
    PULSE-ICU dataset loader compatible with the newly generated split files.

    Expected input features in each pickle item
    -------------------------------------------
    - ehr_seq
    - value_seq
    - unit_seq
    - offset_seq
    - token_type_seq
    - age
    - gender

    Intentionally NOT used
    ----------------------
    - position_seq
    - ordercategoryname_seq
    - ordercategorydescription_seq

    New internal MIMIC split naming
    -------------------------------
    Pretraining:
      pretrain_train.pkl
      pretrain_test.pkl   # used as pretraining validation/checkpoint-selection set

    Finetuning:
      finetune_train_24_fold{fold}.pkl
      finetune_train_24_{ratio}_fold{fold}.pkl
      finetune_val_24_fold{fold}.pkl
      finetune_test_24.pkl

    The finetuning validation/test sets are fixed regardless of training ratio.

    Value handling
    --------------
    - Value embedding always receives RAW value_seq.
    - Only VALUE-PREDICTION regression targets are standardized.
    - VP mean/std are computed from pretrain_train.pkl only.
    - pretrain_test.pkl reuses those PT-train statistics.
      (Despite the filename, it is used as the pretraining validation set.)
    - Finetuning does not normalize values and does not use VP statistics.
    """

    def __init__(
        self,
        data_path: Path,
        split: str,
        tokenizer,
        itemid2idx,
        unit2idx,
        vocab_size,
        block_size=17,
        max_length=4093,
        use_itemid=True,
        mask_token=1,
        mode="pretrain",
        mask_mode="mlm",
        mask_ratio=(0.3, 0.15, 0.3),
        value_mask_ratio=0.15,
        task=None,
        seed=None,
        ablation=None,
        window=None,
        index=0,
        no_gap=False,
        ratio=None,
        selected_data=None,
        locate=None,
        vp_stats_path=None,
        vp_stats_eps=1e-6,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.itemid2idx = itemid2idx
        self.unit2idx = unit2idx
        self.vocab_size = int(vocab_size)
        self.block_size = block_size
        self.max_length = int(max_length)
        self.mask_ratio = tuple(mask_ratio)
        self.value_mask_ratio = float(value_mask_ratio)
        if not 0.0 <= self.value_mask_ratio <= 1.0:
            raise ValueError(
                f"value_mask_ratio must be between 0 and 1, got {self.value_mask_ratio}"
            )
        self.use_itemid = use_itemid
        self.mask_token = mask_token
        self.mode = mode
        self.mask_mode = mask_mode
        self.task = task
        self.ablation = ablation
        self.window = window
        self.no_gap = no_gap
        self.selected_data = selected_data
        self.locate = locate
        self.fold_index = int(index)
        self.ratio = ratio

        # VP target normalization settings.
        # These are used ONLY during pretraining. Finetuning keeps raw values
        # and never loads/uses VP normalization statistics.
        self.vp_stats_eps = float(vp_stats_eps)
        self.vp_stats_path = (
            Path(vp_stats_path)
            if vp_stats_path is not None
            else self.data_path / DEFAULT_VP_STATS_FILENAME
        )
        self.vp_stats = {}

        # Current PULSE-ICU pipeline uses indexed event IDs (ehr_seq).
        # Keeping the argument preserves compatibility with old constructor calls,
        # but the legacy free-text/tokenizer branch is intentionally removed.
        if not self.use_itemid:
            raise ValueError(
                "This PULSE-ICU compatible Dataset expects use_itemid=True "
                "and reads indexed events from data['ehr_seq']."
            )

        if self.mode not in {"pretrain", "finetune"}:
            raise ValueError("mode must be either 'pretrain' or 'finetune'.")

        if self.mode == "pretrain":
            self.cls_token_id = TASK_TO_INDEX["pretrain"]
        else:
            if self.task not in TASK_TO_INDEX:
                raise ValueError(
                    f"Unknown finetuning task={self.task!r}. "
                    f"Available tasks: {sorted(TASK_TO_INDEX)}"
                )
            self.cls_token_id = TASK_TO_INDEX[self.task]

        pickle_name = self._build_pickle_name(split)
        pickle_path = self.data_path / pickle_name

        if not pickle_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {pickle_path}\n"
                f"mode={mode}, split={split}, window={window}, "
                f"fold={index}, ratio={ratio}, selected_data={selected_data}, locate={locate}"
            )

        print(
            f"[EHR_Longformer_Dataset] loading={pickle_path} | "
            f"mode={mode}, split={split}, window={window}, "
            f"fold={index}, ratio={ratio}"
        )
        self.df = pd.read_pickle(pickle_path)

        if not isinstance(self.df, dict):
            raise TypeError(
                f"{pickle_path} must contain a dict keyed by sequence/stay ID, "
                f"but got {type(self.df).__name__}."
            )

        self.keys = list(self.df.keys())

        # Optional deterministic behavior can still be controlled by the caller's
        # global seed. We retain seed only for constructor compatibility.
        self.seed = seed

        self._validate_dataset_schema()

        if self.mode == "pretrain":
            self._initialize_vp_stats()

    # ------------------------------------------------------------------
    # File naming
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_split(split: str) -> str:
        split = split.lower()
        if split in {"val", "valid", "validation"}:
            return "valid"
        if split in {"train", "test"}:
            return split
        raise ValueError(
            f"Unknown split={split!r}. Use train, valid/val, or test."
        )

    @staticmethod
    def _ratio_to_filename_value(ratio) -> Optional[int]:
        if ratio is None:
            return None

        r = float(ratio)

        # Support ratio=0.1 as 10%, while ratio=1 means 1%.
        if 0 < r < 1:
            r *= 100.0

        if r <= 0 or r > 100:
            raise ValueError("ratio must be in (0, 100] or a fraction in (0, 1).")

        rounded = round(r)
        if abs(r - rounded) > 1e-8:
            raise ValueError(
                f"ratio={ratio} does not map to an integer percentage filename."
            )
        return int(rounded)

    def _optional_suffix(self) -> str:
        parts = []
        if self.selected_data:
            parts.append(str(self.selected_data))
        if self.locate:
            parts.append(str(self.locate))
        return "" if not parts else "_" + "_".join(parts)

    def _build_pickle_name(self, split: str) -> str:
        split = self._normalize_split(split)

        if self.mode == "pretrain":
            if split == "train":
                return "pretrain_train.pkl"

            # In the current project naming convention, pretrain_test.pkl is
            # used as the PRETRAINING VALIDATION / checkpoint-selection set.
            # Accept both test and valid aliases to avoid accidental mismatch.
            if split in {"valid", "test"}:
                return "pretrain_test.pkl"

            raise ValueError(
                "Pretraining supports split='train' and split='test' "
                "(or 'valid' as an alias for the same pretrain_test.pkl file)."
            )

        if self.window is None:
            raise ValueError(
                "window must be provided for finetuning, e.g. window=24."
            )

        suffix = self._optional_suffix()

        if split == "train":
            ratio_pct = self._ratio_to_filename_value(self.ratio)
            if ratio_pct is None or ratio_pct == 100:
                return (
                    f"finetune_train_{self.window}_fold{self.fold_index}"
                    f"{suffix}.pkl"
                )
            return (
                f"finetune_train_{self.window}_{ratio_pct}_fold{self.fold_index}"
                f"{suffix}.pkl"
            )

        if split == "valid":
            # Validation is deliberately shared across 1/10/30/50/100% training.
            return (
                f"finetune_val_{self.window}_fold{self.fold_index}"
                f"{suffix}.pkl"
            )

        # Final test is fixed across all folds and ratios.
        return f"finetune_test_{self.window}{suffix}.pkl"

    # ------------------------------------------------------------------
    # Schema / safety checks
    # ------------------------------------------------------------------
    def _validate_dataset_schema(self):
        if len(self.keys) == 0:
            raise ValueError("Loaded dataset is empty.")

        # Checking several examples catches preprocessing mismatches early without
        # scanning the entire potentially large dictionary.
        for key in self.keys[: min(20, len(self.keys))]:
            data = self.df[key]
            required = {
                "stay_id",
                "ehr_seq",
                "value_seq",
                "unit_seq",
                "offset_seq",
                "token_type_seq",
                "age",
                "gender",
            }
            missing = sorted(required - set(data.keys()))
            if missing:
                raise KeyError(
                    f"Dataset item key={key!r} is missing required fields: {missing}"
                )

            lengths = {
                "ehr_seq": len(data["ehr_seq"]),
                "value_seq": len(data["value_seq"]),
                "unit_seq": len(data["unit_seq"]),
                "offset_seq": len(data["offset_seq"]),
                "token_type_seq": len(data["token_type_seq"]),
            }
            if len(set(lengths.values())) != 1:
                raise ValueError(
                    f"Sequence lengths do not match for key={key!r}: {lengths}"
                )

            seq_len = lengths["ehr_seq"]
            if seq_len > self.max_length:
                raise ValueError(
                    f"key={key!r} has {seq_len} events > max_length={self.max_length}. "
                    "The preprocessing/chunking step should guarantee <= max_length."
                )

    # ------------------------------------------------------------------
    # Value-prediction target normalization
    # ------------------------------------------------------------------
    def _initialize_vp_stats(self):
        """
        Load VP target mean/std.

        If the stats file does not exist, it is created from pretrain_train.pkl
        ONLY. pretrain_test.pkl therefore reuses PT-train statistics.

        Finetuning never calls this method.
        """
        if not self.vp_stats_path.exists():
            train_path = self.data_path / "pretrain_train.pkl"
            if not train_path.exists():
                raise FileNotFoundError(
                    f"VP stats file does not exist at {self.vp_stats_path}, "
                    f"and pretrain_train.pkl is not available at {train_path}."
                )

            build_vp_value_stats(
                pretrain_train_path=train_path,
                output_path=self.vp_stats_path,
                eps=self.vp_stats_eps,
            )

        payload = load_vp_value_stats(self.vp_stats_path)
        self.vp_stats = {
            int(event_id): {
                "mean": float(stat["mean"]),
                "std": max(float(stat["std"]), self.vp_stats_eps),
                "count": int(stat["count"]),
            }
            for event_id, stat in payload["stats"].items()
        }

        print(
            f"[VP stats] loaded {len(self.vp_stats)} variable statistics "
            f"from {self.vp_stats_path}"
        )

    def _standardize_vp_targets(self, event_id, raw_values):
        """
        Standardize raw regression targets for one VP variable.

        This function DOES NOT change model input values.
        """
        event_id = int(event_id)
        if event_id not in self.vp_stats:
            return None

        stat = self.vp_stats[event_id]
        return (raw_values - stat["mean"]) / stat["std"]

    # ------------------------------------------------------------------
    # MLM + value prediction masking
    # ------------------------------------------------------------------
    def mask_tokens(
        self,
        tokenized_token,
        tokenized_units,
        tokenized_values,
        tokenized_offsets,
        tokenized_token_type,
        attention_mask,
        mask_ratio,
        value_mask_ratio,
        valid_value_mask=None,
        mask_token=-150,
        mask_label_token=4,
        ignore_token=-100,
    ):
        # Preserve the ORIGINAL event IDs and RAW values for VP target creation.
        # The model input value tensor remains raw except at masked positions.
        original_token = tokenized_token.clone()
        original_values = tokenized_values.clone()

        tokenized_token = tokenized_token.clone()
        tokenized_units = tokenized_units.clone()
        tokenized_values = tokenized_values.clone()
        tokenized_offsets = tokenized_offsets.clone()
        tokenized_token_type = tokenized_token_type.clone()

        labels = tokenized_token.clone()
        mask_labels = torch.full_like(tokenized_token, ignore_token)

        local_attention_mask = attention_mask[: tokenized_token_type.size(0)]

        if len(mask_ratio) != 3:
            raise ValueError(
                f"mask_ratio must have 3 values for event types 0/1/2, got {mask_ratio}"
            )

        for idx, event_type in enumerate([0, 1, 2]):
            indices = torch.where(
                (tokenized_token_type == event_type)
                & (local_attention_mask == 1)
            )[0]

            if len(indices) == 0:
                continue

            num_to_mask = int(mask_ratio[idx] * len(indices))
            if num_to_mask <= 0:
                continue

            selected_indices = indices[
                torch.randperm(len(indices))[:num_to_mask]
            ]

            # 80 / 10 / 10 MLM rule.
            selected_indices = selected_indices[
                torch.randperm(len(selected_indices))
            ]
            n = len(selected_indices)
            num_80 = int(n * 0.8)
            num_10 = int(n * 0.1)

            mask_80 = selected_indices[:num_80]
            rand_10 = selected_indices[num_80 : num_80 + num_10]
            keep_10 = selected_indices[num_80 + num_10 :]

            # 80% -> replace event/unit/value/time with mask values.
            tokenized_token[mask_80] = mask_label_token
            tokenized_units[mask_80] = mask_label_token
            tokenized_values[mask_80] = mask_token
            tokenized_offsets[mask_80] = mask_token
            mask_labels[mask_80] = labels[mask_80]

            # 10% -> random event token; other masked attributes follow old code.
            if len(rand_10) > 0:
                tokenized_token[rand_10] = torch.randint(
                    0,
                    self.vocab_size,
                    (len(rand_10),),
                    device=tokenized_token.device,
                )
                tokenized_units[rand_10] = mask_label_token
                tokenized_values[rand_10] = mask_token
                tokenized_offsets[rand_10] = mask_token
                mask_labels[rand_10] = labels[rand_10]

            # 10% -> keep original input, but predict original event.
            mask_labels[keep_10] = labels[keep_10]

        # --------------------------------------------------------------
        # Value prediction (VP)
        # --------------------------------------------------------------
        # Model input: RAW values.
        # VP regression label: event-wise standardized value using PT-TRAIN stats.
        value_labels = torch.full_like(
            tokenized_values,
            ignore_token,
            dtype=torch.float32,
        )

        if valid_value_mask is None:
            valid_value_mask = torch.ones_like(
                original_token,
                dtype=torch.bool,
            )
        else:
            valid_value_mask = valid_value_mask.to(torch.bool)

        if value_mask_ratio != 0:
            for item_id in VP_VALUE_TARGET_IDS:
                if int(item_id) not in self.vp_stats:
                    continue

                item_mask_candidate = torch.where(
                    (original_token == int(item_id))
                    & (mask_labels == ignore_token)
                    & (local_attention_mask == 1)
                    & valid_value_mask
                )[0]

                # Bernoulli sampling:
                # each eligible value is independently selected with probability
                # `value_mask_ratio`. This avoids the old int(ratio * n) behavior,
                # which selected zero values whenever n was small (e.g. n < 7
                # for ratio=0.15).
                selection_mask = (
                    torch.rand(
                        len(item_mask_candidate),
                        device=item_mask_candidate.device,
                    )
                    < value_mask_ratio
                )
                selected_value_indices = item_mask_candidate[selection_mask]

                if selected_value_indices.numel() == 0:
                    continue

                raw_targets = original_values[selected_value_indices]
                standardized_targets = self._standardize_vp_targets(
                    int(item_id),
                    raw_targets,
                )

                value_labels[selected_value_indices] = standardized_targets

                # Only masked input positions receive the sentinel.
                # All unmasked value inputs stay on the original RAW scale.
                tokenized_values[selected_value_indices] = mask_token

        return (
            tokenized_token,
            tokenized_units,
            tokenized_values,
            tokenized_offsets,
            tokenized_token_type,
            mask_labels,
            value_labels,
        )

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    @staticmethod
    def _require_label(data, key):
        if key not in data:
            raise KeyError(
                f"Required label {key!r} is missing from stay_id={data.get('stay_id')}."
            )
        return data[key]

    def _phenotype_tensor(self, data):
        missing = [col for col in PHENOTYPING_COLS if col not in data]
        if missing:
            raise KeyError(
                f"Phenotype labels missing from stay_id={data.get('stay_id')}: {missing}"
            )
        return torch.tensor(
            [data[col] for col in PHENOTYPING_COLS],
            dtype=torch.float32,
        )

    def _build_finetune_labels(self, data):
        # New internal MIMIC files do not contain *_nogap labels.
        if self.no_gap:
            raise KeyError(
                "no_gap=True requires *_nogap labels, but the new split generator "
                "does not create them. Use no_gap=False for these datasets."
            )

        if self.selected_data == "hirid":
            keys = [
                "mortality_inicu",
                "los_3days",
                "los_7days",
                "transfusion_12hr",
                "vasopressor_need_12hr",
                "ventilator_need_12hr",
                "shock_8hr",
                "sofa_centralnervous_24hr",
                "sofa_cardiovascular_24hr",
                "sofa_respiratory_24hr",
                "sofa_coagulation_24hr",
                "sofa_liver_24hr",
                "sofa_renal_24hr",
            ]
            task_label = torch.tensor(
                [self._require_label(data, k) for k in keys],
                dtype=torch.int64,
            )
            return task_label, None

        if self.selected_data == "P12":
            if self.window == 24:
                keys = ["mortality_inhospital", "los_3days", "los_7days", "ventilator_need_12hr"]
            elif self.window == 48:
                keys = ["mortality_inhospital", "los_3days", "los_7days"]
            task_label = torch.tensor(
                [self._require_label(data, k) for k in keys],
                dtype=torch.int64,
            )
            return task_label, None

        if self.selected_data == "eicu":
            keys = [
                "mortality_inicu",
                "mortality_48hr",
                "los_3days",
                "los_7days",
                "readmission_30days",
                "transfusion_12hr",
                "vasopressor_need_12hr",
                "ventilator_need_12hr",
                "shock_8hr",
                "sofa_centralnervous_24hr",
                "sofa_cardiovascular_24hr",
                "sofa_respiratory_24hr",
                "sofa_coagulation_24hr",
                "sofa_liver_24hr",
                "sofa_renal_24hr",
            ]
            task_label = torch.tensor(
                [self._require_label(data, k) for k in keys],
                dtype=torch.int64,
            )
            return task_label, self._phenotype_tensor(data)

        if self.task == "phenotype":
            return None, self._phenotype_tensor(data)

        if self.window == "entire":
            keys = ["mortality_30days", "readmission_30days"]
            task_label = torch.tensor(
                [self._require_label(data, k) for k in keys],
                dtype=torch.int64,
            )
            return task_label, self._phenotype_tensor(data)

        # Default MIMIC multitask ordering: preserved from the existing loader.
        keys = [
            "mortality_30days",
            "mortality_inhospital",
            "mortality_inicu",
            "mortality_48hr",
            "los_3days",
            "los_7days",
            "readmission_30days",
            "transfusion_12hr",
            "vasopressor_need_12hr",
            "ventilator_need_12hr",
            "shock_8hr",
            "sofa_centralnervous_24hr",
            "sofa_cardiovascular_24hr",
            "sofa_respiratory_24hr",
            "sofa_coagulation_24hr",
            "sofa_liver_24hr",
            "sofa_renal_24hr",
        ]
        task_label = torch.tensor(
            [self._require_label(data, k) for k in keys],
            dtype=torch.int64,
        )
        return task_label, self._phenotype_tensor(data)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.keys)

    @staticmethod
    def _safe_tensor(seq, dtype, fill_value=3):
        clean_seq = [fill_value if pd.isna(x) else x for x in seq]
        return torch.tensor(clean_seq, dtype=dtype)

    @staticmethod
    def _gender_tensor(gender):
        if isinstance(gender, str):
            g = gender.strip().upper()
            if g == "F":
                value = 0
            elif g == "M":
                value = 1
            else:
                raise ValueError(f"Unsupported gender value: {gender!r}")
        else:
            value = int(gender)
            if value not in {0, 1}:
                raise ValueError(f"Unsupported numeric gender value: {gender!r}")
        return torch.tensor(value, dtype=torch.int64).unsqueeze(0)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data = self.df[key]

        ehr = self._safe_tensor(data["ehr_seq"], torch.int64, fill_value=3)
        unit = self._safe_tensor(data["unit_seq"], torch.int64, fill_value=3)
        raw_value_seq = data["value_seq"]
        valid_value_mask = torch.tensor(
            [
                (not pd.isna(x))
                and math.isfinite(float(x))
                for x in raw_value_seq
            ],
            dtype=torch.bool,
        )

        # IMPORTANT: value remains RAW for the value embedding.
        # Missing/non-finite values are filled with 0 only as a tensor-safe
        # placeholder and are excluded from VP target sampling.
        value = torch.tensor(
            [
                float(x)
                if ((not pd.isna(x)) and math.isfinite(float(x)))
                else 0.0
                for x in raw_value_seq
            ],
            dtype=torch.float32,
        )

        offset = self._safe_tensor(data["offset_seq"], torch.float32, fill_value=0.0)
        token_type = self._safe_tensor(
            data["token_type_seq"],
            torch.int64,
            fill_value=3,
        )

        seq_len = ehr.shape[0]
        if seq_len > self.max_length:
            raise ValueError(
                f"stay_id={data.get('stay_id')} has seq_len={seq_len} "
                f"> max_length={self.max_length}. "
                "Fix the preprocessing/chunking rather than truncating here."
            )

        for name, tensor in {
            "unit": unit,
            "value": value,
            "offset": offset,
            "token_type": token_type,
        }.items():
            if tensor.shape[0] != seq_len:
                raise ValueError(
                    f"stay_id={data.get('stay_id')}: ehr length={seq_len}, "
                    f"{name} length={tensor.shape[0]}"
                )

        age = torch.tensor(
            int(data["age"]),
            dtype=torch.int64,
        ).unsqueeze(0)
        gender = self._gender_tensor(data["gender"])

        attention_mask = torch.zeros(
            self.max_length,
            dtype=torch.long,
        )
        attention_mask[:seq_len] = 1

        cls_token_tensor = torch.tensor(
            self.cls_token_id,
            dtype=torch.int64,
        ).unsqueeze(0)

        if self.mode == "pretrain":
            if self.mask_mode != "mlm":
                raise NotImplementedError(
                    "This cleaned loader currently supports mask_mode='mlm' only."
                )

            (
                masked_ehr,
                masked_unit,
                masked_value,
                masked_offset,
                masked_token_type,
                mask_labels,
                value_labels,
            ) = self.mask_tokens(
                ehr,
                unit,
                value,
                offset,
                token_type,
                attention_mask,
                self.mask_ratio,
                self.value_mask_ratio,
                valid_value_mask=valid_value_mask,
            )

            padded_ehr = torch.zeros(self.max_length, dtype=torch.long)
            padded_unit = torch.zeros(self.max_length, dtype=torch.long)
            padded_value = torch.zeros(self.max_length, dtype=torch.float32)
            padded_offset = torch.zeros(self.max_length, dtype=torch.float32)
            padded_token_type = torch.zeros(self.max_length, dtype=torch.long)
            padded_label = torch.full(
                (self.max_length,),
                -100,
                dtype=torch.long,
            )
            padded_value_label = torch.full(
                (self.max_length,),
                -100,
                dtype=torch.float32,
            )

            padded_ehr[:seq_len] = masked_ehr
            padded_unit[:seq_len] = masked_unit
            padded_value[:seq_len] = masked_value
            padded_offset[:seq_len] = masked_offset
            padded_token_type[:seq_len] = masked_token_type
            padded_label[:seq_len] = mask_labels
            padded_value_label[:seq_len] = value_labels

            # 11 outputs
            # Removed: padded_position, padded_ordercategoryname,
            #          padded_ordercategorydescription
            return (
                padded_ehr,
                attention_mask,
                age,
                gender,
                padded_value,
                padded_unit,
                padded_offset,
                padded_token_type,
                cls_token_tensor,
                padded_label,
                padded_value_label,
            )

        # Finetuning: no masking. No fixed-length padding here — batches are
        # padded to their own max length in collate_fn (dynamic padding).
        attention_mask = torch.ones(seq_len, dtype=torch.long)

        task_label, multilabel_label = self._build_finetune_labels(data)

        common = (
            ehr,
            attention_mask,
            age,
            gender,
            value,
            unit,
            offset,
            token_type,
            cls_token_tensor,
        )

        if self.selected_data in {"hirid", "P12"}:
            return (*common, task_label)

        if self.task == "phenotype":
            return (*common, multilabel_label)

        # eICU / entire / default MIMIC multitask.
        return (*common, task_label, multilabel_label)
