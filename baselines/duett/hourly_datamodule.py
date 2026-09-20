from __future__ import annotations

import pickle
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

BINARY_LABELS = [
    "mortality_inicu", "mortality_48hr", "los_3days", "los_7days", "readmission_30",
    "transfusion_12hr", "shock_8hr", "vasopressor_need_12hr", "ventilator_need_12hr",
]
SOFA_LABELS = [
    "SOFA_centralnervous_24hr", "SOFA_cardiovascular_24hr", "SOFA_respiratory_24hr",
    "SOFA_coagulation_24hr", "SOFA_liver_24hr", "SOFA_renal_24hr",
]
PHENOTYPE_LABELS = [
    "Acute and unspecified renal failure", "Acute cerebrovascular disease",
    "Acute myocardial infarction", "Cardiac dysrhythmias", "Chronic kidney disease",
    "Chronic obstructive pulmonary disease and bronchiectasis",
    "Complications of surgical procedures or medical care", "Conduction disorders",
    "Congestive heart failure; nonhypertensive", "Coronary atherosclerosis and other heart disease",
    "Diabetes mellitus with complications", "Diabetes mellitus without complication",
    "Disorders of lipid metabolism", "Essential hypertension", "Fluid and electrolyte disorders",
    "Gastrointestinal hemorrhage", "Hypertension with complications and secondary hypertension",
    "Other liver diseases", "Other lower respiratory disease", "Other upper respiratory disease",
    "Pleurisy; pneumothorax; pulmonary collapse",
    "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
    "Respiratory failure; insufficiency; arrest (adult)", "Septicemia (except in labor)", "Shock",
]
N_SOFA_CLASSES = 4
D_TARGET = len(BINARY_LABELS) + len(SOFA_LABELS) * N_SOFA_CLASSES + len(PHENOTYPE_LABELS)


def collate_into_seqs(batch):
    """(x, y) per sample -> (zipped x, tuple of y). Mirrors physionet.collate_into_seqs."""
    xs, ys = zip(*batch)
    return tuple(zip(*xs)), ys


def collate_multitask_seqs(batch):
    """(x, (y_binary, y_sofa, y_pheno)) per sample -> (zipped x, stacked y_* tensors)."""
    xs, ys = zip(*batch)
    y_binary = torch.stack([y[0] for y in ys])
    y_sofa = torch.stack([y[1] for y in ys])
    y_pheno = torch.stack([y[2] for y in ys])
    return tuple(zip(*xs)), (y_binary, y_sofa, y_pheno)


class HourlyDuettDataset(Dataset):
    def __init__(self, pickle_path: str | Path):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        self.meta = data.pop("_meta")
        self.stay_ids = list(data.keys())
        self.items = data

    def __len__(self) -> int:
        return len(self.stay_ids)

    def _tensors(self, stay_id):
        item = self.items[stay_id]
        x_ts = torch.tensor(item["x_ts"], dtype=torch.float32)
        x_static = torch.tensor(item["x_static"], dtype=torch.float32)
        times = torch.tensor(item["times"], dtype=torch.float32)
        return x_ts, x_static, times

    @property
    def d_static_num(self) -> int:
        return len(self.meta["x_static_names"])

    @property
    def d_time_series_num(self) -> int:
        return len(self.meta["feature_cols"])


class MimicPretrainDataset(HourlyDuettDataset):
    def __getitem__(self, idx):
        x = self._tensors(self.stay_ids[idx])
        y = torch.zeros(1, dtype=torch.float32)
        return x, y


class EicuFinetuneDataset(HourlyDuettDataset):
    def __init__(self, pickle_path: str | Path):
        super().__init__(pickle_path)
        label_cols = self.meta["label_cols"]
        self._binary_idx = [label_cols.index(name) for name in BINARY_LABELS]
        self._sofa_idx = [label_cols.index(name) for name in SOFA_LABELS]
        self._pheno_idx = [label_cols.index(name) for name in PHENOTYPE_LABELS]

    def __getitem__(self, idx):
        stay_id = self.stay_ids[idx]
        x = self._tensors(stay_id)
        y = self.items[stay_id]["y"]
        y_binary = torch.tensor([y[i] for i in self._binary_idx], dtype=torch.float32)
        y_sofa = torch.tensor([y[i] for i in self._sofa_idx], dtype=torch.int64)
        y_pheno = torch.tensor([y[i] for i in self._pheno_idx], dtype=torch.float32)
        return x, (y_binary, y_sofa, y_pheno)


class MimicPretrainDataModule(pl.LightningDataModule):
    def __init__(self, train_pkl, val_pkl, batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.train_pkl = train_pkl
        self.val_pkl = val_pkl
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if hasattr(self, "train_ds"):
            return
        self.train_ds = MimicPretrainDataset(self.train_pkl)
        self.val_ds = MimicPretrainDataset(self.val_pkl)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=collate_into_seqs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_into_seqs,
        )

    def d_static_num(self) -> int:
        self.setup()
        return self.train_ds.d_static_num

    def d_time_series_num(self) -> int:
        self.setup()
        return self.train_ds.d_time_series_num


class EicuFinetuneDataModule(pl.LightningDataModule):
    def __init__(self, train_pkl, val_pkl, test_pkl, batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.train_pkl = train_pkl
        self.val_pkl = val_pkl
        self.test_pkl = test_pkl
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if hasattr(self, "train_ds"):
            return
        self.train_ds = EicuFinetuneDataset(self.train_pkl)
        self.val_ds = EicuFinetuneDataset(self.val_pkl)
        self.test_ds = EicuFinetuneDataset(self.test_pkl)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=collate_multitask_seqs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_multitask_seqs,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_multitask_seqs,
        )

    def d_static_num(self) -> int:
        self.setup()
        return self.train_ds.d_static_num

    def d_time_series_num(self) -> int:
        self.setup()
        return self.train_ds.d_time_series_num
