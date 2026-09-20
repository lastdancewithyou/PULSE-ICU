from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import duett
from hourly_datamodule import BINARY_LABELS, D_TARGET, N_SOFA_CLASSES, PHENOTYPE_LABELS, SOFA_LABELS

N_BINARY = len(BINARY_LABELS)
N_SOFA = len(SOFA_LABELS)
N_PHENO = len(PHENOTYPE_LABELS)


def multitask_loss(binary_logits, sofa_logits, pheno_logits, y_binary, y_sofa, y_pheno):
    binary_loss = F.binary_cross_entropy_with_logits(binary_logits, y_binary, reduction="none").sum(dim=1).mean()
    sofa_loss = sum(F.cross_entropy(sofa_logits[:, i, :], y_sofa[:, i]) for i in range(sofa_logits.shape[1]))
    pheno_loss = F.binary_cross_entropy_with_logits(pheno_logits, y_pheno)  # default mean: batch x 25 averaged
    return binary_loss + sofa_loss + pheno_loss


class MultiTaskDuett(duett.Model):
    def _split_head(self, y_hat):
        binary_logits = y_hat[:, :N_BINARY]
        sofa_logits = y_hat[:, N_BINARY:N_BINARY + N_SOFA * N_SOFA_CLASSES].reshape(-1, N_SOFA, N_SOFA_CLASSES)
        pheno_logits = y_hat[:, N_BINARY + N_SOFA * N_SOFA_CLASSES:]
        return binary_logits, sofa_logits, pheno_logits

    def _forward_batch(self, batch):
        x, (y_binary, y_sofa, y_pheno) = batch
        batch_size = y_binary.shape[0]
        y_hat = self.forward(self.feats_to_input(x, batch_size))
        binary_logits, sofa_logits, pheno_logits = self._split_head(y_hat)
        loss = multitask_loss(binary_logits, sofa_logits, pheno_logits, y_binary, y_sofa, y_pheno)
        return loss, binary_logits, sofa_logits, pheno_logits, y_binary, y_sofa, y_pheno

    def training_step(self, batch, batch_idx):
        loss, _binary_logits, _sofa_logits, _pheno_logits, y_binary, _y_sofa, _y_pheno = self._forward_batch(batch)
        self.log("train_loss", loss, sync_dist=True, batch_size=y_binary.shape[0])
        return loss

    def on_train_epoch_end(self):
        # Base duett.Model logs self.train_auroc/train_ap here, but this subclass never
        # updates those (unused, single-label) metrics -- skip to avoid a compute() on
        # an empty metric.
        pass

    def on_validation_epoch_start(self):
        self._val_binary_probs = []
        self._val_binary_true = []

    def validation_step(self, batch, batch_idx):
        loss, binary_logits, sofa_logits, pheno_logits, y_binary, y_sofa, y_pheno = self._forward_batch(batch)
        self._val_binary_probs.append(torch.sigmoid(binary_logits).detach().cpu())
        self._val_binary_true.append(y_binary.detach().cpu())
        self.log("val_loss", loss, on_epoch=True, sync_dist=True, prog_bar=True, rank_zero_only=True,
                 batch_size=y_binary.shape[0])
        return loss

    def on_validation_epoch_end(self):
        probs = torch.cat(self._val_binary_probs).numpy()
        trues = torch.cat(self._val_binary_true).numpy()
        task_aurocs = [
            roc_auc_score(trues[:, i], probs[:, i])
            for i in range(trues.shape[1]) if len(np.unique(trues[:, i])) == 2
        ]
        val_auroc = float(np.mean(task_aurocs)) if task_aurocs else 0.0
        self.log("val_auroc", val_auroc, prog_bar=True, rank_zero_only=True)

    def test_step(self, batch, batch_idx):
        loss, binary_logits, sofa_logits, pheno_logits, y_binary, y_sofa, y_pheno = self._forward_batch(batch)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True, batch_size=y_binary.shape[0])
        return {
            "binary_probs": torch.sigmoid(binary_logits).detach().cpu(),
            "sofa_probs": torch.softmax(sofa_logits, dim=-1).detach().cpu(),
            "pheno_probs": torch.sigmoid(pheno_logits).detach().cpu(),
            "y_binary": y_binary.detach().cpu(),
            "y_sofa": y_sofa.detach().cpu(),
            "y_pheno": y_pheno.detach().cpu(),
        }


def random_init_multitask_model(**kwargs):
    """Same finetune hyperparameters as finetune_multitask_model, without loading a checkpoint."""
    return MultiTaskDuett(
        pretrain=False, aug_noise=0., aug_mask=0.5, transformer_dropout=0.5,
        lr=1.e-4, weight_decay=1.e-5, fusion_method="rep_token", **kwargs,
    )


def finetune_multitask_model(ckpt_path, **kwargs):
    """Mirrors duett.fine_tune_model but loads into the MultiTaskDuett subclass."""
    return MultiTaskDuett.load_from_checkpoint(
        ckpt_path, pretrain=False, aug_noise=0., aug_mask=0.5, transformer_dropout=0.5,
        lr=1.e-4, weight_decay=1.e-5, fusion_method="rep_token", **kwargs,
    )
