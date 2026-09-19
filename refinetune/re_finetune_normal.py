import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb
from sklearn.metrics import precision_score
from accelerate import Accelerator
from accelerate import DistributedType, DistributedDataParallelKwargs
from peft import get_peft_model, LoraConfig, TaskType
import os
from utils.utils import seed_everything
from transformers import LongformerTokenizer
from re_dataset import EHR_Longformer_Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
import torch.nn as nn
from re_longformernormal import LongformerPretrainNormal, LongformerFinetune, LongformerFinetuneforMultiTask, LongformerFinetuneforPhenotype, LongformerFinetuneforMultiTask_lora
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ExponentialLR, LambdaLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
# from finetune_train_grouping import train
from re_finetune_train import train, train_phenotype
import logging
import sys
from collections import Counter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler
from torch.optim.lr_scheduler import _LRScheduler
import math
import torch.nn.init as init
import warnings
from torch.utils.data import WeightedRandomSampler, DataLoader
import numpy as np
from collections import Counter
import random
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.distributed as dist
from utils.sampler import RandomOversamplingDistributedSampler
from torch.utils.data import ConcatDataset
import bitsandbytes as bnb
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Initialize FocalLoss.
        
        Parameters:
        - gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted.
        - alpha (float or Tensor): Weighting factor for the positive class. Can be a scalar or a tensor of shape (num_classes,).
        - reduction (str): The method used to reduce the loss. Options are 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Convert alpha to a tensor if it's a single float value
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float32)
            elif isinstance(alpha, list):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                raise ValueError("Alpha must be a float, int, or a list of two floats.")
        else:
            self.alpha = None
        
    def forward(self, inputs, targets):
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # Cross-entropy loss per sample
        
        probs = torch.exp(-ce_loss)  # probs = 1 - ce_loss when using log_softmax implicitly in cross_entropy
        probs = torch.clamp(probs, min=1e-8, max=1 - 1e-8)
        focal_loss = ((1 - probs) ** self.gamma) * ce_loss

        if self.alpha is not None:
            self.alpha = self.alpha.to(targets.device)  # Ensure alpha is on the same device
            alpha_t = self.alpha.gather(0, targets)  # Gather alpha values based on targets
            focal_loss = alpha_t * focal_loss  # Apply alpha scaling

        # Reduce loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

        # ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # Shape: (batch_size,)
        
        # # Compute probabilities for each class
        # probs = F.softmax(inputs, dim=1)  # Shape: (batch_size, num_classes)
        
        # # Extract probabilities for the true class
        # pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # Shape: (batch_size,)
        # focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # if self.alpha is not None:
        #     # self.alpha = self.alpha.to(targets.device)
        #     alpha_t = self.alpha.gather(0, targets)  # Shape: (batch_size,)
        #     focal_loss = alpha_t * focal_loss
        
        
        # # Reduce loss
        # if self.reduction == 'mean':
        #     return focal_loss.mean()
        # elif self.reduction == 'sum':
        #     return focal_loss.sum()
        # else:
        #     return focal_loss
        

class CB_Loss(nn.Module):
    def __init__(self, beta, gamma, class_counts):
        super(CB_Loss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.class_counts = class_counts

        # Effective number of samples
        effective_num = 1.0 - torch.pow(self.beta, self.class_counts)
        self.class_weights = (1.0 - self.beta) / effective_num
        self.class_weights = self.class_weights / self.class_weights.sum() * len(self.class_counts)
    
    def forward(self, logits, labels):
        labels_one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
        weights = self.class_weights[labels].unsqueeze(1)
        
        focal_weight = torch.pow(1 - torch.softmax(logits, dim=1), self.gamma)
        loss = F.cross_entropy(logits, labels, reduction='none')
        loss = focal_weight * loss
        loss = weights * loss

        return loss.mean()
    
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, beta, gamma=2.0, alpha=None, loss_type="focal", no_of_classes=None):
        """
        Initialize Class-Balanced Focal Loss.

        Parameters:
        - beta (float): Hyperparameter for Class-Balanced Loss.
        - gamma (float): Focusing parameter for Focal Loss, adjusts rate at which easy examples are down-weighted.
        - alpha (Tensor or None): Optional weight for each class.
        - loss_type (str): The type of loss to compute ("focal", "sigmoid", "softmax").
        - no_of_classes (int): The number of classes.
        """
        super(ClassBalancedFocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.loss_type = loss_type
        self.no_of_classes = no_of_classes

    def focal_loss(self, labels, logits, alpha, gamma):
        """
        Compute the Focal Loss.
        
        Parameters:
        - labels (Tensor): One-hot encoded ground truth labels.
        - logits (Tensor): Model predictions (logits).
        - alpha (Tensor): Class weighting factor.
        - gamma (float): Focal loss focusing parameter.
        
        Returns:
        - Focal loss (Tensor): The computed Focal loss.
        """
        bce_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction='none')
        
        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-logits)))

        loss = modulator * bce_loss
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
        focal_loss /= torch.sum(labels)
        return focal_loss

    def class_balanced_loss(self, labels, logits, samples_per_cls):
        """
        Compute the Class-Balanced Loss with Focal Loss as an option.

        Parameters:
        - labels (Tensor): Ground truth labels of shape [batch].
        - logits (Tensor): Model predictions (logits) of shape [batch, num_classes].
        - samples_per_cls (list): List with the number of samples per class.
        - no_of_classes (int): Total number of classes.
        
        Returns:
        - Class-balanced loss (Tensor).
        """
        effective_num = 1.0 - np.power(self.beta, samples_per_cls)
        class_weights = (1.0 - self.beta) / np.array(effective_num)
        class_weights = class_weights / np.sum(class_weights) * self.no_of_classes

        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()

        weights = torch.tensor(class_weights).float().unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1).unsqueeze(1).repeat(1, self.no_of_classes)

        if self.loss_type == "focal":
            cb_loss = self.focal_loss(labels_one_hot, logits, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)

        return cb_loss

    def forward(self, labels, logits, samples_per_cls):
        """
        Forward pass for the Class-Balanced Focal Loss.

        Parameters:
        - labels (Tensor): Ground truth labels.
        - logits (Tensor): Model predictions (logits).
        - samples_per_cls (list): List with the number of samples per class.

        Returns:
        - Computed loss (Tensor).
        """
        return self.class_balanced_loss(labels, logits, samples_per_cls)


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mㄴult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        


def configure_optimizers(model, args, n_steps, loss_weighter=None):
    base_lr = args.learning_rate  
    classifier_lr = args.learning_rate * args.classifier_weight

    pretrained_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif "classifier" in name:
            classifier_params.append(param)
        else:
            pretrained_params.append(param)
            
    if loss_weighter is not None:
        optimizer = torch.optim.AdamW([
            {'params': pretrained_params, 'lr': base_lr, 'weight_decay': 0.01},
            {'params': classifier_params, 'lr': classifier_lr, 'weight_decay': 0.0},
            {'params': loss_weighter.parameters(), 'lr': base_lr, 'weight_decay': 0.0}
        ], lr=base_lr, eps=1e-8, betas=(0.9, 0.98))
        
    else:
        optimizer = torch.optim.AdamW([
            {'params': pretrained_params, 'lr': base_lr, 'weight_decay': 0.01},
            {'params': classifier_params, 'lr': classifier_lr, 'weight_decay': 0.01},
        ], lr=base_lr, eps=1e-8, betas=(0.9, 0.98))

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='max',
    #     factor=0.5,
    #     patience=5,
    #     threshold=0.001,  
    #     verbose=True
    # )
    num_warmup_steps = int(n_steps * 0.2)  
    num_training_steps = n_steps 

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler

def configure_optimizers_lora(model, args, n_steps, loss_weighter=None):
    base_lr = args.learning_rate
    classifier_lr = args.learning_rate * args.classifier_weight

    lora_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "lora_" in name or "loraA" in name or "loraB" in name:
            lora_params.append(param)
        elif "classifier" in name:
            classifier_params.append(param)
        else:
            pass

    assert len(lora_params) > 0, "LoRA params not detected! Check PEFT wrapping."
    
    optimizer_grouped_params = [
        {"params": lora_params, "lr": base_lr, "weight_decay": 0.01},
        {"params": classifier_params, "lr": classifier_lr, "weight_decay": 0.0},
    ]

    if loss_weighter is not None:
        optimizer_grouped_params.append(
            {"params": loss_weighter.parameters(), "lr": base_lr, "weight_decay": 0.0}
        )

    optimizer = torch.optim.AdamW(
        optimizer_grouped_params,
        lr=base_lr,
        eps=1e-8,
        betas=(0.9, 0.98),
    )

    num_warmup_steps = int(n_steps * 0.2)
    num_training_steps = n_steps

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler

# def configure_optimizers(model, args, n_steps, loss_weighter=None):
#     base_lr   = getattr(args, "learning_rate", 5e-5)
#     use_lora  = bool(getattr(args, "use_lora", False))

#     lr_head = getattr(args, "lr_head", base_lr * 2)

#     if use_lora:
#         lr_lora = getattr(args, "lr_lora", base_lr * 10)
#     else:
#         lr_lora = base_lr

#     # Encoder body 
#     lr_body = getattr(args, "lr_body", base_lr)

#     # --- weight decay ---
#     wd_head = getattr(args, "wd_head", 0.01)
#     wd_body = getattr(args, "wd_body", 0.01)
#     wd_lora = getattr(args, "wd_lora", 0.0)


#     lora_params, head_params, body_params = [], [], []
#     for name, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         if "lora_" in name:
#             lora_params.append(p)
#         elif (
#             "binary_classifiers" in name
#             or "sofa_classifiers" in name
#             or "phenotype_classifier" in name
#             or "classifier" in name
#         ):
#             head_params.append(p)
#         else:
#             body_params.append(p)

#     param_groups = []
#     if body_params:
#         param_groups.append({"params": body_params, "lr": lr_body, "weight_decay": wd_body})
#     if head_params:
#         param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": wd_head})
#     if lora_params:
#         param_groups.append({"params": lora_params, "lr": lr_lora, "weight_decay": wd_lora})
#     if loss_weighter is not None:
#         param_groups.append({"params": loss_weighter.parameters(), "lr": base_lr, "weight_decay": 0.0})

#     if not param_groups:  # fallback
#         trainable = [p for p in model.parameters() if p.requires_grad]
#         param_groups = [{"params": trainable, "lr": base_lr, "weight_decay": wd_body}]

#     # --- Optimizer ---
# #     optimizer = bnb.optim.AdamW8bit(
# #     param_groups,
# #     betas=(0.9, 0.98),
# #     eps=1e-8,
# # )
    
#     optimizer = torch.optim.AdamW(
#     param_groups,
#     betas=(0.9, 0.98),
#     eps=1e-8,
#     fused=True
# )

#     # --- Scheduler ---
#     warmup_ratio = getattr(args, "warmup_ratio", 0.1)  
#     num_warmup_steps = max(1, int(n_steps * warmup_ratio))
#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=num_warmup_steps,
#         num_training_steps=n_steps,
#     )

#     return optimizer, scheduler


# def configure_optimizers(model, args, n_steps):
#     base_lr = args.learning_rate  
#     lora_lr = args.learning_rate * args.lora_weight
#     classifier_lr = args.learning_rate * args.lora_weight
    
#     pretrained_params = model.pretrained_parameters()
#     lora_params = model.lora_parameters()
#     classifier_params = model.classifier_parameters()

#     optimizer = optim.AdamW([
#         {'params': pretrained_params, 'lr': base_lr},  
#         {'params': lora_params, 'lr': lora_lr},  
#         {'params': classifier_params, 'lr': classifier_lr}, 
#     ])

#     warmup_steps = int(n_steps * 0.1)
#     warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    
#     cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

#     scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

#     return optimizer, {"scheduler": scheduler, "interval": "step"}

# def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
#     def lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
        
#         # Cosine Decay start
#         progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
#         return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * progress)))

#     return LambdaLR(optimizer, lr_lambda, last_epoch)

# def configure_optimizers(model, args, n_steps):
#     base_lr = args.learning_rate  
#     lora_lr = args.learning_rate * args.lora_weight
#     classifier_lr = args.learning_rate * args.lora_weight
    
#     pretrained_params = model.pretrained_parameters()
#     lora_params = model.lora_parameters()
#     classifier_params = model.classifier_parameters()


#     optimizer = optim.AdamW([
#     {'params': pretrained_params, 'lr': base_lr, 'name': 'pretrained_params'},  
#     {'params': lora_params, 'lr': lora_lr, 'name': 'lora_params'},  
#     {'params': classifier_params, 'lr': classifier_lr, 'name': 'classifier_params'}
# ], weight_decay=0.01)

   
#     num_warmup_steps = int(n_steps * 0.1)  
#     num_training_steps = n_steps 

#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=num_warmup_steps,
#         num_training_steps=num_training_steps
#     )

#     return optimizer, {"scheduler": scheduler, "interval": "step"}

def calculate_alpha(dataset, num_classes):
    """
    Calculate the alpha values based on the class distribution in the dataset.
    
    Parameters:
    - dataset: The dataset to calculate class distribution from.
    - num_classes: The total number of classes in the dataset.
    
    Returns:
    - alpha: Tensor containing the alpha values for each class.
    """
    # Get all targets in the dataset
    all_targets = [label.item() for *_, label in dataset]

    # Count the occurrences of each class
    class_counts = Counter(all_targets)

    # Calculate the total number of samples
    total_count = sum(class_counts.values())

    # Calculate alpha as the inverse of the class frequency
    alpha = torch.zeros(num_classes)
    for cls, count in class_counts.items():
        alpha[cls] = total_count / (num_classes * count)
    
    # Normalize alpha to sum to 1
    alpha = alpha / alpha.sum()

    return alpha


def calculate_class_weights(dataset):
    labels = [label.item() for *_, label in dataset]
    

    class_counts = Counter(labels)
    total_samples = len(dataset)
      
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_samples / (len(class_counts) * count)
    

    weights = torch.tensor([class_weights[i] for i in range(len(class_counts))], dtype=torch.float32)
    
    return weights

def calculate_weights_for_sampler(dataset):
    labels = [label.item() for *_, label in dataset]
    class_counts = Counter(labels)

    weights = [1.0 / class_counts[label] for label in labels]
    
    return weights, class_counts

def normalize_class_weights(class_weights):
    total = sum(class_weights.values())
    return {k: v / total for k, v in class_weights.items()}



class MultiTaskUncertaintyWeighting(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_binary = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_multiclass = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_multilabel = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, binary_loss, multiclass_loss, multilabel_loss):
        loss = (
            torch.exp(-2 * self.log_sigma_binary) * binary_loss +
            torch.exp(-2 * self.log_sigma_multiclass) * multiclass_loss +
            torch.exp(-2 * self.log_sigma_multilabel) * multilabel_loss + 
            self.log_sigma_binary + self.log_sigma_multiclass + self.log_sigma_multilabel
        )
        
        return loss
    
def calculate_multilabel_pos_weight(data_loader, num_labels, device, task_type="phenotype"):
    pos_counts = torch.zeros(num_labels, dtype=torch.float64)
    total_counts = torch.zeros(num_labels, dtype=torch.float64)

    for batch in data_loader:
        if task_type == "phenotype":
            multilabel_labels = batch[-1]  
        elif task_type == "multitask":
            multilabel_labels = batch[-1]  
        else:
            raise ValueError(f"Task {task_type} does not have multilabel labels")

        multilabel_labels = multilabel_labels.float()  # shape: [B, num_labels]

        pos_counts += multilabel_labels.sum(dim=0)
        total_counts += multilabel_labels.size(0)

    neg_counts = total_counts - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-8)
    return pos_weight.to(device)



def apply_lora_to_encoder(pretrained_model, args):

    base_embeddings = getattr(pretrained_model, "embeddings", None)
    if base_embeddings is None and hasattr(pretrained_model.encoder, "embeddings"):
        base_embeddings = pretrained_model.encoder.embeddings
    if base_embeddings is not None:
        for p in base_embeddings.parameters():
            p.requires_grad = False

    target_modules = [s.strip() for s in args.lora_targets.split(",") if s.strip()]

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    peft_encoder = get_peft_model(pretrained_model.encoder, lora_cfg)
    pretrained_model.encoder = peft_encoder  

    for name, p in pretrained_model.encoder.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    return pretrained_model  


def print_trainable_params(module, prefix="[LoRA]"):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"{prefix} trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")


def main(args):
    warnings.filterwarnings('ignore')
    if args.single_task and not (str(args.selected_data).lower() == "p12" and args.window == 48):
        raise ValueError("--single_task is only supported for --selected_data P12 with --window 48.")
    if args.tuning_mode == "linearprobing" and not args.pretrain:
        print(
            "[WARNING] --tuning_mode linearprobing without --pretrain: "
            "the backbone stays at its RANDOM initialization and is frozen. "
            "This is the random-init linear-probe baseline, not a real probe "
            "of pretrained representations."
        )
    if args.use_lora:
        raise ValueError(
            "--use_lora is not combined with the revised --tuning_mode interface. "
            "Use --tuning_mode finetune for PULSE-ICU full fine-tuning or "
            "--tuning_mode linearprobing for a frozen-backbone linear probe."
        )
    if args.freeze:
        print(
            "[WARNING] --freeze is deprecated in this revised pipeline. "
            "Use --tuning_mode linearprobing instead."
        )

    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=False)]
    accelerator = Accelerator(
        mixed_precision="bf16" if args.gpu_mixed_precision else "no",
        gradient_accumulation_steps=args.acc,
        kwargs_handlers=kwargs_handlers,
    )
    print(f"Distributed Type: {accelerator.distributed_type}")
    device = accelerator.device
    
    
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
        
    seed_everything(args.seed)

    
    save_path = Path(args.save_path) / args.mode
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Logger 
    if accelerator.is_local_main_process:
        logging.basicConfig(
            format="[%(asctime)s][%(levelname)s]\t %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
            level=logging.INFO,
            filename=os.path.join(save_path, "exp.log"),
            filemode='w',
        )
   
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    
    itemid2idx = pd.read_pickle("/home/DAHS2/DAHS_EHR/datasets/new_label2idx.pkl")
    unit2idx = pd.read_pickle("/home/DAHS2/DAHS_EHR/datasets/new_unit2idx.pkl")
    idx2label = pd.read_pickle("/home/DAHS2/DAHS_EHR/datasets/new_idx2label.pkl") ##############
    
    # if args.window == 48:
    #     multitask_labels = ["mortality_30days",
    #     "mortality_inhospital",
    #     "mortality_icu",
    #     "mortality48hr",
    #     "los_7days",
    #     "readmission_30days",
    #     "transfusion_12hr",
    #     "shock_8hr",
    #     "vasopressor_need_12hr",
    #     "ventilation_need_12hr",
    #     ]
    # else:
    #     multitask_labels = ["mortality_30days",
    #     "mortality_inhospital",
    #     "mortality_icu",
    #     "mortality48hr",
    #     "los_3days",
    #     "los_7days",
    #     "readmission_30days",
    #     "transfusion_12hr",
    #     "vasopressor_need_12hr",
    #     "ventilation_need_12hr",
    #     "shock_8hr",
    #     ]
    
    multitask_labels = ["mortality_30days",
        "mortality_inhospital",
        "mortality_icu",
        "mortality48hr",
        "los_3days",
        "los_7days",
        "readmission_30days",
        "transfusion_12hr",
        "vasopressor_need_12hr",
        "ventilation_need_12hr",
        "shock_8hr",
        ]
        
    sofa_labels = [
        "sofa_centralnervous_24hr",
        "sofa_cardiovascular_24hr",
        "sofa_respiratory_24hr",
        "sofa_coagulation_24hr",
        "sofa_liver_24hr",
        "sofa_renal_24hr",
    ]
            
    if args.selected_data in ['final', 'benchmark', 'eicu']:
            
        multilabel_labels = [
                            'Acute and unspecified renal failure',
                            'Acute cerebrovascular disease',
                            'Acute myocardial infarction',
                            'Cardiac dysrhythmias',
                            'Chronic kidney disease',
                            'Chronic obstructive pulmonary disease and bronchiectasis',
                            'Complications of surgical procedures or medical care',
                            'Conduction disorders',
                            'Congestive heart failure; nonhypertensive',
                            'Coronary atherosclerosis and other heart disease',
                            'Diabetes mellitus with complications',
                            'Diabetes mellitus without complication',
                            'Disorders of lipid metabolism',
                            'Essential hypertension',
                            'Fluid and electrolyte disorders',
                            'Gastrointestinal hemorrhage',
                            'Hypertension with complications and secondary hypertension',
                            'Other liver diseases',
                            'Other lower respiratory disease',
                            'Other upper respiratory disease',
                            'Pleurisy; pneumothorax; pulmonary collapse',
                            'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
                            'Respiratory failure; insufficiency; arrest (adult)',
                            'Septicemia (except in labor)',
                            'Shock'
                        ]
    else:
        multilabel_labels = []
    
    
    
    # WandB initialization
    if not args.debug:
        if accelerator.is_local_main_process:
            logging.info("Wandb initialization")
            wandb.init(project="EHR_" + args.task,
                    name=args.exp_name + "_resume" if args.resume else args.exp_name,
                    config=args,
                    )
        
    logging.info("Setting Model")
    # Downstream finetuning uses the raw-value embedding, but not the
    # value-prediction pretraining objective/head.
    use_value_prediction = False
    pretrained_model = LongformerPretrainNormal(
        # idx2label=idx2label, #################
        # idx2ordername=idx2ordername,
        # idx2orderdescription=idx2orderdescription,
        name_size=args.name_size,
        description_size=args.description_size,
        token_type_size=args.token_type_size,
        vocab_size=args.vocab_size,
        itemid_size=args.itemid_size,
        # embedding_tokenizer=embedding_tokenizer,
        # embedding_model=embedding_model,
        # embedding_map=embedding_map,
        max_position_embeddings=args.max_position_embeddings,
        unit_size=args.unit_size,
        task_size=args.task_size,
        max_age=args.max_age,
        gender_size=args.gender_size,
        embedding_size=args.embedding_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        learning_rate=args.learning_rate,
        dropout_prob=args.dropout_prob,
        loss_factor=args.loss_factor,
        use_discriminator=args.use_discriminator,
        use_value_prediction=use_value_prediction,
        gpu_mixed_precision=args.gpu_mixed_precision,
        args=args
    ).to(device)
    
    pretrained_model.config.classifier_dropout = 0.3
    pretrained_model.config.num_labels = args.num_labels
    if args.pretrain:
        pretrain_path = Path(args.pretrain_path)
        if not pretrain_path.exists():
            pretrain_path = Path(args.save_path) / args.pretrain_path
        if not pretrain_path.exists():
            raise FileNotFoundError(
                f"Pretraining checkpoint not found: {args.pretrain_path}"
            )

        checkpoint = torch.load(pretrain_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        cleaned_state = {}
        for key, value in state_dict.items():
            if key.startswith("module.module."):
                key = key[14:]
            elif key.startswith("module."):
                key = key[7:]
            cleaned_state[key] = value

        # Fine-tuning needs the learned EHR embedding + Longformer encoder only.
        # MLM / VP / discriminator heads are intentionally ignored.
        backbone_state = {
            key: value
            for key, value in cleaned_state.items()
            if key.startswith("embeddings.") or key.startswith("encoder.")
        }

        missing, unexpected = pretrained_model.load_state_dict(
            backbone_state,
            strict=False,
        )
        critical_missing = [
            key for key in missing
            if key.startswith("embeddings.") or key.startswith("encoder.")
        ]
        if critical_missing:
            raise RuntimeError(
                "Missing pretrained backbone parameters: "
                f"{critical_missing[:20]}"
            )

        print(f"Pre-trained backbone loaded successfully: {pretrain_path}")

    else:
        print("No pretrained backbone loaded.")
    
    print("freeze", args.freeze)
    
    
    # if args.window == 48:
    #     model = LongformerFinetuneforMultiTask(
    #         pretrained_model=pretrained_model,
    #         num_labels=args.num_labels-1,
    #         classifier_dropout=args.classifier_dropout,
    #         freeze_pretrained=False,
    #         freeze_layers=args.freeze_layers,
    #         ablation=args.ablation,
    #         args=args
    #     ).to(device)
    # else:
    #     model = LongformerFinetuneforMultiTask(
    #         pretrained_model=pretrained_model,
    #         num_labels=args.num_labels,
    #         # num_tasks=args.num_tasks,
    #         num_binary_tasks=args.num_binary_tasks,
    #         num_sofa_tasks=args.num_sofa_tasks,
    #         classifier_dropout=args.classifier_dropout,
    #         freeze_pretrained=False,
    #         freeze_layers=args.freeze_layers,
    #         ablation=args.ablation,
    #         args=args
    #     ).to(device)
    
    def set_longformer_window(model, win):
        enc = model.encoder  # LongformerModel
        for layer in enc.encoder.layer:
            layer.attention.self.one_sided_attn_window_size = win
        if hasattr(enc.config, "attention_window"):
            enc.config.attention_window = [win] * len(enc.encoder.layer)
        print(f"[INFO] Longformer attention_window set to {win}")

    
    if args.task == "multitask":
        if args.use_lora:
            print("use lora!!")
            model = LongformerFinetuneforMultiTask_lora(
                pretrained_model=pretrained_model,
                num_labels=args.num_labels,
                # num_tasks=args.num_tasks,
                num_binary_tasks=args.num_binary_tasks,
                num_sofa_tasks=args.num_sofa_tasks,
                classifier_dropout=args.classifier_dropout,
                freeze_pretrained=False,
                freeze_layers=args.freeze_layers,
                ablation=args.ablation,
                args=args
            ).to(device)
            
            # set_longformer_window(model, 256)
            
    #         print([layer.attention.self.one_sided_attn_window_size 
    #    for layer in model.encoder.encoder.layer])
        else:
            model = LongformerFinetuneforMultiTask(
                    pretrained_model=pretrained_model,
                    num_labels=args.num_labels,
                    # num_tasks=args.num_tasks,
                    num_binary_tasks=args.num_binary_tasks,
                    num_sofa_tasks=args.num_sofa_tasks,
                    classifier_dropout=args.classifier_dropout,
                    freeze_pretrained=False,
                    freeze_layers=args.freeze_layers,
                    ablation=args.ablation,
                    args=args
                ).to(device)
    elif args.task == "phenotype":
        model = LongformerFinetuneforPhenotype(
                pretrained_model=pretrained_model,
                num_labels=args.num_labels,
                classifier_dropout=args.classifier_dropout,
                freeze_pretrained=False,
                freeze_layers=args.freeze_layers,
                ablation=args.ablation,
                args=args
            ).to(device)
    
    # EHREmbedding_finetune is a separate module instance. Initialize it
    # from the revised pretrained embedding rather than leaving it random.
    if (
        args.pretrain
        and hasattr(model, "embedding")
        and hasattr(pretrained_model, "embeddings")
        and model.embedding is not pretrained_model.embeddings
    ):
        emb_missing, emb_unexpected = model.embedding.load_state_dict(
            pretrained_model.embeddings.state_dict(),
            strict=False,
        )
        if emb_missing or emb_unexpected:
            print(
                "[Finetune embedding init] "
                f"missing={emb_missing[:20]}, unexpected={emb_unexpected[:20]}"
            )
        else:
            print("[Finetune embedding init] pretrained embedding copied.")

    # ----------------------------------------------------------
    # Explicit tuning policy
    # ----------------------------------------------------------
    # The old `freeze_pretrained` flag did not freeze the entire encoder.
    # We therefore enforce the requested mode explicitly here.
    if args.tuning_mode == "finetune":
        # PULSE-ICU full fine-tuning:
        #   - custom EHR embedding: trainable
        #   - Longformer encoder layers: trainable
        #   - Longformer internal embedding block: FROZEN
        #
        # The internal Longformer embedding block was frozen during
        # pretraining, so it must remain frozen downstream to preserve the
        # pretrained architecture and positional signal exactly as learned.
        if hasattr(model, "embedding"):
            for p in model.embedding.parameters():
                p.requires_grad = True

        if hasattr(model, "encoder"):
            for p in model.encoder.parameters():
                p.requires_grad = True

            if hasattr(model.encoder, "embeddings"):
                for p in model.encoder.embeddings.parameters():
                    p.requires_grad = False

        # Downstream task-token rows were not trained during pretraining.
        # Reinitialize them and learn them jointly during full finetuning.
        if hasattr(model, "embedding") and hasattr(model.embedding, "task_embedding"):
            torch.nn.init.xavier_uniform_(
                model.embedding.task_embedding.task_embedding.weight
            )

        print(
            "[Tuning mode] FINETUNE: custom EHR embedding + Longformer encoder "
            "layers + attention pooling + classifier are trainable; "
            "Longformer internal embeddings remain frozen."
        )

    elif args.tuning_mode == "linearprobing":
        # Strict frozen-backbone probing. The model forward uses a fixed
        # parameter-free mean pooling in this mode, so only the one-layer
        # downstream classifier heads remain trainable.
        if hasattr(model, "embedding"):
            # Pretraining used task ID 0. Downstream task IDs were not trained
            # during pretraining, so reuse the learned pretraining task token
            # for every downstream task before freezing the embedding.
            if hasattr(model.embedding, "task_embedding"):
                with torch.no_grad():
                    task_weight = model.embedding.task_embedding.task_embedding.weight
                    pretrained_task = task_weight[0].detach().clone()
                    task_weight.copy_(pretrained_task.unsqueeze(0).expand_as(task_weight))

            for p in model.embedding.parameters():
                p.requires_grad = False

        if hasattr(model, "encoder"):
            for p in model.encoder.parameters():
                p.requires_grad = False

        # MultiTask full fine-tuning uses a learned attention pooling module.
        # It is bypassed in linear-probing forward and frozen for clarity.
        if hasattr(model, "shared_pool"):
            for p in model.shared_pool.parameters():
                p.requires_grad = False

        print(
            "[Tuning mode] LINEARPROBING: pretrained embedding + encoder frozen; "
            "only downstream linear classifier head(s) are trainable."
        )

    else:
        raise ValueError(f"Unknown tuning_mode: {args.tuning_mode}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[Tuning mode] trainable params: {trainable:,} / {total:,} "
        f"({100.0 * trainable / max(1, total):.4f}%)"
    )

    # try:
    #     model.gradient_checkpointing_enable()
    # except Exception:
    #     pass
    try:
        model.gradient_checkpointing_disable()
    except:
        pass
    
    
    # DataLoader
    logging.info("Setting Dataset")
    
    print("ablation: ", args.ablation)
    if args.no_gap:
        print("gap: ", False)
    else:
        print("gap: ", True)
        
    if args.window == 0:
        print("window: ", "Entire")
        window = "entire"
    else:
        print("window: ", args.window)
        window = args.window
    
    if args.locate is not None:
        locate = args.locate
    else:
        locate = None
    print("locater: ", locate)

    data_path = Path(args.data_path)
    selected_lower = str(args.selected_data).lower()
    if selected_lower == "p12":
        # Dataset/model code use the canonical uppercase spelling.
        args.selected_data = "P12"
        dataset_selected_data = "P12"
    elif selected_lower in {"hirid", "eicu"}:
        args.selected_data = selected_lower
        dataset_selected_data = selected_lower
    else:
        # Internal MIMIC (e.g. "final"): filenames are
        # finetune_*_24[_ratio]_fold*_<selected_data>.pkl
        dataset_selected_data = selected_lower
    
    if args.task == "multitask":
        train_dataset = EHR_Longformer_Dataset(data_path, "train", tokenizer, itemid2idx, unit2idx, vocab_size=args.vocab_size, use_itemid=True, mode=args.mode, mask_mode=args.mask_mode, task=args.task, seed=args.seed, ablation=args.ablation, window=window, index=args.index, no_gap=args.no_gap, ratio=args.ratio, selected_data=dataset_selected_data, locate=locate)
        valid_dataset = EHR_Longformer_Dataset(data_path, "valid", tokenizer, itemid2idx, unit2idx, vocab_size=args.vocab_size, use_itemid=True, mode=args.mode, mask_mode=args.mask_mode, task=args.task, seed=args.seed, ablation=args.ablation, window=window, index=args.index, no_gap=args.no_gap, ratio=args.ratio, selected_data=dataset_selected_data, locate=locate)
        test_dataset = EHR_Longformer_Dataset(data_path, "test", tokenizer, itemid2idx, unit2idx, vocab_size=args.vocab_size, use_itemid=True, mode=args.mode, mask_mode=args.mask_mode, task=args.task, seed=args.seed, ablation=args.ablation, window=window, index=args.index, no_gap=args.no_gap, ratio=args.ratio, selected_data=dataset_selected_data, locate=locate)
    elif args.task == "phenotype":
        train_dataset = EHR_Longformer_Dataset(data_path, "train", tokenizer, itemid2idx, unit2idx, vocab_size=args.vocab_size, use_itemid=True, mode=args.mode, mask_mode=args.mask_mode, task=args.task, seed=args.seed, ablation=args.ablation, window=window, index=args.index, no_gap=args.no_gap, ratio=args.ratio, selected_data=dataset_selected_data)
        valid_dataset = EHR_Longformer_Dataset(data_path, "valid", tokenizer, itemid2idx, unit2idx, vocab_size=args.vocab_size, use_itemid=True, mode=args.mode, mask_mode=args.mask_mode, task=args.task, seed=args.seed, ablation=args.ablation, window=window, index=args.index, no_gap=args.no_gap, ratio=args.ratio, selected_data=dataset_selected_data)
        test_dataset = EHR_Longformer_Dataset(data_path, "test", tokenizer, itemid2idx, unit2idx, vocab_size=args.vocab_size, use_itemid=True, mode=args.mode, mask_mode=args.mask_mode, task=args.task, seed=args.seed, ablation=args.ablation, window=window, index=args.index, no_gap=args.no_gap, ratio=args.ratio, selected_data=dataset_selected_data)
    
    
    if args.debug:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(16)))  
        valid_dataset = torch.utils.data.Subset(valid_dataset, list(range(16))) 
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(16))) 
    # weights = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())], dtype=torch.float32).to(args.device)
    
    # Custom Sampler
    # sample_weights, class_counts = calculate_weights_for_sampler(train_dataset)
    # ns = int(class_counts[1] /len(train_dataset) * args.batch_size)
    # train_custom_sampler = RandomOversamplingSampler(train_dataset)
    # train_custom_sampler = RandomOversamplingDistributedSampler(train_dataset, num_replicas=world_size, rank=rank, ratio=args.oversampling_ratio)
    
    # valid_custom_sampler = CustomSampler2(valid_dataset, batch_size=args.batch_size)
    # train_custom_sampler = BalancedWeightSampler(train_dataset)
    # valid_custom_sampler = BalancedWeightSampler(valid_dataset)
    
    from torch.nn.utils.rnn import pad_sequence

    SEQ_FIELD_IDX = {0, 1, 4, 5, 6, 7}  # ehr, attention_mask, value, unit, offset, token_type

    def custom_collate_fn(batch):
        fields = list(zip(*batch))
        out = []
        for idx, samples in enumerate(fields):
            if idx in SEQ_FIELD_IDX:
                out.append(pad_sequence(samples, batch_first=True, padding_value=0))
            else:
                out.append(torch.stack(samples, dim=0))
        return tuple(out)

    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size,
                            shuffle=True,  # shuffle should be False if using DistributedSampler
                            # sampler=train_sampler,
                            pin_memory=args.pin_memory, 
                            # num_workers=max(4, os.cpu_count()//2), persistent_workers=True,
                            num_workers=args.num_workers,
                            drop_last=True,
                            collate_fn=custom_collate_fn
                            )
    
    valid_loader = DataLoader(valid_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False,  # Validation should not be shuffled
                            # sampler=valid_sampler,
                            pin_memory=args.pin_memory, 
                            # num_workers=max(4, os.cpu_count()//2), persistent_workers=True,
                            num_workers=args.num_workers,
                            drop_last=False,
                            collate_fn=custom_collate_fn
                            )
    
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size, 
                            shuffle=False,  # Test should not be shuffled
                            pin_memory=args.pin_memory, 
                            # num_workers=max(4, os.cpu_count()//2), persistent_workers=True,
                            num_workers=args.num_workers,
                            drop_last=False,
                            collate_fn=custom_collate_fn
                            )
    
    
    if args.inference_mode and (args.selected_data == 'hirid' or args.selected_data == 'P12' or args.selected_data == 'eicu'):
        inference_dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])
        # inference_dataset = ConcatDataset([train_dataset, valid_dataset])
        test_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False,
                                 pin_memory=args.pin_memory, num_workers=args.num_workers,
                                 persistent_workers=False, drop_last=False,
                                 collate_fn=custom_collate_fn)
        print("inference_dataset: ", len(inference_dataset))
        print("inference_loader: ", len(test_loader))

    else:
    
        print("train_dataset: ", len(train_dataset))
        print("valid_dataset: ", len(valid_dataset))
        print("test_dataset: ", len(test_dataset))
        print("train_loader: ", len(train_loader))
        print("valid_loader: ", len(valid_loader))
        print("test_loader: ", len(test_loader))
    
    
    
    def calculate_multilabel_pos_weight(data_loader, num_labels, device, task_type="phenotype"):
        pos_counts = torch.zeros(num_labels, dtype=torch.float64)
        total_counts = torch.zeros(num_labels, dtype=torch.float64)

        for batch in data_loader:
            if task_type in ["phenotype", "multitask"]:
                multilabel_labels = batch[-1]  
            else:
                raise ValueError(f"Task {task_type} does not have multilabel labels")

            multilabel_labels = multilabel_labels.float()  # [B, num_labels]
            pos_counts += multilabel_labels.sum(dim=0)
            total_counts += multilabel_labels.size(0)

        neg_counts = total_counts - pos_counts
        pos_weight = neg_counts / (pos_counts + 1e-8)
        return pos_weight.to(device)
    
    

    
    # test_loader = DataLoader(test_dataset, 
    #                         batch_size=args.batch_size, 
    #                         # sampler=test_custom_sampler,
    #                         shuffle=False,  # testation should not be shuffled
    #                         pin_memory=args.pin_memory, 
    #                         num_workers=args.num_workers,
    #                         )
    
    # class_weights = calculate_alpha(train_dataset, 2)
    # valid_class_weights = calculate_alpha(valid_dataset, 2)
    
    # print("train:", class_weights)
    
    # # class_weights = [0.25, 0.75]
    # print("valid:", valid_class_weights)
    # print("train:", train_custom_sampler.weights)
    # print("valid:", valid_custom_sampler.weights)
  
    # Loss
    
    if args.loss == "focal_loss":
        # criterion = FocalLoss(gamma=args.gamma, alpha=class_weights.to(args.device), reduction='mean')
        criterion = FocalLoss(gamma=args.gamma, reduction='mean')
        print("loss: focal loss")
    elif args.loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.CrossEntropyLoss()
        print("loss: cross entropy loss")
    elif args.loss == "cb_loss":
        criterion = CB_Loss()
        print("loss: cb loss")
    elif args.loss == "class_balanced_focal_loss":
        criterion = ClassBalancedFocalLoss()
        print("loss: class balanced focal loss")
    elif args.loss == "binary_cross_entropy":
        # pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32).to(args.device)
        # pos_weight = torch.tensor([2.0], dtype=torch.float32).to(args.device)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        multilabel_criterion = nn.BCEWithLogitsLoss()
        criterion = nn.BCEWithLogitsLoss()
        multiclass_criterion = nn.CrossEntropyLoss()
        print("loss: binary cross-entropy with logits")


    # Optimizer, Scheduler, z`radient Scaler
    # n_steps = (len(train_dataset) // (args.batch_size * args.acc)) * args.epochs
    # print("n_steps: ", n_steps)
    
    # optimizer = configure_optimizers(model, args, n_steps)
    # scaler = GradScaler(enabled=args.gpu_mixed_precision)
    # if accelerator.distributed_type == DistributedType.MULTI_GPU:
    #     model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    # model, train_loader, valid_loader, test_loader = accelerator.prepare(
    #     model, train_loader, valid_loader, test_loader
    # )
    
    model, train_loader, valid_loader, test_loader = accelerator.prepare(
        model,
        train_loader,
        valid_loader,
        test_loader,
    )
    
    if args.loss_mode == "weighted":
        loss_weighter = MultiTaskUncertaintyWeighting().to(device)
    else:
        loss_weighter = None
        
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / max(1, args.acc)
    )
    n_steps = optimizer_steps_per_epoch * args.epochs
    print(
        f"optimizer steps/epoch: {optimizer_steps_per_epoch}, "
        f"total optimizer steps: {n_steps}"
    )
    if args.use_lora:
        print("Using LoRA - different learning rates for different parameter groups")
        optimizer, scheduler = configure_optimizers_lora(model, args, n_steps, loss_weighter=loss_weighter)
    else:
        optimizer, scheduler = configure_optimizers(model, args, n_steps, loss_weighter=loss_weighter)
    
    optimizer = accelerator.prepare(optimizer)
    scheduler = scheduler

    # model, optimizer, train_loader, valid_loader, test_loader = accelerator.prepare(
    #     model, optimizer, train_loader, valid_loader, test_loader
    # )
    
    print("Task: ", args.task)
    
    print("Value Embedding Type: ", args.value_embedding_type)
    
    target = accelerator.unwrap_model(model) if "accelerator" in globals() else model
    
    if accelerator.is_local_main_process:
        for name, p in target.named_parameters():
            if p.requires_grad:
                print(f"TRAINABLE: {name} | shape={tuple(p.shape)} | numel={p.numel():,}")
                
                
        print("\n---- LoRA params ----")
        for name, p in target.named_parameters():
            if p.requires_grad and "lora_" in name:
                print(f"LoRA: {name} | {tuple(p.shape)}")

        print("\n---- Non-LoRA trainables ----")
        for name, p in target.named_parameters():
            if p.requires_grad and "lora_" not in name:
                print(f"HEAD/OTHER: {name} | {tuple(p.shape)}")
                
    if getattr(args, "external_validation", False):
        best_model_path  = Path(args.save_path) / f"best_{args.load_exp_name}.pth"
      
        if args.use_lora:
            head_bundle_path = Path(args.save_path) / f"{args.exp_name}_head_bundle.pth"
            adapter_dir      = Path(getattr(args, "adapter_dir", "./adapters"))
            adapter_name     = getattr(args, "adapter_name", getattr(args, "exp_name", "default"))
            adapter_path     = adapter_dir / adapter_name
            
            accelerator.unwrap_model(model).eval()

        accelerator.wait_for_everyone()

        # try:
        #     if hasattr(model, "gradient_checkpointing_disable"):
        #         model.gradient_checkpointing_disable()
        # except Exception:
        #     pass
        

        if getattr(args, "use_lora", False):
            try:
                enc_only = None
                pretrain_path = getattr(args, "pretrain_path", None)
                if pretrain_path:
                    p = Path(pretrain_path)
                    if not p.exists():
                        p = Path(args.save_path) / str(pretrain_path)
                    if p.exists():
                        base = torch.load(p, map_location=device)
                        base_state = base.get("model_state_dict", base)
                        enc_only = {k: v for k, v in base_state.items() if k.startswith("encoder.")}
                        print(f"[LoRA/Infer] Load encoder from pretrain: {p}")
                if enc_only is None:
                    ckpt_best = torch.load(best_model_path, map_location=device)
                    st_best   = ckpt_best.get("model_state_dict", ckpt_best)
                    enc_only  = {k: v for k, v in st_best.items() if k.startswith("encoder.")}
                    print(f"[LoRA/Infer] Load encoder from best ckpt: {best_model_path}")


                missing, unexpected = model.load_state_dict(enc_only, strict=False)
                print(f"[LoRA/Infer] Encoder load done (missing={len(missing)}, unexpected={len(unexpected)})")
            except Exception as e:
                logging.error(f"[LoRA/Infer] Base encoder load failed: {e}")
                raise

            try:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.encoder.load_adapter(str(adapter_path), adapter_name=adapter_name, is_trainable=False)
                unwrapped.encoder.set_adapter(adapter_name)
                print(f"[LoRA/Infer] Adapter set: {adapter_name} from {adapter_path}")
            except Exception as e:
                logging.error(f"[LoRA/Infer] Adapter load failed: {e}")
                raise

            try:
                if head_bundle_path.exists():
                    bundle  = torch.load(head_bundle_path, map_location=device)
                    bstate  = bundle.get("model_state_dict", bundle)
                    missing, unexpected = model.load_state_dict(bstate, strict=False)
                    print(f"[LoRA/Infer] Head/Pool bundle loaded (missing={len(missing)}, unexpected={len(unexpected)})")
                else:
                    ckpt = torch.load(best_model_path, map_location=device)
                    state = ckpt.get('model_state_dict', ckpt)
                    filtered = {k: v for k, v in state.items()
                                if ("lora_" not in k) and (not k.startswith("encoder."))}
                    missing, unexpected = model.load_state_dict(filtered, strict=False)
                    print(f"[LoRA/Infer] Fallback: non-encoder/non-LoRA from best ckpt "
                        f"(missing={len(missing)}, unexpected={len(unexpected)})")
            except Exception as e:
                logging.error(f"[LoRA/Infer] Head/Pool load failed: {e}")
                raise

            active = getattr(accelerator.unwrap_model(model).encoder, "active_adapter", None)
            print("Active adapter ->", active)

        else:
            print("Loading full checkpoint...")
            model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'], strict=False)
            print("Full model loaded from", best_model_path)
            # if not args.pretrain:
            #     print("Loading full checkpoint...")
            #     model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'])
            #     print("Full model loaded from", best_model_path)
            # else:
            #     print("Based on Pretrain")
            
    else:
        print("Training from base...")
                
            

        
    
    # print(model)
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"? TRAINABLE: {name}")
    
    # if args.use_lora:
    #     unfreeze_lora_top_layers_with_frozen_bottom(model, lora_layers=args.lora_layers)
    #     print(f"lora_layers: {args.lora_layers}")
        
    print(f"Train Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    

    
    if args.task == "multitask": 
        train(
            device,
            model,
            train_loader,
            valid_loader,
            test_loader,
            criterion,
            multiclass_criterion,
            multilabel_criterion,
            optimizer,
            scheduler,
            accelerator.scaler,
            accelerator,
            multitask_labels,
            sofa_labels,
            multilabel_labels,
            loss_weighter,
            args.epochs,
            args.start_epoch,
            args.patience,
            args.save_path,
            args
        )
    elif args.task == "phenotype":
        train_phenotype(
            device,
            model,
            train_loader,
            valid_loader,
            test_loader,
            multilabel_criterion,
            optimizer,
            scheduler,
            accelerator,
            multilabel_labels,
            args.epochs,
            args.start_epoch,
            args.patience,
            args.save_path,
            args
        )
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--exp_name", type=str, default="mortality30")
    parser.add_argument("--load_exp_name", type=str, default="mortality30")
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_path", type=str, default="./datasets/new_data_preparation")
    
    # Model parameters
    parser.add_argument("--mode", type=str, default="finetune")
    parser.add_argument("--task", type=str, default="mortality30")
    parser.add_argument("--vocab_size", type=int, default=50265)
    parser.add_argument("--itemid_size", type=int, default=3892)
    parser.add_argument("--unit_size", type=int, default=81)
    parser.add_argument("--gender_size", type=int, default=2)
    parser.add_argument("--task_size", type=int, default=20)
    parser.add_argument("--token_type_size", type=int, default=5)
    parser.add_argument("--name_size", type=int, default=35)
    parser.add_argument("--description_size", type=int, default=12)
    parser.add_argument("--max_position_embeddings", type=int, default=10000)
    parser.add_argument("--max_age", type=int, default=101)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--acc", type=int, default=8)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--embedding_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--num_attention_heads", type=int, default=1)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    # parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--classifier_dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_mixed_precision", type=bool, default=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_labels", type=int, default=1)
    # parser.add_argument("--use_lora", action='store_true', default=False)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.99)
    # parser.add_argument('--lora_weight', type=int, default=2)
    parser.add_argument('--classifier_weight', type=int, default=5)
    parser.add_argument('--adapter_weight', type=int, default=1)
    parser.add_argument("--loss", type=str, default="cross_entropy")
    parser.add_argument("--pretrain", action='store_true', default=False)
    parser.add_argument(
        "--tuning_mode",
        type=str,
        default="finetune",
        choices=["finetune", "linearprobing"],
        help=(
            "finetune: train the custom EHR embedding, Longformer encoder layers, "
            "attention pooling, and task head while keeping Longformer internal "
            "embeddings frozen as in pretraining; linearprobing: freeze the pretrained "
            "representation backbone and train only the linear downstream classifier(s)."
        ),
    )
    parser.add_argument("--clip_interval", type=int, default=1)
    parser.add_argument("--pretrain_path", type=str, default="best_pretrain_model.pth")
    # parser.add_argument("--lora_layers", type=list, default=[8,9,10,11])
    # parser.add_argument("--lora_r", type=int, default=4)
    # parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--loss_factor", type=float, default=0.5)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--use_discriminator", action="store_true", help="Enable discriminator")
    parser.add_argument("--regression_mode", type=bool, default=False)
    parser.add_argument("--similarity_mode", type=bool, default=False)
    parser.add_argument("--similarity_factor", type=float, default=0.25)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--freeze", action="store_true", default=False)
    # parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument("--oversampling_ratio", type=float, default=1.0)
    parser.add_argument("--mask_mode", type=str, default="mlm")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--freeze_layers", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=7)
    parser.add_argument("--num_binary_tasks", type=int, default=11)
    parser.add_argument("--num_sofa_tasks", type=int, default=6)
    parser.add_argument("--num_multiclass_labels", type=int, default=4)
    parser.add_argument("--num_binary_tasks_eicu", type=int, default=9)
    parser.add_argument("--num_sofa_tasks_eicu", type=int, default=6)
    parser.add_argument("--num_binary_tasks_hirid", type=int, default=7)
    parser.add_argument("--num_sofa_tasks_hirid", type=int, default=6)
    parser.add_argument("--num_binary_tasks_P12", type=int, default=1)
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--no_gap", action='store_true', default=False)
    parser.add_argument("--ratio", type=int, default=1)
    parser.add_argument("--external_validation", action='store_true', default=False)
    parser.add_argument("--num_basetask_tasks", type=int, default=1)
    parser.add_argument("--num_intervention_tasks", type=int, default=1)
    parser.add_argument("--value_mask_ratio", type=float, default=0.0, help="Deprecated in finetuning; VP masking is not used downstream.")
    parser.add_argument("--loss_mode", type=str, default="None")
    parser.add_argument("--task_idx", type=lambda x: int(x) if x != "None" else None, default=None, help="Index of single binary task to run (None = run all tasks)")
    parser.add_argument("--single_task", type=str, default=None, choices=["mortality_inhospital"],
                        help="Train only this task (P12, window 48). Default None keeps the 3-task multitask setup.")
    parser.add_argument("--inference_mode", action='store_true', default=False)
    parser.add_argument("--selected_data", type=str, default="all")
    parser.add_argument("--locate", type=str, default=None)
    parser.add_argument("--num_multilabel_labels", type=int, default=25, help="Use LoRA for fine-tuning")
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--adapter_dir", type=str, default="./adapters")
    parser.add_argument("--adapter_name", type=str, default=None) 
    parser.add_argument("--value_embedding_type", type=str, default="continuous", choices=["simple", "continuous"])

    parser.add_argument(
        "--lora_targets",
        type=str,
        default="query,key,value,dense,intermediate.dense,output.dense"
    )
    args = parser.parse_args()
    args.attention_window = [512] * args.num_hidden_layers
    
    main(args=args)