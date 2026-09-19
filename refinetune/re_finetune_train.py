import logging
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import torch.nn as nn
import time
import os
import torch
from pathlib import Path
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ExponentialLR, LambdaLR, CosineAnnealingWarmRestarts
import numpy as np
from torch.cuda.amp import autocast
from accelerate import Accelerator
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score, average_precision_score, accuracy_score
from utils.sampler import RandomOversamplingDistributedSampler
from torch.autograd import grad
from sklearn.preprocessing import label_binarize
from torchmetrics.functional import auroc, average_precision

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import numpy as np

def calculate_metrics_test(predictions, labels, thresholds): 
    probs = F.softmax(predictions, dim=1)
    # preds = torch.argmax(probs, dim=1)
    
    positive_probs = probs[:, 1]
    preds = (positive_probs >= thresholds).long()
    
    labels = labels.view(-1) 
    
    preds_np = preds.cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy()
    
    if probs.size(1) > 1:
        probs_np = probs[:, 1].cpu().detach().numpy()
    else:
        probs_np = probs[:, 0].cpu().detach().numpy() 
    
    accuracy = round(accuracy_score(labels_np, preds_np), 4)
    precision = round(precision_score(labels_np, preds_np, zero_division=0, pos_label=1), 4)
    recall = round(recall_score(labels_np, preds_np, zero_division=0, pos_label=1), 4)
    f1 = round(f1_score(labels_np, preds_np, zero_division=0, pos_label=1), 4)
    
    
    if len(np.unique(labels_np)) > 1 and np.sum(labels_np) > 0:
        auroc = round(roc_auc_score(labels_np, probs_np), 4)
        auprc = round(average_precision_score(labels_np, probs_np), 4)
        # if auprc > 1:
        #     print("auprc > 1!!!!!!!!!!!!!!!!!")
    else:
        auroc = "one class"
        auprc = -1 
 
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auroc': auroc,
        'auprc': auprc
    }
    return metrics


# def calculate_metrics(predictions, labels): 
#     # positive_class = 1
#     # positive_probs = torch.sigmoid(predictions[:, positive_class])
#     positive_probs = torch.sigmoid(predictions)
#     # preds = torch.argmax(probs, dim=1)
    
    
#     # positive_probs = probs[:, positive_class]
#     # thresholds = 0.4
    
    
#     positive_probs_np = positive_probs.detach().cpu().numpy()
#     labels_np = labels.view(-1).detach().cpu().numpy()
    
#     precisions, recalls, thresholds = precision_recall_curve(labels_np, positive_probs_np)
        
#     f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
#     best_threshold_idx = f1_scores.argmax()
#     best_threshold = thresholds[best_threshold_idx] if len(thresholds) > 0 else 0.5
    
#     preds = (positive_probs >= best_threshold).long()
#     preds_np = preds.cpu().detach().numpy()
    
#     accuracy = round(accuracy_score(labels_np, preds_np), 4)
#     precision = round(precision_score(labels_np, preds_np, zero_division=0, pos_label=1), 4)
#     recall = round(recall_score(labels_np, preds_np, zero_division=0, pos_label=1), 4)
#     f1 = round(f1_score(labels_np, preds_np, zero_division=0, pos_label=1), 4)
    
    
#     if len(np.unique(labels_np)) > 1 and np.sum(labels_np) > 0:
#         auroc = round(roc_auc_score(labels_np, positive_probs_np), 4)
#         auprc = round(average_precision_score(labels_np, positive_probs_np), 4)
#         # if auprc > 1:
#         #     print("auprc > 1!!!!!!!!!!!!!!!!!")
#     else:
#         auroc = None
#         auprc = None
 
#     metrics = {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1_score': f1,
#         'auroc': auroc,
#         'auprc': auprc,
#         'best_threshold': round(best_threshold, 4)
#     }
#     return metrics


def calculate_metrics(predictions, labels): 
    positive_probs = torch.sigmoid(predictions)  # For binary classification
    
    # Handle multi-class case: Use sigmoid or softmax as appropriate
    if predictions.ndim > 1 and predictions.size(1) > 1:
        positive_probs = torch.softmax(predictions, dim=1)[:, 1]  # Assume second column is positive class
    
    # Convert tensors to numpy arrays
    positive_probs_np = positive_probs.detach().cpu().numpy().squeeze()
    labels_np = labels.view(-1).detach().cpu().numpy()

    preds = (positive_probs >= 0.5).long()
    preds_np = preds.cpu().numpy().squeeze()


    # Calculate classification metrics
    accuracy = round(accuracy_score(labels_np, preds_np), 4)
    precision = round(precision_score(labels_np, preds_np, zero_division=0), 4)
    recall = round(recall_score(labels_np, preds_np, zero_division=0), 4)
    f1 = round(f1_score(labels_np, preds_np, zero_division=0), 4)
    print("labels: ", labels_np)
    print("preds: ", preds_np)
    # Calculate AUROC and AUPRC
    try:
        auroc = round(roc_auc_score(labels_np, positive_probs_np), 4)
        auprc = round(average_precision_score(labels_np, positive_probs_np), 4)
    except ValueError:
        auroc, auprc = None, None 


    # Return all metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auroc': auroc,
        'auprc': auprc,
        # 'best_threshold': round(best_threshold, 4)
    }
    return metrics


# def calculate_binary_metrics(preds, labels, num_tasks=7):
#     task_metrics = []

#     for i in range(num_tasks):
#         task_preds = preds[:, i]  # 모델 예측값
#         task_labels = labels[:, i]  # 실제 정답 레이블

#         # AUROC 예외 처리 (한 개의 클래스만 존재하는 경우)
#         if len(np.unique(task_labels)) < 2:  
#             auroc = np.nan  # 또는 0.5 (랜덤 분류 성능)
#             auprc = np.nan
#         else:
#             auroc = roc_auc_score(task_labels, task_preds)
#             auprc = average_precision_score(task_labels, task_preds)

#         task_metrics.append({
#             "auroc": auroc,
#             "auprc": auprc,
#             "accuracy": accuracy_score(task_labels, task_preds > 0.5),
#             "precision": precision_score(task_labels, task_preds > 0.5, zero_division=0),
#             "recall": recall_score(task_labels, task_preds > 0.5, zero_division=0),
#             "f1_score": f1_score(task_labels, task_preds > 0.5, zero_division=0),
#         })

#     return task_metrics


# def calculate_multiclass_metrics(preds, labels, num_tasks=5, num_classes=5):
#     task_metrics = []

#     for i in range(num_tasks):
#         prob = preds[i]    # shape: (N, num_classes)
#         true = labels[i]   # shape: (N,)

#         unique_labels = np.unique(true)
#         pred_classes = np.argmax(prob, axis=1)

#         # 기본 classification metrics
#         task_result = {
#             "accuracy": accuracy_score(true, pred_classes),
#             "precision_macro": precision_score(true, pred_classes, average="macro", zero_division=0),
#             "recall_macro": recall_score(true, pred_classes, average="macro", zero_division=0),
#             "f1_score_macro": f1_score(true, pred_classes, average="macro", zero_division=0),
#             "precision_micro": precision_score(true, pred_classes, average="micro", zero_division=0),
#             "recall_micro": recall_score(true, pred_classes, average="micro", zero_division=0),
#             "f1_score_micro": f1_score(true, pred_classes, average="micro", zero_division=0),
#         }

#         if len(unique_labels) < 2:
#             task_result.update({
#                 "macro_auroc": np.nan,
#                 "macro_auprc": np.nan,
#                 "micro_auroc": np.nan,
#                 "micro_auprc": np.nan,
#             })
#         else:
#             try:
#                 y_true_bin = label_binarize(true, classes=np.arange(num_classes))
#                 y_true_subset = y_true_bin[:, unique_labels]
#                 prob_subset = prob[:, unique_labels]

#                 task_result.update({
#                     "macro_auroc": roc_auc_score(y_true_subset, prob_subset, average="macro", multi_class="ovr"),
#                     "macro_auprc": average_precision_score(y_true_subset, prob_subset, average="macro"),
#                     "micro_auroc": roc_auc_score(y_true_subset, prob_subset, average="micro", multi_class="ovr"),
#                     "micro_auprc": average_precision_score(y_true_subset, prob_subset, average="micro"),
#                 })
#             except ValueError:
#                 task_result.update({
#                     "macro_auroc": np.nan,
#                     "macro_auprc": np.nan,
#                     "micro_auroc": np.nan,
#                     "micro_auprc": np.nan,
#                 })

#         task_metrics.append(task_result)

#     return task_metrics



# def calculate_phenotype_metrics(preds, labels, num_tasks=25):
#     # macro auroc, micro auroc
#     macro_auroc = []
#     macro_auprc = []
    
#     for i in range(num_tasks):
#         task_preds = preds[:, i]
#         task_labels = labels[:, i]

#         # AUROC 예외 처리 (한 개의 클래스만 존재하는 경우)
#         if len(np.unique(task_labels)) < 2:
#             macro_auroc.append(np.nan)
#             macro_auprc.append(np.nan)
#         else:
#             macro_auroc.append(roc_auc_score(task_labels, task_preds))
#             macro_auprc.append(average_precision_score(task_labels, task_preds))
            
#     preds_flat = preds.ravel()
#     labels_flat = labels.ravel()
    
#     try:
#         micro_auroc = roc_auc_score(labels_flat, preds_flat)
#         micro_auprc = average_precision_score(labels_flat, preds_flat)
#     except ValueError:
#         micro_auroc = np.nan
#         micro_auprc = np.nan
        
#     return {
#         "macro_auroc": np.nanmean(macro_auroc),
#         "macro_auprc": np.nanmean(macro_auprc),
#         "micro_auroc": micro_auroc,
#         "micro_auprc": micro_auprc,
#         "per_task_auroc": macro_auroc,
#         "per_task_auprc": macro_auprc,
#     }
def calculate_binary_metrics(preds, labels, num_tasks=7):
    task_metrics = []

    for i in range(num_tasks):
        task_preds = preds[:, i]  # 모델 예측값
        task_labels = labels[:, i]  # 실제 정답 레이블

        # AUROC 예외 처리 (한 개의 클래스만 존재하는 경우)
        if len(np.unique(task_labels)) < 2:  
            auroc = np.nan  # 또는 0.5 (랜덤 분류 성능)
            auprc = np.nan
        else:
            auroc = roc_auc_score(task_labels, task_preds)
            auprc = average_precision_score(task_labels, task_preds)

        task_metrics.append({
            "auroc": auroc,
            "auprc": auprc,
            "accuracy": accuracy_score(task_labels, task_preds > 0.5),
            "precision": precision_score(task_labels, task_preds > 0.5, zero_division=0),
            "recall": recall_score(task_labels, task_preds > 0.5, zero_division=0),
            "f1_score": f1_score(task_labels, task_preds > 0.5, zero_division=0),
        })

    return task_metrics

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import numpy as np

def calculate_multiclass_metrics(preds, labels, num_tasks=5, num_classes=5):
    task_metrics = []

    for i in range(num_tasks):
        prob = preds[i]    # shape: (N, num_classes)
        true = labels[i]   # shape: (N,)

        unique_labels = np.unique(true)
        pred_classes = np.argmax(prob, axis=1)

        # 기본 classification metrics
        task_result = {
            "accuracy": accuracy_score(true, pred_classes),
            "precision_macro": precision_score(true, pred_classes, average="macro", zero_division=0),
            "recall_macro": recall_score(true, pred_classes, average="macro", zero_division=0),
            "f1_score_macro": f1_score(true, pred_classes, average="macro", zero_division=0),
            "precision_micro": precision_score(true, pred_classes, average="micro", zero_division=0),
            "recall_micro": recall_score(true, pred_classes, average="micro", zero_division=0),
            "f1_score_micro": f1_score(true, pred_classes, average="micro", zero_division=0),
        }

        if len(unique_labels) < 2:
            task_result.update({
                "macro_auroc": np.nan,
                "macro_auprc": np.nan,
                "micro_auroc": np.nan,
                "micro_auprc": np.nan,
            })
        else:
            try:
                y_true_bin = label_binarize(true, classes=np.arange(num_classes))
                y_true_subset = y_true_bin[:, unique_labels]
                prob_subset = prob[:, unique_labels]

                task_result.update({
                    "macro_auroc": roc_auc_score(y_true_subset, prob_subset, average="macro", multi_class="ovr"),
                    "macro_auprc": average_precision_score(y_true_subset, prob_subset, average="macro"),
                    "micro_auroc": roc_auc_score(y_true_subset, prob_subset, average="micro", multi_class="ovr"),
                    "micro_auprc": average_precision_score(y_true_subset, prob_subset, average="micro"),
                })
            except ValueError:
                task_result.update({
                    "macro_auroc": np.nan,
                    "macro_auprc": np.nan,
                    "micro_auroc": np.nan,
                    "micro_auprc": np.nan,
                })

        task_metrics.append(task_result)

    return task_metrics



def calculate_phenotype_metrics(preds, labels, num_tasks=25):
    # macro auroc, micro auroc
    macro_auroc = []
    macro_auprc = []
    
    for i in range(num_tasks):
        task_preds = preds[:, i]
        task_labels = labels[:, i]

        # AUROC 예외 처리 (한 개의 클래스만 존재하는 경우)
        if len(np.unique(task_labels)) < 2:
            macro_auroc.append(np.nan)
            macro_auprc.append(np.nan)
        else:
            macro_auroc.append(roc_auc_score(task_labels, task_preds))
            macro_auprc.append(average_precision_score(task_labels, task_preds))
            
    preds_flat = preds.ravel()
    labels_flat = labels.ravel()
    
    try:
        micro_auroc = roc_auc_score(labels_flat, preds_flat)
        micro_auprc = average_precision_score(labels_flat, preds_flat)
    except ValueError:
        micro_auroc = np.nan
        micro_auprc = np.nan
        
    return {
        "macro_auroc": np.nanmean(macro_auroc),
        "macro_auprc": np.nanmean(macro_auprc),
        "micro_auroc": micro_auroc,
        "micro_auprc": micro_auprc,
        "per_task_auroc": macro_auroc,
        "per_task_auprc": macro_auprc,
    }
            
            
            

# def find_best_threshold(probs_np, labels_np):
#     precision, recall, thresholds = precision_recall_curve(labels_np, probs_np)
    
#     f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

#     best_threshold = thresholds[f1_scores.argmax()]
    
#     return best_threshold, f1_scores.max()

# def calculate_metrics(predictions, labels): 
#     probs = F.softmax(predictions, dim=1)
#     probs_np = probs[:, 1].cpu().detach().numpy()
#     labels_np = labels.view(-1).cpu().detach().numpy()

#     best_threshold, best_f1 = find_best_threshold(probs_np, labels_np)

#     preds_np = (probs_np >= best_threshold).astype(int)
    
#     accuracy = accuracy_score(labels_np, preds_np) 
#     precision = precision_score(labels_np, preds_np, zero_division=0)
#     recall = recall_score(labels_np, preds_np, zero_division=0)
#     f1 = f1_score(labels_np, preds_np, zero_division=0)
    
#     if len(np.unique(labels_np)) > 1:
#         auroc = roc_auc_score(labels_np, probs_np)
#         auprc = average_precision_score(labels_np, probs_np)
#     else:
#         auroc = "one class"
#         auprc = -1

#     metrics = {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1_score': f1,
#         'auroc': auroc,
#         'auprc': auprc,
#         'best_threshold': best_threshold,
#         'best_f1': best_f1
#     }
#     return metrics

def get_attr(model, name):
    if hasattr(model, "module"):
        return getattr(model.module, name)
    else:
        return getattr(model, name)
    


def train(
    device: torch.device,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    binary_criterion: torch.nn,
    multiclass_criterion: torch.nn,
    multilabel_criterion: torch.nn,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    scaler: torch.amp.GradScaler,
    accelerator: Accelerator,
    multitask_labels: list,
    multiclass_labels: list,
    multilabel_labels: list,
    loss_weighter: torch.nn.Module,
    epochs: int,
    start_epoch: int,
    patience: int,
    save_path: Path,
    args: dict,
):
    use_lora = bool(getattr(args, "use_lora", False))
    if use_lora:
        adapter_dir = Path(getattr(args, "adapter_dir", "./adapters"))

        # 1) adapter_name이 None이거나 빈 문자열이면 exp_name으로 fallback
        adapter_name = getattr(args, "adapter_name", None) or getattr(args, "exp_name", None) or "default"

        if adapter_name is None:
            # 여기까지 올 일은 없지만, 안전빵으로 한 번 더 체크
            raise ValueError("adapter_name check")

        adapter_path = adapter_dir / adapter_name

        adapter_source = getattr(args, "adapter_source", "self")
        adapter_base_ckpt = getattr(
            args,
            "adapter_base_ckpt",
            getattr(args, "pretrain_path", None),
        )

        save_path = Path(save_path)

        # adapter_name도 문자열이므로 그대로 사용 가능
        default_head_path = save_path / f"{adapter_name}_head_bundle.pth"

        head_bundle_path = Path(getattr(args, "head_bundle_path", default_head_path))
    if accelerator.is_local_main_process and use_lora:
        adapter_path.mkdir(parents=True, exist_ok=True)
    
    if args.selected_data == "hirid":
        needed_binary_idx = [2, 4, 5, 7, 8, 9, 10]
    elif args.selected_data == "P12":
        if args.window == 24:
            needed_binary_idx = [1, 4, 5, 9]
        elif args.window == 48:
            needed_binary_idx = [1] if getattr(args, "single_task", None) else [1, 4, 5]
    elif args.selected_data == "eicu":
        needed_binary_idx = [idx for idx in range(2, 11)]

    elif args.window == 0:
        args.window = 'entire'
        needed_binary_idx = [0, 6]

    else:
        needed_binary_idx = [idx for idx in range(args.num_binary_tasks)]

    binary_tasks = [multitask_labels[i] for i in needed_binary_idx]

    if args.selected_data == "hirid":
        num_binary_tasks = args.num_binary_tasks_hirid
        num_sofa_tasks = args.num_sofa_tasks_hirid
    elif args.selected_data == "P12":
        if args.window == 24:
            num_binary_tasks = 4
        elif args.window == 48:
            num_binary_tasks = 1 if getattr(args, "single_task", None) else 3
        num_sofa_tasks = 0
    elif args.selected_data == "eicu":
        num_binary_tasks = args.num_binary_tasks_eicu
        num_sofa_tasks = args.num_sofa_tasks
    elif args.window == 0:   
        num_binary_tasks = 2

    else:
        num_binary_tasks = args.num_binary_tasks
        num_sofa_tasks = args.num_sofa_tasks
        
    if args.inference_mode:
        logging.info("Inference mode")

        test_loss, test_multitask_metrics, test_multiclass_metrics, test_phenotype_metrics = test(
            device, model, test_loader, accelerator,
            multitask_labels, multiclass_labels, multilabel_labels,
            binary_criterion, multiclass_criterion, multilabel_criterion, args, needed_binary_idx
        )

        logging.info("[Inference Mode] Test finished.")
        return  # 조기 종료
    
    logging.info("Start training...")
    check_patience = 0
    best_auroc = 0.0
    best_auprc = 0.0
    best_epoch = 0

    for epoch in tqdm(range(start_epoch, epochs), desc='Epochs', total=epochs-start_epoch, smoothing=0.1):
        if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
            data_loader.sampler.set_epoch(epoch)

        model.train()
        if getattr(args, "tuning_mode", "finetune") == "linearprobing":
            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "embedding"):
                unwrapped_model.embedding.eval()
            if hasattr(unwrapped_model, "encoder"):
                unwrapped_model.encoder.eval()
            if hasattr(unwrapped_model, "shared_pool"):
                unwrapped_model.shared_pool.eval()
        total_loss = 0
        binary_task_losses = [[] for _ in range(num_binary_tasks)]
        phenotype_losses = []

        all_preds, all_labels = [], []
        all_phenotype_preds, all_phenotype_labels = [], []
        
        if args.window != "entire":
            multiclass_task_losses = [[] for _ in range(num_sofa_tasks)]
            all_sofa_preds = [[] for _ in range(num_sofa_tasks)]
            all_sofa_labels = [[] for _ in range(num_sofa_tasks)]

        for step, batch in tqdm(enumerate(data_loader), desc="Steps", total=len(data_loader), leave=False):

            with accelerator.accumulate(model):
                start_time = time.time()

                t0 = time.time()
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                
                if args.selected_data == "hirid" or args.selected_data == "P12":
                    (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
                        token_type_ids,
                        task_ids, labels) = batch

                else:
                    (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
                    token_type_ids,
                    task_ids, labels, multi_labels) = batch
                data_time = time.time() - t0
                t1 = time.time()
                with accelerator.autocast():
                    losses = []
                    outputs = model(
                        input_ids=input_ids,
                        value_ids=value_ids,
                        unit_ids=unit_ids,
                        time_ids=offset_ids,
                        token_type_ids=token_type_ids,
                        age_ids=age_ids,
                        gender_ids=gender_ids,
                        task_token=task_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
                    
                    if outputs['binary_logits'] is not None:
                        logits = outputs['binary_logits'].squeeze(1)
                        if torch.isnan(logits).any():
                            print("[Error] NaN detected in binary_logits")
                    
                        binary_labels = labels[:, :num_binary_tasks].float()
                        selected_logits = logits

                        binary_losses = []
                        
                        for i in range(len(needed_binary_idx)):
                            loss_i = binary_criterion(selected_logits[:, i], binary_labels[:, i])
                            binary_losses.append(loss_i)
                            binary_task_losses[i].append(loss_i.item())
                            
                        all_preds.append(torch.sigmoid(selected_logits).detach().cpu())
                        all_labels.append(binary_labels.detach().cpu())
                        # all_preds.append(torch.sigmoid(selected_logits).detach())
                        # all_labels.append(binary_labels.detach())
                    
                        binary_loss = sum(binary_losses)
                        losses.append(binary_loss)

                    
                    if outputs['sofa_logits'] is not None and args.window != "entire":
                        sofa_logits = outputs['sofa_logits']                
                        sofa_labels = labels[:, num_binary_tasks:num_binary_tasks+num_sofa_tasks].long()
                           
                        multiclass_losses = []
                        for i in range(num_sofa_tasks):
                            pred_i = sofa_logits[:, i, :]
                            label_i = sofa_labels[:, i]
                            loss_i = multiclass_criterion(pred_i, label_i)
                            multiclass_losses.append(loss_i)
                            multiclass_task_losses[i].append(loss_i.item())
                            all_sofa_preds[i].append(torch.softmax(pred_i, dim=-1).detach().cpu())
                            all_sofa_labels[i].append(label_i.detach().cpu())
                            # all_sofa_preds[i].append(torch.softmax(pred_i, dim=-1).detach())
                            # all_sofa_labels[i].append(label_i.detach())

                        multiclass_loss = sum(multiclass_losses)
                        losses.append(multiclass_loss)
                    
                    if args.selected_data != "hirid" and args.selected_data != "P12":
                        if outputs['phenotype_logits'] is not None:
                            phenotype_logits = outputs['phenotype_logits'].squeeze(1)                    
                            phenotype_labels = multi_labels.float()
                            
                            phenotype_loss = multilabel_criterion(phenotype_logits, phenotype_labels)
                            phenotype_losses.append(phenotype_loss.item())
                            # loss += phenotype_loss          
                            all_phenotype_preds.append(torch.sigmoid(phenotype_logits).detach().cpu())
                            all_phenotype_labels.append(phenotype_labels.detach().cpu())
                            # all_phenotype_preds.append(torch.sigmoid(phenotype_logits).detach())
                            # all_phenotype_labels.append(phenotype_labels.detach())
                            losses.append(phenotype_loss)
                            
                    if len(losses) > 0:
                        loss = torch.stack(losses).sum()
                    else:
                        loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)
                            
                forward_time = time.time() - t1

                # ---------------- Backward ----------------
                t2 = time.time()
                accelerator.backward(loss)
                backward_time = time.time() - t2

                # ---------------- Optimizer + Scheduler ----------------
                t3 = time.time()
        
          
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if not isinstance(
                        scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                    ):
                        scheduler.step()
                    optimizer.zero_grad()
                
                optim_time = time.time() - t3

                step_time = time.time() - start_time

                if step % 50 == 0 and accelerator.is_local_main_process:
                    print(f"[Step {step}] "
                        f"Data: {data_time:.3f}s | Forward: {forward_time:.3f}s | "
                        f"Backward: {backward_time:.3f}s | Optim: {optim_time:.3f}s | "
                        f"Total: {step_time:.3f}s")

                total_loss += loss.item()

                # all_sofa_preds.append(torch.softmax(sofa_logits, dim=-1).detach().cpu())
                # all_sofa_labels.append(sofa_labels.detach().cpu())
                
                if step != 0 and step % 100 == 0 and accelerator.is_local_main_process:
                    log_data = {"Step": step+1, "Step Loss": loss.item()}
                    if outputs['binary_logits'] is not None:
                        batch_binary_preds = torch.sigmoid(selected_logits).detach().cpu().numpy()
                        # batch_binary_preds = torch.sigmoid(selected_logits).detach()
                        batch_binary_labels = binary_labels.detach().cpu().numpy()
                        # batch_binary_labels = binary_labels.detach()
                        

                        taskwise_metrics = calculate_binary_metrics(
                            batch_binary_preds, batch_binary_labels, num_tasks=len(needed_binary_idx))
                        for i, metrics in enumerate(taskwise_metrics):
                            log_data.update({
                                f"Step Train {binary_tasks[i]} AUROC": metrics['auroc'],
                                f"Step Train {binary_tasks[i]} AUPRC": metrics['auprc'],
                            })
                    else:
                        taskwise_metrics = None
                    
                    if args.window != "entire" and args.selected_data != "P12":
                        if outputs['sofa_logits'] is not None:
                            batch_sofa_preds = [torch.softmax(sofa_logits[:, i, :], dim=-1).detach().cpu()
                                                for i in range(num_sofa_tasks)]
                            batch_sofa_labels = [sofa_labels[:, i].detach().cpu()
                                                for i in range(num_sofa_tasks)]
                            # batch_sofa_preds = [torch.softmax(sofa_logits[:, i, :], dim=-1).detach()
                            #                     for i in range(num_sofa_tasks)]
                            # batch_sofa_labels = [sofa_labels[:, i].detach()
                            #                     for i in range(num_sofa_tasks)]
                            sofa_metrics = calculate_multiclass_metrics(
                                                batch_sofa_preds, batch_sofa_labels,
                                                num_tasks=num_sofa_tasks, num_classes=args.num_multiclass_labels)
                            for i, metrics in enumerate(sofa_metrics):
                                log_data.update({
                                    f"Step Train {multiclass_labels[i]} Macro AUROC": metrics['macro_auroc'],
                                    f"Step Train {multiclass_labels[i]} Macro AUPRC": metrics['macro_auprc'],
                                    f"Step Train {multiclass_labels[i]} Micro AUROC": metrics['micro_auroc'],
                                    f"Step Train {multiclass_labels[i]} Micro AUPRC": metrics['micro_auprc'],
                                })
                        else:
                            sofa_metrics = None
                            
                    if args.selected_data != "hirid" and args.selected_data != "P12":
                        if outputs['phenotype_logits'] is not None:
                            batch_pheno_preds = torch.sigmoid(phenotype_logits).detach().cpu().numpy()
                            batch_pheno_labels = phenotype_labels.detach().cpu().numpy()
                            # batch_pheno_preds = torch.sigmoid(phenotype_logits).detach()
                            # batch_pheno_labels = phenotype_labels.detach()

                            phenotype_metrics = calculate_phenotype_metrics(
                                batch_pheno_preds, batch_pheno_labels)
                            log_data.update({
                                "Step Train Phenotype Macro AUROC": phenotype_metrics['macro_auroc'],
                                "Step Train Phenotype Macro AUPRC": phenotype_metrics['macro_auprc'],
                                "Step Train Phenotype Micro AUROC": phenotype_metrics['micro_auroc'],
                                "Step Train Phenotype Micro AUPRC": phenotype_metrics['micro_auprc'],
                            })
                        else:
                            phenotype_metrics = None
                    
                    for k, v in log_data.items():
                        print(f"{k}: {v}")
                    wandb.log(log_data)

        epoch_loss = total_loss / (step + 1)
        
        # binary_preds = torch.cat(all_preds).numpy()
        # binary_labels = torch.cat(all_labels).numpy()
        

        binary_preds = accelerator.gather(torch.cat(all_preds).to(device)).cpu().numpy()
        binary_labels = accelerator.gather(torch.cat(all_labels).to(device)).cpu().numpy()
        taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=len(needed_binary_idx))
        
        
        
        # binary_preds = accelerator.gather(torch.cat(all_preds).to(device))
        # binary_labels = accelerator.gather(torch.cat(all_labels).to(device))
        # taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=len(needed_binary_idx))
        
        if args.window != "entire" and args.selected_data != "P12":
            # sofa_preds = [torch.cat(p).numpy() for p in all_sofa_preds]
            # sofa_labels = [torch.cat(l).numpy() for l in all_sofa_labels]
            sofa_preds = [accelerator.gather(torch.cat(p).to(device)).cpu().numpy() for p in all_sofa_preds]
            sofa_labels = [accelerator.gather(torch.cat(l).to(device)).cpu().numpy() for l in all_sofa_labels]
            sofa_metrics = calculate_multiclass_metrics(sofa_preds, sofa_labels,
                                                        num_tasks=num_sofa_tasks,
                                                        num_classes=args.num_multiclass_labels)
        if args.selected_data != "hirid" and args.selected_data != "P12":
            # phenotype_preds = torch.cat(all_phenotype_preds).numpy()
            # phenotype_labels = torch.cat(all_phenotype_labels).numpy()
            phenotype_preds = accelerator.gather(torch.cat(all_phenotype_preds).to(device)).cpu().numpy()
            phenotype_labels = accelerator.gather(torch.cat(all_phenotype_labels).to(device)).cpu().numpy()
            phenotype_metrics = calculate_phenotype_metrics(phenotype_preds, phenotype_labels)
        
        # if args.window != "entire":
        #     sofa_preds = [accelerator.gather(torch.cat(p).to(device)) for p in all_sofa_preds]
        #     sofa_labels = [accelerator.gather(torch.cat(l).to(device)) for l in all_sofa_labels]
        #     sofa_metrics = calculate_multiclass_metrics(sofa_preds, sofa_labels,
        #                                                 num_tasks=num_sofa_tasks,
        #                                                 num_classes=args.num_multiclass_labels)

        # if args.selected_data != "hirid":
        #     phenotype_preds = accelerator.gather(torch.cat(all_phenotype_preds).to(device))
        #     phenotype_labels = accelerator.gather(torch.cat(all_phenotype_labels).to(device))
        #     phenotype_metrics = calculate_phenotype_metrics(phenotype_preds, phenotype_labels)

        
        if accelerator.is_local_main_process:
            log_data = {"Epoch": epoch + 1, "Train Loss": epoch_loss}
            
            for i, metrics in enumerate(taskwise_metrics):
                log_data.update({
                    f"Train {binary_tasks[i]} AUROC": metrics['auroc'],
                    f"Train {binary_tasks[i]} AUPRC": metrics['auprc'],
                    f"Train {binary_tasks[i]} Loss": np.mean(binary_task_losses[i]),
                })
                
            if args.window != "entire" and args.selected_data != "P12":
                for i, metrics in enumerate(sofa_metrics):
                    log_data.update({
                        f"Train {multiclass_labels[i]} Macro AUROC": metrics['macro_auroc'],
                        f"Train {multiclass_labels[i]} Macro AUPRC": metrics['macro_auprc'],
                        f"Train {multiclass_labels[i]} Micro AUROC": metrics['micro_auroc'],
                        f"Train {multiclass_labels[i]} Micro AUPRC": metrics['micro_auprc'],    
                        f"Train {multiclass_labels[i]} Loss": np.mean(multiclass_task_losses[i]),
                    })
            if args.selected_data != "hirid" and args.selected_data != "P12":
                log_data.update({
                    "Train Phenotype Macro AUROC": phenotype_metrics['macro_auroc'],
                    "Train Phenotype Macro AUPRC": phenotype_metrics['macro_auprc'],
                    "Train Phenotype Micro AUROC": phenotype_metrics['micro_auroc'],
                    "Train Phenotype Micro AUPRC": phenotype_metrics['micro_auprc'],
                    "Train Phenotype Loss": np.mean(phenotype_losses),
                })
            wandb.log(log_data)
            for k, v in log_data.items():
                print(f"{k}: {v}")

        valid_loss, valid_multitask_metrics, valid_multiclass_metrics, valid_phenotype_metrics = validation(
            device, model, val_loader, accelerator,
            multitask_labels, multiclass_labels, multilabel_labels, binary_criterion, multiclass_criterion, multilabel_criterion, epoch + 1, args, needed_binary_idx)

        
        if args.selected_data == "hirid":
            mean_auroc = np.nanmean([m['auroc'] for m in valid_multitask_metrics] + [m['macro_auroc'] for m in valid_multiclass_metrics])
            mean_auprc = np.nanmean([m['auprc'] for m in valid_multitask_metrics] + [m['macro_auprc'] for m in valid_multiclass_metrics])
            
        elif args.selected_data == "P12":
            mean_auroc = np.nanmean([m['auroc'] for m in valid_multitask_metrics])
            mean_auprc = np.nanmean([m['auprc'] for m in valid_multitask_metrics])
                             
        elif args.window == "entire":
            mean_auroc = np.nanmean([m['auroc'] for m in valid_multitask_metrics] + [valid_phenotype_metrics['macro_auroc']])
            mean_auprc = np.nanmean([m['auroc'] for m in valid_multitask_metrics] + [valid_phenotype_metrics['macro_auprc']])    
        else:   
            mean_auroc = np.nanmean([m['auroc'] for m in valid_multitask_metrics] + [m['macro_auroc'] for m in valid_multiclass_metrics] + [valid_phenotype_metrics['macro_auroc']])
            mean_auprc = np.nanmean([m['auprc'] for m in valid_multitask_metrics] + [m['macro_auprc'] for m in valid_multiclass_metrics] + [valid_phenotype_metrics['macro_auprc']])

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(mean_auprc)

        if mean_auprc > best_auprc:
            best_auprc = mean_auprc
            best_epoch = epoch + 1
            check_patience = 0

            if accelerator.is_local_main_process:
                wandb.log({
                    "Best Valid Epoch": best_epoch,
                    "Best Valid Loss": valid_loss,
                    "Best Valid AUROC": mean_auroc,
                    "Best Valid AUPRC": mean_auprc,
                })
                output_path = Path(save_path) / f"best_{args.exp_name}.pth"
                accelerator.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, output_path)
                
                if use_lora:
                    unwrapped = accelerator.unwrap_model(model)
                    
                    try:
                        unwrapped.encoder.save_pretrained(str(adapter_path))
                        logging.info(f"[LoRA] Saved adapter to {adapter_path}")
                    except Exception as e:
                        logging.error(f"[LoRA] Adapter save failed: {e}")
                        
                    try:
                        full_state = unwrapped.state_dict()
                        bundle = {k: v.cpu() for k, v in full_state.items()
                                  if ("lora_" not in k) and (not k.startswith("encoder."))}
                        head_bundle_path = Path(save_path) / f"{args.exp_name}_head_bundle.pth"
                        torch.save({"model_state_dict": bundle}, head_bundle_path)
                        logging.info(f"[LoRA] Saved head bundle to {head_bundle_path}")
                    except Exception as e:
                        logging.error(f"[LoRA] Head bundle save failed: {e}")
        else:
            check_patience += 1

        if check_patience >= patience:
            print(f"Early Stopping triggered after {epoch + 1} epochs.")
            break
        
    if accelerator.is_local_main_process:
        print(f"Loading best model from epoch {best_epoch}")
    
    best_model_path = Path(save_path) / f"best_{args.exp_name}.pth"
    
    accelerator.wait_for_everyone()
    
    if use_lora:
        unwrapped = accelerator.unwrap_model(model)
        best_bundle_path = Path(save_path) / f"{args.exp_name}_head_bundle.pth"
        try:
            if getattr(args, "use_lora", False) and getattr(args, "pretrain_path", None):
                base_path = Path(args.save_path) / f"{args.pretrain_path}"
                base = torch.load(base_path, map_location=device)
                base_state = base["model_state_dict"] if "model_state_dict" in base else base
                enc_only = {k: v for k, v in base_state.items() if k.startswith("encoder.")}
                missing, unexpected = model.load_state_dict(enc_only, strict=False)
                logging.info(f"[LoRA/Infer] Loaded encoder from pretrain "
                            f"(missing={len(missing)}, unexpected={len(unexpected)})")
                print(f"[LoRA/Infer] Loaded encoder from pretrain "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})")
            else:
                ckpt_best = torch.load(best_model_path, map_location=device)
                st_best = ckpt_best["model_state_dict"] if "model_state_dict" in ckpt_best else ckpt_best
                enc_only = {k: v for k, v in st_best.items() if k.startswith("encoder.")}
                missing, unexpected = model.load_state_dict(enc_only, strict=False)
                logging.info(f"[LoRA/Infer] Loaded encoder from best ckpt "
                            f"(missing={len(missing)}, unexpected={len(unexpected)})")
                print(f"[LoRA/Infer] Loaded encoder from best ckpt "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})")
        except Exception as e:
            logging.error(f"[LoRA] Base encoder load failed: {e}")
            raise
        
        try:
            unwrapped.encoder.load_adapter(str(adapter_path), adapter_name=adapter_name, is_trainable=False)
            unwrapped.encoder.set_adapter(adapter_name)
            logging.info(f"[LoRA] Loaded & set adapter '{adapter_name}' from {adapter_path}")
            print(f"[LoRA] Loaded & set adapter '{adapter_name}' from {adapter_path}")
        except Exception as e:
            logging.error(f"[LoRA] Adapter load failed: {e}")
            raise
        
        try:
            if best_bundle_path.exists():
                bundle = torch.load(best_bundle_path, map_location=device)
                bstate = bundle["model_state_dict"] if "model_state_dict" in bundle else bundle
                missing, unexpected = model.load_state_dict(bstate, strict=False)
                logging.info(f"[LoRA] Loaded head/pool bundle "
                             f"(missing={len(missing)}, unexpected={len(unexpected)})")
                print(f"[LoRA] Loaded head/pool bundle "
                      f"(missing={len(missing)}, unexpected={len(unexpected)})")
            else:
                ckpt = torch.load(best_model_path, map_location=device)
                state = ckpt['model_state_dict']
                filtered = {k: v for k, v in state.items()
                            if ("lora_" not in k) and (not k.startswith("encoder."))}
                missing, unexpected = model.load_state_dict(filtered, strict=False)
                logging.info(f"[LoRA] Loaded non-encoder & non-LoRA weights from best ckpt "
                             f"(missing={len(missing)}, unexpected={len(unexpected)})")
        except Exception as e:
            logging.error(f"[LoRA] Non-encoder/Non-LoRA weight load failed: {e}")
            raise
        
    else:
        print("full finetuning weight loaded")
        model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'])
        
        
    test_loss, test_multitask_metrics, test_multiclass_metrics, test_phenotype_metrics = test(
            device, model, test_loader, accelerator,
            multitask_labels, multiclass_labels, multilabel_labels,
            binary_criterion, multiclass_criterion, multilabel_criterion, args, needed_binary_idx
        )
      
def validation(
    device: torch.device,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    multitask_labels: list,
    multiclass_labels: list,
    multilabel_labels: list,
    binary_criterion: torch.nn,
    multiclass_criterion: torch.nn,
    multilabel_criterion: torch.nn,
    epoch: int,
    args: dict,
    needed_binary_idx: list
):
    
    binary_tasks = [multitask_labels[i] for i in needed_binary_idx]
    
    if args.selected_data == "hirid":
        num_binary_tasks = args.num_binary_tasks_hirid
        num_sofa_tasks = args.num_sofa_tasks_hirid
        
    elif args.selected_data == "P12":
        if args.window == 24:
            num_binary_tasks = 4
        elif args.window == 48:
            num_binary_tasks = 1 if getattr(args, "single_task", None) else 3
        num_sofa_tasks = 0

    elif args.selected_data == "eicu":
        num_binary_tasks = args.num_binary_tasks_eicu
        num_sofa_tasks = args.num_sofa_tasks

    elif args.window == "entire":
        num_binary_tasks = 2

    else:
        num_binary_tasks = args.num_binary_tasks
        num_sofa_tasks = args.num_sofa_tasks

    logging.info("Start validation...")
    model.eval()
    total_loss = 0

    binary_task_losses = [[] for _ in range(num_binary_tasks)]
    phenotype_losses = []

    all_preds, all_labels = [], []
    all_phenotype_preds, all_phenotype_labels = [], []
    
    if args.window != "entire":
        multiclass_task_losses = [[] for _ in range(num_sofa_tasks)]
        all_sofa_preds = [[] for _ in range(num_sofa_tasks)]
        all_sofa_labels = [[] for _ in range(num_sofa_tasks)]

    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), desc="Validation", total=len(data_loader)):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            if args.selected_data == "hirid" or args.selected_data == "P12":
                (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
                    token_type_ids,
                    task_ids, labels) = batch
                
            else:
                (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
                token_type_ids,
                task_ids, labels, multi_labels) = batch

            losses = []
            outputs = model(
                input_ids=input_ids,
                value_ids=value_ids,
                unit_ids=unit_ids,
                time_ids=offset_ids,
                token_type_ids=token_type_ids,
                age_ids=age_ids,
                gender_ids=gender_ids,
                task_token=task_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            if outputs['binary_logits'] is not None:
                logits = outputs['binary_logits'].squeeze(1)
                binary_labels = labels[:, :num_binary_tasks].float()
                selected_logits = logits
                
                binary_losses = []
                
                for i in range(len(needed_binary_idx)):
                    loss_i = binary_criterion(selected_logits[:, i], binary_labels[:, i])
                    binary_losses.append(loss_i)
                    binary_task_losses[i].append(loss_i.item())
                    
                all_preds.append(torch.sigmoid(selected_logits).detach().cpu())
                all_labels.append(binary_labels.detach().cpu())
                
                binary_loss = sum(binary_losses)

                losses.append(binary_loss)
            
            if outputs['sofa_logits'] is not None and args.window != "entire" and args.selected_data != "P12":
                sofa_logits = outputs['sofa_logits']                
                sofa_labels = labels[:, num_binary_tasks:num_binary_tasks+num_sofa_tasks].long()

                multiclass_losses = []
                for i in range(num_sofa_tasks):
                    pred_i = sofa_logits[:, i, :]
                    label_i = sofa_labels[:, i]
                    loss_i = multiclass_criterion(pred_i, label_i)
                    multiclass_losses.append(loss_i)
                    multiclass_task_losses[i].append(loss_i.item())
                    all_sofa_preds[i].append(torch.softmax(pred_i, dim=-1).detach().cpu())
                    all_sofa_labels[i].append(label_i.detach().cpu())
                multiclass_loss = sum(multiclass_losses)
                losses.append(multiclass_loss)
                
            if args.selected_data != "hirid" and args.selected_data != "P12":
                if outputs['phenotype_logits'] is not None:
                    phenotype_logits = outputs['phenotype_logits'].squeeze(1)
                    phenotype_labels = multi_labels.float()
                
                    phenotype_loss = multilabel_criterion(phenotype_logits, phenotype_labels)
                    phenotype_losses.append(phenotype_loss.item())
                    
                    all_phenotype_preds.append(torch.sigmoid(phenotype_logits).detach().cpu())
                    all_phenotype_labels.append(phenotype_labels.detach().cpu())
                    losses.append(phenotype_loss)
            # if args.window == "entire":
            #     loss = binary_loss + phenotype_loss
            # elif args.selected_data == "hirid":
            #     loss = binary_loss + multiclass_loss
            # else:
            #     loss = binary_loss + multiclass_loss + phenotype_loss
            
            total_loss += sum(losses).item()

    
    
    epoch_loss = total_loss / len(data_loader)
    
    
    # binary_preds = accelerator.gather(torch.cat(all_preds).to(device)).numpy()
    # binary_labels = accelerator.gather(torch.cat(all_labels).to(device)).numpy()
    # taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=len(needed_binary_idx))
    
    # if args.window != "entire":
    #     sofa_preds = [accelerator.gather(torch.cat(p).to(device)).numpy() for p in all_sofa_preds]
    #     sofa_labels = [accelerator.gather(torch.cat(l).to(device)).numpy() for l in all_sofa_labels]
    #     sofa_metrics = calculate_multiclass_metrics(sofa_preds, sofa_labels,
    #                                                 num_tasks=num_sofa_tasks,
    #                                                 num_classes=args.num_multiclass_labels)
    
    # if args.selected_data != "hirid":
    #     phenotype_preds = accelerator.gather(torch.cat(all_phenotype_preds).to(device)).numpy()
    #     phenotype_labels = accelerator.gather(torch.cat(all_phenotype_labels).to(device)).numpy()
    #     phenotype_metrics = calculate_phenotype_metrics(phenotype_preds, phenotype_labels)
    binary_preds = accelerator.gather(torch.cat(all_preds).to(device)).cpu().numpy()
    binary_labels = accelerator.gather(torch.cat(all_labels).to(device)).cpu().numpy()
    taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=len(needed_binary_idx))

    if args.window != "entire" and args.selected_data != "P12":
        sofa_preds = [accelerator.gather(torch.cat(p).to(device)).cpu().numpy() for p in all_sofa_preds]
        sofa_labels = [accelerator.gather(torch.cat(l).to(device)).cpu().numpy() for l in all_sofa_labels]
        sofa_metrics = calculate_multiclass_metrics(sofa_preds, sofa_labels,
                                                    num_tasks=num_sofa_tasks,
                                                    num_classes=args.num_multiclass_labels)
    if args.selected_data != "hirid" and args.selected_data != "P12":
        phenotype_preds = accelerator.gather(torch.cat(all_phenotype_preds).to(device)).cpu().numpy()
        phenotype_labels = accelerator.gather(torch.cat(all_phenotype_labels).to(device)).cpu().numpy()
        phenotype_metrics = calculate_phenotype_metrics(phenotype_preds, phenotype_labels)


    if accelerator.is_local_main_process:
        log_data = {"Epoch": epoch, "Validation Loss": epoch_loss}
        
        for i, metrics in enumerate(taskwise_metrics):
            log_data.update({
                f"Validation {binary_tasks[i]} AUROC": metrics['auroc'],
                f"Validation {binary_tasks[i]} AUPRC": metrics['auprc'],
                f"Validation {binary_tasks[i]} Loss": np.mean(binary_task_losses[i]),
            })
            
        if args.window != "entire" and args.selected_data != "P12":
            for i, metrics in enumerate(sofa_metrics):
                log_data.update({
                    f"Validation {multiclass_labels[i]} Macro AUROC": metrics['macro_auroc'],
                    f"Validation {multiclass_labels[i]} Macro AUPRC": metrics['macro_auprc'],
                    f"Validation {multiclass_labels[i]} Micro AUROC": metrics['micro_auroc'],
                    f"Validation {multiclass_labels[i]} Micro AUPRC": metrics['micro_auprc'],
                    f"Validation {multiclass_labels[i]} Loss": np.mean(multiclass_task_losses[i]),
                })
                
        if args.selected_data != "hirid" and args.selected_data != "P12":
            log_data.update({
                "Validation Phenotype Macro AUROC": phenotype_metrics['macro_auroc'],
                "Validation Phenotype Macro AUPRC": phenotype_metrics['macro_auprc'],
                "Validation Phenotype Micro AUROC": phenotype_metrics['micro_auroc'],
                "Validation Phenotype Micro AUPRC": phenotype_metrics['micro_auprc'],
                "Validation Phenotype Loss": np.mean(phenotype_losses),
            })
            for i, auroc in enumerate(phenotype_metrics['per_task_auroc']):
                log_data[f"Validation Phenotype {multilabel_labels[i]} AUROC"] = auroc
            for i, auprc in enumerate(phenotype_metrics['per_task_auprc']):
                log_data[f"Validation Phenotype {multilabel_labels[i]} AUPRC"] = auprc

        for k, v in log_data.items():
            print(f"{k}: {v}")
        wandb.log(log_data)

    if args.selected_data != "hirid" and args.window != "entire" and args.selected_data != "P12":
        return epoch_loss, taskwise_metrics, sofa_metrics, phenotype_metrics
    elif args.window == "entire":
        return epoch_loss, taskwise_metrics, None, phenotype_metrics
    elif args.selected_data == "P12":
        return epoch_loss, taskwise_metrics, None, None
    else:
        return epoch_loss, taskwise_metrics, sofa_metrics, None

def test(
    device: torch.device,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    multitask_labels: list,
    multiclass_labels: list,
    multilabel_labels: list,
    binary_criterion: torch.nn,
    multiclass_criterion: torch.nn,
    multilabel_criterion: torch.nn,
    args: dict,
    needed_binary_idx: list
):
    
    binary_tasks = [multitask_labels[i] for i in needed_binary_idx]
    
    if args.selected_data == "hirid":
        num_binary_tasks = args.num_binary_tasks_hirid
        num_sofa_tasks = args.num_sofa_tasks_hirid
    elif args.selected_data == "P12":
        if args.window == 24:
            num_binary_tasks = 4
        elif args.window == 48:
            num_binary_tasks = 1 if getattr(args, "single_task", None) else 3
        num_sofa_tasks = 0
    elif args.selected_data == "eicu":
        num_binary_tasks = args.num_binary_tasks_eicu
        num_sofa_tasks = args.num_sofa_tasks
    elif args.window == "entire":
        num_binary_tasks = 2

    else:
        num_binary_tasks = args.num_binary_tasks
        num_sofa_tasks = args.num_sofa_tasks

    logging.info("Start test...")
    model.eval()
    total_loss = 0
    
    binary_task_losses = [[] for _ in range(num_binary_tasks)]
    phenotype_losses = []

    all_preds, all_labels = [], []
    all_phenotype_preds, all_phenotype_labels = [], []
    
    if args.window != "entire":
        multiclass_task_losses = [[] for _ in range(num_sofa_tasks)]
        all_sofa_preds = [[] for _ in range(num_sofa_tasks)]
        all_sofa_labels = [[] for _ in range(num_sofa_tasks)]
    
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), desc="Test", total=len(data_loader)):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            if args.selected_data == "hirid" or args.selected_data == "P12":
                (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
                    token_type_ids,
                    task_ids, labels) = batch
                
            else:
                (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
                token_type_ids,
                task_ids, labels, multi_labels) = batch

            losses = []
            outputs = model(
                input_ids=input_ids,
                value_ids=value_ids,
                unit_ids=unit_ids,
                time_ids=offset_ids,
                token_type_ids=token_type_ids,
                age_ids=age_ids,
                gender_ids=gender_ids,
                task_token=task_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            
            if outputs['binary_logits'] is not None:
                logits = outputs['binary_logits'].squeeze(1)
                binary_labels = labels[:, :num_binary_tasks].float()
                selected_logits = logits
                
                binary_losses = []
                
                for i in range(len(needed_binary_idx)):
                    loss_i = binary_criterion(selected_logits[:, i], binary_labels[:, i])
                    binary_losses.append(loss_i)
                    binary_task_losses[i].append(loss_i.item())
                    
                all_preds.append(torch.sigmoid(selected_logits).detach().cpu())
                all_labels.append(binary_labels.detach().cpu())
                
                binary_loss = sum(binary_losses)
                losses.append(binary_loss)
                
            if outputs['sofa_logits'] is not None and args.window != "entire" and args.selected_data != "P12":
                sofa_logits = outputs['sofa_logits']                
                sofa_labels = labels[:, num_binary_tasks:num_binary_tasks+num_sofa_tasks].long()

                multiclass_losses = []
                for i in range(num_sofa_tasks):
                    pred_i = sofa_logits[:, i, :]
                    label_i = sofa_labels[:, i]
                    loss_i = multiclass_criterion(pred_i, label_i)
                    multiclass_losses.append(loss_i)
                    multiclass_task_losses[i].append(loss_i.item())
                    all_sofa_preds[i].append(torch.softmax(pred_i, dim=-1).detach().cpu())
                    all_sofa_labels[i].append(label_i.detach().cpu())
                multiclass_loss = sum(multiclass_losses)
                losses.append(multiclass_loss)
            
            if args.selected_data != "hirid" and args.selected_data != "P12":
                if outputs['phenotype_logits'] is not None:
                    phenotype_logits = outputs['phenotype_logits'].squeeze(1)
                    phenotype_labels = multi_labels.float()
                
                    phenotype_loss = multilabel_criterion(phenotype_logits, phenotype_labels)
                    phenotype_losses.append(phenotype_loss.item())
                    
                    all_phenotype_preds.append(torch.sigmoid(phenotype_logits).detach().cpu())
                    all_phenotype_labels.append(phenotype_labels.detach().cpu())
                    losses.append(phenotype_loss)
                    
            # if args.window == "entire":
            #     loss = binary_loss + phenotype_loss
            # elif args.selected_data == "hirid":
            #     loss = binary_loss + multiclass_loss
            # else:
            #     loss = binary_loss + multiclass_loss + phenotype_loss

            total_loss += sum(losses).item()

    epoch_loss = total_loss / len(data_loader)
           
    binary_preds = accelerator.gather(torch.cat(all_preds).to(device)).cpu().numpy()
    binary_labels = accelerator.gather(torch.cat(all_labels).to(device)).cpu().numpy()
    taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=len(needed_binary_idx))

    if args.window != "entire" and args.selected_data != "P12":
        sofa_preds = [accelerator.gather(torch.cat(p).to(device)).cpu().numpy() for p in all_sofa_preds]
        sofa_labels = [accelerator.gather(torch.cat(l).to(device)).cpu().numpy() for l in all_sofa_labels]
        sofa_metrics = calculate_multiclass_metrics(sofa_preds, sofa_labels,
                                                    num_tasks=num_sofa_tasks,
                                                    num_classes=args.num_multiclass_labels)
    if args.selected_data != "hirid" and args.selected_data != "P12":
        phenotype_preds = accelerator.gather(torch.cat(all_phenotype_preds).to(device)).cpu().numpy()
        phenotype_labels = accelerator.gather(torch.cat(all_phenotype_labels).to(device)).cpu().numpy()
        phenotype_metrics = calculate_phenotype_metrics(phenotype_preds, phenotype_labels)

    # binary_preds = accelerator.gather(torch.cat(all_preds).to(device))
    # binary_labels = accelerator.gather(torch.cat(all_labels).to(device))
    # taskwise_metrics = calculate_binary_metrics(binary_preds, binary_labels, num_tasks=len(needed_binary_idx))

    # if args.window != "entire":
    #     sofa_preds = [accelerator.gather(torch.cat(p).to(device)) for p in all_sofa_preds]
    #     sofa_labels = [accelerator.gather(torch.cat(l).to(device)) for l in all_sofa_labels]
    #     sofa_metrics = calculate_multiclass_metrics(sofa_preds, sofa_labels,
    #                                                 num_tasks=num_sofa_tasks,
    #                                                 num_classes=args.num_multiclass_labels)

    # if args.selected_data != "hirid":
    #     phenotype_preds = accelerator.gather(torch.cat(all_phenotype_preds).to(device))
    #     phenotype_labels = accelerator.gather(torch.cat(all_phenotype_labels).to(device))
    #     phenotype_metrics = calculate_phenotype_metrics(phenotype_preds, phenotype_labels)

        
    if accelerator.is_local_main_process:
        log_data = {"Test Loss": epoch_loss}

        for i, metrics in enumerate(taskwise_metrics):
            log_data.update({
                f"Test {binary_tasks[i]} AUROC": metrics['auroc'],
                f"Test {binary_tasks[i]} AUPRC": metrics['auprc'],
                f"Test {binary_tasks[i]} Loss": np.mean(binary_task_losses[i]),
            })
        
        if args.window != "entire" and args.selected_data != "P12":
            for i, metrics in enumerate(sofa_metrics):
                log_data.update({
                    f"Test {multiclass_labels[i]} Macro AUROC": metrics['macro_auroc'],
                    f"Test {multiclass_labels[i]} Macro AUPRC": metrics['macro_auprc'],
                    f"Test {multiclass_labels[i]} Micro AUROC": metrics['micro_auroc'],
                    f"Test {multiclass_labels[i]} Micro AUPRC": metrics['micro_auprc'],
                    f"Test {multiclass_labels[i]} Loss": np.mean(multiclass_task_losses[i]),
                })

        if args.selected_data != "hirid" and args.selected_data != "P12":
            log_data.update({
                "Test Phenotype Macro AUROC": phenotype_metrics['macro_auroc'],
                "Test Phenotype Macro AUPRC": phenotype_metrics['macro_auprc'],
                "Test Phenotype Micro AUROC": phenotype_metrics['micro_auroc'],
                "Test Phenotype Micro AUPRC": phenotype_metrics['micro_auprc'],
                "Test Phenotype Loss": np.mean(phenotype_losses),
            })

            for i, auroc in enumerate(phenotype_metrics['per_task_auroc']):
                log_data[f"Test Phenotype {multilabel_labels[i]} AUROC"] = auroc
            for i, auprc in enumerate(phenotype_metrics['per_task_auprc']):
                log_data[f"Test Phenotype {multilabel_labels[i]} AUPRC"] = auprc


        for k, v in log_data.items():
            print(f"{k}: {v}")
        wandb.log(log_data)

    if args.selected_data != "hirid" and args.window != "entire" and args.selected_data != "P12":
        return epoch_loss, taskwise_metrics, sofa_metrics, phenotype_metrics
    elif args.window == "entire":
        return epoch_loss, taskwise_metrics, None, phenotype_metrics
    elif args.selected_data == "P12":
        return epoch_loss, taskwise_metrics, None, None
    else:
        return epoch_loss, taskwise_metrics, sofa_metrics, None

  
# def test(
#     device: torch.device,
#     model: torch.nn.Module,
#     data_loader: torch.utils.data.DataLoader,
#     scaler: torch.amp.GradScaler,
#     accelerator: Accelerator,
#     binary_criterion: torch.nn,
#     threshold: float,
#     args: dict
# ):
    
#     logging.info("Start test...")
#     # val_loss = []
#     # val_precision = []  
#     # val_recall = []
#     # val_f1_score = []
#     # val_auroc = []
#     # val_auprc = []
#     test_loss = []
#     total_loss = 0  
#     total_samples = 0  

#     all_preds = []
#     all_labels = []
    
#     model.eval()
    
#     with torch.no_grad():
#         for step, batch in tqdm(enumerate(data_loader), desc="Test", total=len(data_loader)):
#             batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
#             input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, time_ids, continuous_ids, position_ids, token_type_ids, task_token, labels = batch
         
            
#             outputs = model(
#                 input_ids = input_ids,
#                 value_ids = value_ids,
#                 unit_ids = unit_ids,
#                 time_ids = time_ids,                
#                 continuous_ids = continuous_ids,
#                 position_ids = position_ids,
#                 token_type_ids = token_type_ids,
#                 age_ids = age_ids,
#                 gender_ids = gender_ids,
#                 task_token = task_token,
#                 attention_mask=attention_mask,
#                 global_attention_mask=None,
#                 labels=labels,
#                 return_dict=True,
#                 criterion=criterion,
#                 )
            
#             # loss = criterion(outputs.view(-1, args.num_labels), labels.view(-1))
#             # metrics = calculate_metrics(outputs, labels)
            
#             # val_loss.append(loss.item())
#             # val_precision.append(metrics['precision'])
#             # val_recall.append(metrics['recall'])
#             # val_f1_score.append(metrics['f1_score'])
#             # val_auroc.append(metrics['auroc'])
#             # val_auprc.append(metrics['auprc'])
            
#             # val_loss.append(loss.item())
#             # loss = criterion(outputs.view(-1, args.num_labels), labels.view(-1))
#             batch_size = labels.size(0)
#             total_loss += accelerator.gather(outputs['loss']).mean().item() * batch_size
#             total_samples += batch_size
#             # valid_loss.append(loss.item())
#             # metrics = calculate_metrics(outputs, labels)
            
#             all_preds.append(accelerator.gather(outputs).detach().numpy())
#             all_labels.append(accelerator.gather(labels).detach().cpu().numpy())
                        
            
#             if step % 30 == "entire":   
#                 if accelerator.is_local_main_process:
#                     metrics = calculate_metrics_test(outputs, labels, thresholds=threshold)
#                     print(f"Test | Step {step+1} / {len(data_loader)} | Loss: {outputs['loss'].item()} | Accuracy: {metrics['accuracy']} | Precision: {metrics['precision']} | Recall: {metrics['recall']} | F1_Score: {metrics['f1_score']} | AUROC: {metrics['auroc']} | AUPRC: {metrics['auprc']}")     
         
            
#     #     accelerator.wait_for_everyone()
        
#     #     if accelerator.is_local_main_process:
#     #         wandb.log({
#     #             "Val Loss": np.mean(val_loss),
#     #             "Val Precision": np.mean(val_precision),
#     #             "Val Recall": np.mean(val_recall),
#     #             "Val F1_score": np.mean(val_f1_score),
#     #             "Val Auroc": np.mean(val_auroc),
#     #             "Val Auprc": np.mean(val_auprc)
#     #         })
            
#     #         metrics = {
#     #             'precision': np.mean(val_precision),
#     #             'recall': np.mean(val_recall),
#     #             'f1_score': np.mean(val_f1_score),
#     #             'auroc': np.mean(val_auroc),
#     #             'auprc': np.mean(val_auprc)
#     #         }
        
#     # return np.mean(val_loss), metrics
#         epoch_loss = total_loss / total_samples
#         all_preds = torch.cat([torch.tensor(pred).to(device) for pred in all_preds])
#         all_labels = torch.cat([torch.tensor(label).to(device) for label in all_labels])
        
#         # all_preds = np.concatenate([pred for pred in all_preds])
#         # all_labels = np.concatenate([label for label in all_labels])

#         all_metrics = calculate_metrics_test(all_preds, all_labels, thresholds=threshold)
        

#         accelerator.wait_for_everyone()
#         if accelerator.is_local_main_process:
#             log_data = {
#                 "Test Loss": epoch_loss,
#                 **{f"Test {key}": value for key, value in all_metrics.items()},
#                             }
#             for k, v in log_data.items():
            # print(f"{k}: {v}")
#             wandb.log(log_data)
        
#     return epoch_loss, all_metrics

import torchmetrics
from torchmetrics.functional import auroc, average_precision

def train_phenotype(
    device: torch.device,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    multilabel_criterion: torch.nn,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    accelerator: Accelerator,
    multilabel_labels: list,
    epochs: int,
    start_epoch: int,
    patience: int,
    save_path: Path,
    args: dict,
):
    logging.info("Start phenotype-only training...")
    check_patience = 0
    best_macro_auroc = 0.0
    best_macro_auprc = 0.0
    best_epoch = 0

    num_classes = len(multilabel_labels)

    # torchmetrics 객체 정의 (macro + micro)
    def get_metrics():
        return {
            "auroc_macro": torchmetrics.AUROC(task="multilabel", num_labels=num_classes, average="macro").to(device),
            "auroc_micro": torchmetrics.AUROC(task="multilabel", num_labels=num_classes, average="micro").to(device),
            "ap_macro": torchmetrics.AveragePrecision(task="multilabel", num_labels=num_classes, average="macro").to(device),
            "ap_micro": torchmetrics.AveragePrecision(task="multilabel", num_labels=num_classes, average="micro").to(device),
            "auroc_per": torchmetrics.AUROC(task="multilabel", num_labels=num_classes, average=None).to(device),
            "ap_per": torchmetrics.AveragePrecision(task="multilabel", num_labels=num_classes, average=None).to(device),

        }

    train_metrics = get_metrics()

    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs"):
        model.train()
        if getattr(args, "tuning_mode", "finetune") == "linearprobing":
            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "embedding"):
                unwrapped_model.embedding.eval()
            if hasattr(unwrapped_model, "encoder"):
                unwrapped_model.encoder.eval()
            if hasattr(unwrapped_model, "shared_pool"):
                unwrapped_model.shared_pool.eval()
        total_loss = 0
        for m in train_metrics.values():
            m.reset()

        for step, batch in tqdm(enumerate(data_loader), desc="Steps", total=len(data_loader), leave=False):

            with accelerator.accumulate(model):
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
                 token_type_ids,
                 task_ids, multi_labels) = batch

                with accelerator.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        value_ids=value_ids,
                        unit_ids=unit_ids,
                        time_ids=offset_ids,
                        token_type_ids=token_type_ids,
                        age_ids=age_ids,
                        gender_ids=gender_ids,
                        task_token=task_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )

                    phenotype_logits = outputs['phenotype_logits'].squeeze(1)
                    phenotype_labels = multi_labels.float()

                    loss = multilabel_criterion(phenotype_logits, phenotype_labels)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if not isinstance(
                        scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                    ):
                        scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item()

                preds = torch.sigmoid(phenotype_logits).detach()
                phenotype_labels = phenotype_labels.detach().int()

                for m in train_metrics.values():
                    m.update(preds.to(accelerator.device), phenotype_labels.to(accelerator.device))

                # step 단위 logging
                if step != 0 and step % 100 == 0 and accelerator.is_local_main_process:
                    step_auroc_macro = auroc(preds.cpu(), phenotype_labels.cpu().int(),
                                             task="multilabel", num_labels=num_classes, average="macro")
                    step_auroc_micro = auroc(preds.cpu(), phenotype_labels.cpu().int(),
                                             task="multilabel", num_labels=num_classes, average="micro")
                    step_ap_macro = average_precision(preds.cpu(), phenotype_labels.cpu().int(),
                                                      task="multilabel", num_labels=num_classes, average="macro")
                    step_ap_micro = average_precision(preds.cpu(), phenotype_labels.cpu().int(),
                                                      task="multilabel", num_labels=num_classes, average="micro")

                    log_data = {
                        "Step": step+1,
                        "Step Loss": loss.item(),
                        "Step Train Phenotype Macro AUROC": step_auroc_macro.item(),
                        "Step Train Phenotype Micro AUROC": step_auroc_micro.item(),
                        "Step Train Phenotype Macro AUPRC": step_ap_macro.item(),
                        "Step Train Phenotype Micro AUPRC": step_ap_micro.item(),
                    }
                    wandb.log(log_data)
                    for k, v in log_data.items():
                        print(f"{k}: {v}")

        # === epoch 끝 ===
        epoch_loss = total_loss / (step + 1)
        macro_auroc = train_metrics["auroc_macro"].compute().item()
        micro_auroc = train_metrics["auroc_micro"].compute().item()
        macro_ap = train_metrics["ap_macro"].compute().item()
        micro_ap = train_metrics["ap_micro"].compute().item()
        per_task_auroc = train_metrics["auroc_per"].compute().cpu().tolist()
        per_task_ap = train_metrics["ap_per"].compute().cpu().tolist()


        if accelerator.is_local_main_process:
            log_data = {
                "Epoch": epoch + 1,
                "Train Loss": epoch_loss,
                "Train Phenotype Macro AUROC": macro_auroc,
                "Train Phenotype Micro AUROC": micro_auroc,
                "Train Phenotype Macro AUPRC": macro_ap,
                "Train Phenotype Micro AUPRC": micro_ap,
            }
            for i, (auroc_i, ap_i) in enumerate(zip(per_task_auroc, per_task_ap)):
                log_data[f"Train Phenotpye {multilabel_labels[i]} AUROC"] = auroc_i
                log_data[f"Train Phenotpye {multilabel_labels[i]} AUPRC"] = ap_i
            wandb.log(log_data)
            for k, v in log_data.items():
                print(f"{k}: {v}")

        # ----- Validation -----
        val_loss, val_metrics = validate_phenotype(device, model, val_loader, accelerator,
                                                   multilabel_criterion, multilabel_labels, epoch+1)
        macro_auroc = val_metrics['macro_auroc']
        macro_auprc = val_metrics['macro_auprc']

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(macro_auroc)

        if macro_auprc > best_macro_auprc:
            best_macro_auprc = macro_auprc
            best_epoch = epoch + 1
            check_patience = 0
            if accelerator.is_local_main_process:
                output_path = Path(save_path) / f"best_{args.exp_name}.pth"
                accelerator.save({'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict()}, output_path)
        else:
            check_patience += 1

        if check_patience >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if accelerator.is_local_main_process:
        print(f"Loading best model from epoch {best_epoch}")
    best_model_path = Path(save_path) / f"best_{args.exp_name}.pth"
    
    accelerator.wait_for_everyone()
    model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'])
        
    test_loss, test_metrics = test_phenotype(device, model, test_loader, accelerator, multilabel_criterion, multilabel_labels)
    return test_loss, test_metrics


def validate_phenotype(device, model, data_loader, accelerator, multilabel_criterion, multilabel_labels, epoch):
    model.eval()
    total_loss = 0
    num_classes = len(multilabel_labels)

    val_metrics = {
        "auroc_macro": torchmetrics.AUROC(task="multilabel", num_labels=num_classes, average="macro").to(device),
        "auroc_micro": torchmetrics.AUROC(task="multilabel", num_labels=num_classes, average="micro").to(device),
        "ap_macro": torchmetrics.AveragePrecision(task="multilabel", num_labels=num_classes, average="macro").to(device),
        "ap_micro": torchmetrics.AveragePrecision(task="multilabel", num_labels=num_classes, average="micro").to(device),
        "auroc_per": torchmetrics.AUROC(task="multilabel", num_labels=num_classes, average=None).to(device),
        "ap_per": torchmetrics.AveragePrecision(task="multilabel", num_labels=num_classes, average=None).to(device),

    }

    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), desc="Validation", total=len(data_loader)):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
             token_type_ids,
             task_ids, multi_labels) = batch

            outputs = model(
                input_ids=input_ids,
                value_ids=value_ids,
                unit_ids=unit_ids,
                time_ids=offset_ids,
                token_type_ids=token_type_ids,
                age_ids=age_ids,
                gender_ids=gender_ids,
                task_token=task_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            phenotype_logits = outputs['phenotype_logits'].squeeze(1)
            phenotype_labels = multi_labels.float()
            preds = torch.sigmoid(phenotype_logits).detach()

            loss = multilabel_criterion(phenotype_logits, phenotype_labels)
            total_loss += loss.item()
            

            phenotype_labels = phenotype_labels.detach().int()

            for m in val_metrics.values():
                m.update(preds.to(accelerator.device), phenotype_labels.to(accelerator.device))

    epoch_loss = total_loss / len(data_loader)
    macro_auroc = val_metrics["auroc_macro"].compute().item()
    micro_auroc = val_metrics["auroc_micro"].compute().item()
    macro_ap = val_metrics["ap_macro"].compute().item()
    micro_ap = val_metrics["ap_micro"].compute().item()
    per_task_auroc = val_metrics["auroc_per"].compute().cpu().tolist()
    per_task_ap = val_metrics["ap_per"].compute().cpu().tolist()

    if accelerator.is_local_main_process:
        log_data = {
            "Epoch": epoch,
            "Validation Loss": epoch_loss,
            "Validation Phenotype Macro AUROC": macro_auroc,
            "Validation Phenotype Micro AUROC": micro_auroc,
            "Validation Phenotype Macro AUPRC": macro_ap,
            "Validation Phenotype Micro AUPRC": micro_ap,
        }
        for i, (auroc_i, ap_i) in enumerate(zip(per_task_auroc, per_task_ap)):
            log_data[f"Validation Phenotpye {multilabel_labels[i]} AUROC"] = auroc_i
            log_data[f"Validation Phenotpye {multilabel_labels[i]} AUPRC"] = ap_i
        wandb.log(log_data)
        for k, v in log_data.items():
            print(f"{k}: {v}")

    return epoch_loss, {"macro_auroc": macro_auroc, "micro_auroc": micro_auroc,
                        "macro_auprc": macro_ap, "micro_auprc": micro_ap}


def test_phenotype(device, model, data_loader, accelerator, multilabel_criterion, multilabel_labels):
    model.eval()
    total_loss = 0
    num_classes = len(multilabel_labels)

    test_metrics = {
        "auroc_macro": torchmetrics.AUROC(task="multilabel", num_labels=num_classes, average="macro").to(device),
        "auroc_micro": torchmetrics.AUROC(task="multilabel", num_labels=num_classes, average="micro").to(device),
        "ap_macro": torchmetrics.AveragePrecision(task="multilabel", num_labels=num_classes, average="macro").to(device),
        "ap_micro": torchmetrics.AveragePrecision(task="multilabel", num_labels=num_classes, average="micro").to(device),
        "auroc_per": torchmetrics.AUROC(task="multilabel", num_labels=num_classes, average=None).to(device),
        "ap_per": torchmetrics.AveragePrecision(task="multilabel", num_labels=num_classes, average=None).to(device),

    }

    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), desc="Test", total=len(data_loader)):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            (input_ids, attention_mask, age_ids, gender_ids, value_ids, unit_ids, offset_ids,
             token_type_ids,
             task_ids, multi_labels) = batch

            outputs = model(
                input_ids=input_ids,
                value_ids=value_ids,
                unit_ids=unit_ids,
                time_ids=offset_ids,
                token_type_ids=token_type_ids,
                age_ids=age_ids,
                gender_ids=gender_ids,
                task_token=task_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            phenotype_logits = outputs['phenotype_logits'].squeeze(1)
            phenotype_labels = multi_labels.float()
            
            preds = torch.sigmoid(phenotype_logits).detach()

            loss = multilabel_criterion(phenotype_logits, phenotype_labels)
            total_loss += loss.item()
            
            phenotype_labels = phenotype_labels.detach().int()

            for m in test_metrics.values():
                m.update(preds.to(accelerator.device), phenotype_labels.to(accelerator.device))

    epoch_loss = total_loss / len(data_loader)
    macro_auroc = test_metrics["auroc_macro"].compute().item()
    micro_auroc = test_metrics["auroc_micro"].compute().item()
    macro_ap = test_metrics["ap_macro"].compute().item()
    micro_ap = test_metrics["ap_micro"].compute().item()
    per_task_auroc = test_metrics["auroc_per"].compute().cpu().tolist()
    per_task_ap = test_metrics["ap_per"].compute().cpu().tolist()

    if accelerator.is_local_main_process:
        log_data = {
            "Test Loss": epoch_loss,
            "Test Phenotype Macro AUROC": macro_auroc,
            "Test Phenotype Micro AUROC": micro_auroc,
            "Test Phenotype Macro AUPRC": macro_ap,
            "Test Phenotype Micro AUPRC": micro_ap,
        }
        for i, (auroc_i, ap_i) in enumerate(zip(per_task_auroc, per_task_ap)):
            log_data[f"Test Phenotpye {multilabel_labels[i]} AUROC"] = auroc_i
            log_data[f"Test Phenotpye {multilabel_labels[i]} AUPRC"] = ap_i
        wandb.log(log_data)
        for k, v in log_data.items():
            print(f"{k}: {v}")

    return epoch_loss, {"macro_auroc": macro_auroc, "micro_auroc": micro_auroc,
                    "macro_auprc": macro_ap, "micro_auprc": micro_ap,
                    "per_task_auroc": per_task_auroc, "per_task_auprc": per_task_ap}