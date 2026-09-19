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
import pandas as pd
import pickle
import time


# def compute_mlm_loss(predictions, labels):
#     mask = torch.ones_like(labels)
#     mask[:, :3] = 0  
    
#     predictions = predictions.reshape(-1, predictions.size(-1))
#     labels = labels.reshape(-1)
#     mask = mask.reshape(-1)
#     loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
#     loss = loss_fct(predictions[mask.bool()], labels[mask.bool()])
    
#     return loss

# def calculate_mlm_precision(predictions, labels):
#     mask = torch.ones_like(labels)
#     mask[:, :3] = 0  
    
#     predicted_labels = predictions.argmax(dim=-1)
#     valid_mask = labels != -100
    
#     correct_predictions = (predicted_labels == labels) & valid_mask & mask.bool()
    
#     num_correct = correct_predictions.sum().item()
#     num_valid = (valid_mask & mask.bool()).sum().item()
    
#     precision = num_correct / num_valid if num_valid > 0 else 0.0
    
#     return precision

def compute_mlm_loss(predictions, labels):
    loss_fct = nn.CrossEntropyLoss()  
    return loss_fct(predictions.view(-1, predictions.size(-1)), labels.view(-1))


def calculate_mlm_precision(predictions, labels):
    correct_predictions = (predictions == labels)
    num_correct = correct_predictions.sum().item()
    num_valid = labels.numel()
    
    precision = num_correct / num_valid if num_valid > 0 else 0.0
    
    return precision

def calculate_mlm_precision_by_type(preds, labels, token_types, type_list=[0, 1, 2]):
    results = {}
    for t in type_list:
        mask = (token_types == t) & (labels != -100)
        if mask.sum() == 0:
            results[f"type_{t}_precision"] = None
            continue
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        results[f"type_{t}_precision"] = correct / total
    return results

def get_similarity(predictions, labels, similarity_map):
    similarity = []
    for pred, label in zip(predictions, labels):
        # pred_label = similarity_map[similarity_map]
        true_label = similarity_map[similarity_map['itemid'] == label]['label'].values[0]
        
        similarity_value = similarity_map[similarity_map['itemid'] == pred][true_label].values[0]
        similarity.append(similarity_value)
        
    return similarity

def compute_mlm_loss_with_similarity(predictions, labels, similarity_map, idx2label, similarity_factor):
    
    prediction_score = predictions.argmax(dim=-1).float().to(predictions.device)
    labels = labels.long().to(labels.device)
    # tqdm.write(prediction_score)
    predictions_item = [idx2label[tensor.item()] for tensor in prediction_score]
    label_item = [idx2label[tensor.item()] for tensor in labels]
    
    similarities = get_similarity(predictions_item, label_item, similarity_map)
    
    # tqdm.write(predictions)
    # tqdm.write(labels)
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    base_loss = loss_fct(predictions.view(-1, predictions.size(-1)), labels.view(-1))
    similarities = torch.tensor(similarities, dtype=torch.float32).to(predictions.device)
    
    # tqdm.write(base_loss)
    # tqdm.write("========")
    # tqdm.write(similarities.view(-1))
    # similarity_loss = torch.mean(1 - similarities)
    
    total_loss = base_loss * (1 - similarities) * similarity_factor
 
    return torch.mean(total_loss), torch.mean(1 - similarities)

def safe_format(x):
    return f"{x:.4f}" if x is not None else "N/A"

def safe_round(x):
    return round(x, 4) if x is not None else None



def train(
    device: torch.device,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    accelerator: Accelerator,
    epochs: int,
    start_epoch: int,
    patience: int,
    save_path: Path,
    args: dict
):
    logging.info("Start training...")
    check_patience = 0
    
    best_loss = 1e10
    best_epoch = 0
    
    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs", total=epochs-start_epoch):
        # train_loss = 0
        # train_mlm_loss = 0
        # train_value_pred_loss = 0
        # train_discriminator_loss = 0
        
        train_loss, train_mlm_loss, train_value_pred_loss = 0, 0, 0
        
        all_predictions, all_labels, all_token_types = [], [], []
        # start = time.time()
        model.train()
        
        for step, batch in tqdm(enumerate(data_loader), desc="Steps", total=len(data_loader)):
            step_start_time = time.time()
            with accelerator.accumulate(model):
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)   
                if args.mask_mode in ["mlm", "span_mlm"]:
                    (
                        input_ids,
                        attention_mask,
                        age_ids,
                        gender_ids,
                        value_ids,
                        unit_ids,
                        offset_ids,
                        token_type_ids,
                        task_ids,
                        labels,
                        value_labels,
                    ) = batch
                    discriminator_labels = None
                else:
                    raise NotImplementedError(
                        "The current PULSE-ICU Dataset supports MLM + value prediction only."
                    )
                # batch_preprocess_time = time.time() - step_start_time
                # print(f"Batch preprocess time: {batch_preprocess_time}")
                # batch_size = labels.size(0)
                # additional_tokens = torch.tensor([1, 1, 1]).unsqueeze(0).repeat(batch_size, 1).to(device)
                # The model prepends [task, age, gender] to the event sequence.
                # Add 3 dummy positions to labels so model-side [:, 3:] aligns them
                # back to the event positions. Use the actual batch size so the
                # final short batch also works.
                batch_size = labels.size(0)
                additional_tokens = torch.ones(
                    (batch_size, 3),
                    dtype=labels.dtype,
                    device=device,
                )
                additional_value_tokens = torch.ones(
                    (batch_size, 3),
                    dtype=value_labels.dtype,
                    device=device,
                )
                labels = torch.cat([additional_tokens, labels], dim=1)
                value_labels = torch.cat(
                    [additional_value_tokens, value_labels],
                    dim=1,
                )
                # values = torch.cat([additional_tokens, values], dim=1)
                # with torch.autocast(device_type=device.type, dtype=torch.float16):   
                # model_start_time = time.time()
                outputs = model(
                    input_ids = input_ids,
                    value_ids = value_ids,
                    unit_ids = unit_ids,
                    time_ids = offset_ids,
                    token_type_ids = token_type_ids,
                    age_ids = age_ids,
                    gender_ids = gender_ids,
                    task_token = task_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=None,
                    labels=labels,
                    value_labels=value_labels,
                    discriminator_labels=discriminator_labels,
                    return_dict=True,)
                # model_end_time = time.time() - model_start_time
                # print(f"Model time: {model_end_time}")
                
                # calculate_start_time = time.time()
                prediction_scores = outputs["mlm_logits"][:, 3:, :]
                labels = labels[:, 3:]
                value_labels = value_labels[:, 3:]
                
                valid_mask = labels!=-100
                valid_predictions = prediction_scores.argmax(dim=-1)[valid_mask]
                valid_labels = labels[valid_mask]
                valid_token_types = token_type_ids[valid_mask]
                
                step_precision = calculate_mlm_precision(valid_predictions, valid_labels)
                step_type_precision = calculate_mlm_precision_by_type(valid_predictions, valid_labels, valid_token_types)
                # mlm_loss = compute_mlm_loss(prediction_scores[valid_mask], valid_labels)
                mlm_loss = outputs["mlm_loss"]
                value_loss = outputs["value_pred_loss"]
                
                total_loss = outputs['loss']
                accelerator.backward(total_loss)
                
                train_loss += total_loss.item()
                train_mlm_loss += mlm_loss.item()
                train_value_pred_loss += value_loss.item()
                

                all_predictions.append(valid_predictions)
                all_labels.append(valid_labels)
                all_token_types.append(valid_token_types)
                
                # With Accelerator gradient accumulation, gradients are synchronized
                # only at the real optimizer-update boundary.
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                # AcceleratedOptimizer handles accumulation internally. Advance the
                # raw scheduler only when a real optimizer update occurs.
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad()
                
                # valid_scores = prediction_scores[valid_mask]
                
                # mlm_loss = compute_mlm_loss(valid_scores, valid_labels)
                # train_mlm_loss += mlm_loss.item()
                # calculate_end_time = time.time() - calculate_start_time
                # print(f"Calculate time: {calculate_end_time}")
                
                # if "loss_discriminator" in outputs and outputs["discriminator_loss"] is not None:
                #     train_discriminator_loss += outputs["discriminator_loss"].item()
       
                # backprop_start_time = time.time()
                # train_loss += outputs['loss'].item()
                # accelerator.backward(outputs['loss'])
                # backprop_end_time = time.time() - backprop_start_time
                
                

                
                # print(f"Backprop time: {backprop_end_time}")
                if torch.isnan(outputs['loss']):
                    print(f"NaN detected at epoch {epoch+1}, step {step}")
                if torch.isnan(outputs['mlm_loss']):
                    print(f"NaN detected mlm_loss at epoch {epoch+1}, step {step}")
                                    
                    
                if step % 1000 == 0 and accelerator.is_local_main_process:   
                    # tqdm.write(f"Epoch {epoch+1}, Step {step}, Learning Rate: {scheduler.get_lr()}")
                    tqdm.write(f"Epoch {epoch+1}, Step {step}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")


                    if args.use_discriminator:
                        tqdm.write(f"Epoch {epoch+1} | Step {step} | Train Loss: {outputs['loss'].item():.4f} | MLM Loss: {outputs['mlm_loss'].item():.4f} | Discriminator Loss: {outputs['discriminator_loss'].item():.4f} | Precision: {step_precision:.4f}")   
                        wandb.log({
                        "Step": step,
                        "Step Loss": round(outputs['loss'].item(), 4),
                        "Step MLM Loss": round(outputs['mlm_loss'].item(), 4),
                        "Step Value Prediction Loss": round(outputs['value_pred_loss'].item(), 4),
                        "Step Discriminator Loss": round(outputs['discriminator_loss'].item()),
                        "Step Precision": round(step_precision, 4),
                    }) 
            
                    
                    else:
                        # tqdm.write(f"Epoch {epoch+1} | Step {step} | Train Loss: {outputs['loss'].item():.4f} | MLM Loss: {outputs['mlm_loss'].item():.4f} | Valid Value Prediction Loss: {outputs['value_pred_loss']:.4f} | Precision: {step_precision:.4f} | Precision Medication: {step_type_precision['type_0_precision']:.4f} | Precision Chart: {step_type_precision['type_1_precision']:.4f} | Precision Procedure: {step_type_precision['type_2_precision']:.4f}")
                        tqdm.write(
                            f"Epoch {epoch+1} | Step {step} | Train Loss: {outputs['loss'].item():.4f} | "
                            f"MLM Loss: {outputs['mlm_loss'].item():.4f} | "
                            f"Value Prediction Loss: {outputs['value_pred_loss'].item():.4f} | "
                            f"Precision: {safe_format(step_precision)} | "
                            f"Precision Medication: {safe_format(step_type_precision.get('type_0_precision'))} | "
                            f"Precision Chart: {safe_format(step_type_precision.get('type_1_precision'))} | "
                            f"Precision Procedure: {safe_format(step_type_precision.get('type_2_precision'))}"
                        )

                        # wandb 로그
                        wandb.log({
                            "Step": step,
                            "Step Loss": round(outputs['loss'].item(), 4),
                            "Step MLM Loss": round(outputs['mlm_loss'].item(), 4),
                            "Step Value Prediction Loss": round(outputs['value_pred_loss'].item(), 4),
                            "Step Precision": safe_round(step_precision),
                            "Step Precision Medication": safe_round(step_type_precision.get('type_0_precision')),
                            "Step Precision Chart": safe_round(step_type_precision.get('type_1_precision')),
                            "Step Precision Procedure": safe_round(step_type_precision.get('type_2_precision')),
                        })
        
                        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)   
        all_token_types = torch.cat(all_token_types, dim=0)
        
        precision = calculate_mlm_precision(all_predictions, all_labels)
        precision_by_type = calculate_mlm_precision_by_type(all_predictions, all_labels, all_token_types)
        avg_loss = train_loss / len(data_loader)
        avg_mlm_loss = train_mlm_loss / len(data_loader)
        avg_value_pred_loss = train_value_pred_loss / len(data_loader)
        
        
        accelerator.wait_for_everyone()

                
        if accelerator.is_local_main_process:
            if args.use_discriminator:
                tqdm.write(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Train MLM Loss: {train_mlm_loss / len(data_loader):.4f} | Train Discriminator Loss: {train_discriminator_loss / len(data_loader):.4f} | Precision: {precision:.4f}")
                wandb.log({
                "Epoch": epoch+1,
                "Train Loss": round(avg_loss, 4),
                "Train MLM Loss": round(train_mlm_loss / len(data_loader), 4),
                # "Train Discriminator Loss": round(train_discriminator_loss / len(data_loader), 4),
                "Train Precision": round(precision, 4),
            })
            else:
                # tqdm.write(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}| Train MLM Loss: {train_mlm_loss / len(data_loader):.4f} | Train Value Prediction Loss: {outputs['value_pred_loss']:.4f} | Train Precision: {precision:.4f} | Train Precision Medication: {precision_by_type['type_0_precision']:.4f} | Train Precision Chart: {precision_by_type['type_1_precision']:.4f} | Train Precision Procedure: {precision_by_type['type_2_precision']:.4f}")
                # tqdm.write(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}| Train MLM Loss: {train_mlm_loss / len(data_loader):.4f} | Train Precision: {precision:.4f} | Train Precision Medication: {precision_by_type['type_0_precision']:.4f} | Train Precision Chart: {precision_by_type['type_1_precision']:.4f} | Train Precision Procedure: {precision_by_type['type_2_precision']:.4f}")
                tqdm.write(
                    f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | "
                    f"Train MLM Loss: {avg_mlm_loss:.4f} | "
                    f"Train Value Prediction Loss: {avg_value_pred_loss:.4f} | "
                    f"Train Precision: {safe_format(precision)} | "
                    f"Train Precision Medication: {safe_format(precision_by_type.get('type_0_precision'))} | "
                    f"Train Precision Chart: {safe_format(precision_by_type.get('type_1_precision'))} | "
                    f"Train Precision Procedure: {safe_format(precision_by_type.get('type_2_precision'))}"
                )
                
                wandb.log({
                    "Epoch": epoch+1,
                    "Train Loss": round(avg_loss, 4),
                    "Train MLM Loss": round(avg_mlm_loss, 4),
                    "Train Value Prediction Loss": round(avg_value_pred_loss, 4),
                    "Train Precision": round(precision, 4),
                    "Train Precision Medication": round(precision_by_type['type_0_precision'], 4),
                    "Train Precision Chart": round(precision_by_type['type_1_precision'], 4),
                    "Train Precision Procedure": round(precision_by_type['type_2_precision'], 4),
                }) 

        if args.use_discriminator:    
            valid_loss, valid_mlm_loss, valid_discriminator_loss, valid_precision  = validation(device, model, val_loader, accelerator, args)
        else:
            valid_loss, valid_mlm_loss, valid_value_pred_loss, valid_precision, valid_precision_by_type = validation(device, model, val_loader, accelerator, args)

        if accelerator.is_local_main_process:
            if args.use_discriminator:
                tqdm.write(f"Epoch {epoch+1} | Valid Loss: {valid_loss:.4f} | Valid MLM Loss: {valid_mlm_loss:.4f} | Valid Discriminator Loss: {valid_discriminator_loss:.4f} | Valid Precision: {valid_precision:.4f}")
                wandb.log({
                "Validation Loss": round(valid_loss, 4),
                "Validation MLM Loss": round(valid_mlm_loss, 4),
                "Validation Discriminator Loss": round(valid_discriminator_loss, 4),
                "Validation Precision": round(valid_precision, 4),
            })
                
            else:
                # tqdm.write(f"Epoch {epoch+1} | Valid Loss: {valid_loss:.4f} | Valid MLM Loss: {valid_mlm_loss:.4f} | Valid Value Prediction Loss: {valid_value_pred_loss:.4f} | Valid Precision: {valid_precision:.4f} | Valid Precision Medication: {valid_precision_by_type['type_0_precision']:.4f} | Valid Precision Chart: {valid_precision_by_type['type_1_precision']:.4f} | Valid Precision Procedure: {valid_precision_by_type['type_2_precision']:.4f}")


                # tqdm 출력
                tqdm.write(
                    f"Epoch {epoch+1} | Valid Loss: {valid_loss:.4f} | "
                    f"Valid MLM Loss: {valid_mlm_loss:.4f} | "
                    f"Valid Value Prediction Loss: {valid_value_pred_loss:.4f} | "
                    f"Valid Precision: {safe_format(valid_precision)} | "
                    f"Valid Precision Medication: {safe_format(valid_precision_by_type.get('type_0_precision'))} | "
                    f"Valid Precision Chart: {safe_format(valid_precision_by_type.get('type_1_precision'))} | "
                    f"Valid Precision Procedure: {safe_format(valid_precision_by_type.get('type_2_precision'))}"
                )

                # wandb 로그
                wandb.log({
                    "Validation Loss": round(valid_loss, 4),
                    "Validation MLM Loss": round(valid_mlm_loss, 4),
                    "Validation Value Prediction Loss": round(valid_value_pred_loss, 4),
                    "Validation Precision": round(valid_precision, 4),
                    "Validation Precision Medication": round(valid_precision_by_type['type_0_precision'], 4),
                    "Validation Precision Chart": round(valid_precision_by_type['type_1_precision'], 4),
                    "Validation Precision Procedure": round(valid_precision_by_type['type_2_precision'], 4),
                })
                
        if epoch+1 in [30, 50, 100]:
            output_path_manual = Path(save_path) / f"pretrain_model_{args.exp_name}_epoch{epoch+1}.pth"
            accelerator.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, output_path_manual)
                    
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch + 1
            
            if accelerator.is_local_main_process:
                if args.use_discriminator:
                    wandb.log({
                        'Best Epoch': best_epoch,
                        'Best Loss': round(best_loss,4),
                        'Best MLM Loss': round(valid_mlm_loss,4),
                        'Best Discriminator Loss': round(valid_discriminator_loss,4),
                        'Best Precision': round(valid_precision,4),
                        # 'Best Value Regression loss': valid_reg_loss
                    })
                else:                   
                    wandb.log({
                        'Best Epoch': best_epoch,
                        'Best Loss': round(best_loss,4),
                        'Best MLM Loss': round(valid_mlm_loss,4),
                        'Best Value Prediction Loss': round(valid_value_pred_loss,4),
                        'Best Precision': round(valid_precision,4),
                        # 'Best Value Regression loss': valid_reg_loss
                        'Best Precision Medication': round(valid_precision_by_type['type_0_precision'], 4),
                        'Best Precision Chart': round(valid_precision_by_type['type_1_precision'], 4),
                        'Best Precision Procedure': round(valid_precision_by_type['type_2_precision'], 4),
                    })
                print("Saving best model at epoch", best_epoch)
                output_path = Path(save_path) / f"best_pretrain_model_{args.exp_name}.pth"
                accelerator.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, output_path)
                
            check_patience = 0
        else:
            check_patience += 1
            
        if epoch > epochs // 2 and check_patience >= patience:
            break
        
        accelerator.wait_for_everyone()

        
def validation(
    device: torch.device,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    # scaler: torch.amp.GradScaler,
    accelerator: Accelerator,
    # similarity_map: pd.DataFrame,
    # idx2label: pickle,
    args: dict
):
    
    logging.info("Start validation...")
    # val_loss = 0
    # val_mlm_loss = 0
    # val_discriminator_loss = 0
    
    val_loss, val_mlm_loss, val_value_pred_loss = 0, 0, 0
    
    all_predictions, all_labels, all_token_types = [], [], []
    model.eval()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), desc="Validation", total=len(data_loader)):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            if args.mask_mode in ["mlm", "span_mlm"]:
                (
                    input_ids,
                    attention_mask,
                    age_ids,
                    gender_ids,
                    value_ids,
                    unit_ids,
                    offset_ids,
                    token_type_ids,
                    task_ids,
                    labels,
                    value_labels,
                ) = batch
                discriminator_labels = None
            else:
                raise NotImplementedError(
                    "The current PULSE-ICU Dataset supports MLM + value prediction only."
                )
   
            # batch_size = labels.size(0)
            # additional_tokens = torch.tensor([1, 1, 1]).unsqueeze(0).repeat(batch_size, 1).to(device)
            batch_size = labels.size(0)
            additional_tokens = torch.ones(
                (batch_size, 3),
                dtype=labels.dtype,
                device=device,
            )
            additional_value_tokens = torch.ones(
                (batch_size, 3),
                dtype=value_labels.dtype,
                device=device,
            )
            labels = torch.cat([additional_tokens, labels], dim=1)
            value_labels = torch.cat(
                [additional_value_tokens, value_labels],
                dim=1,
            )
            # with torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(
                input_ids = input_ids,
                value_ids = value_ids,
                unit_ids = unit_ids,
                time_ids = offset_ids,
                token_type_ids = token_type_ids,
                age_ids = age_ids,
                gender_ids = gender_ids,
                task_token = task_ids,
                attention_mask=attention_mask,
                global_attention_mask=None,
                labels=labels,
                value_labels=value_labels,
                discriminator_labels=discriminator_labels,
                return_dict=True,)
                
            prediction_scores = outputs["mlm_logits"][:, 3:, :]
            labels = labels[:, 3:]
            value_labels = value_labels[:, 3:]
            
            valid_mask = labels!=-100
            valid_predictions = prediction_scores.argmax(dim=-1)[valid_mask]
            valid_labels = labels[valid_mask]
            valid_token_types = token_type_ids[valid_mask]
        
            all_predictions.append(valid_predictions)
            all_labels.append(valid_labels)
            all_token_types.append(valid_token_types)
            
            # valid_scores = prediction_scores[valid_mask]

            # mlm_loss = compute_mlm_loss(valid_scores, valid_labels)
            # val_mlm_loss += mlm_loss.item()
            
            if "loss_discriminator" in outputs and outputs["discriminator_loss"] is not None:
                val_discriminator_loss += outputs["discriminator_loss"].item()
            
            val_loss += outputs['loss'].item()
            val_mlm_loss += outputs['mlm_loss'].item()
            val_value_pred_loss += outputs['value_pred_loss'].item()
            
            
            
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)  
        all_token_types = torch.cat(all_token_types, dim=0)
        
        precision = calculate_mlm_precision(all_predictions, all_labels)
        precision_by_type = calculate_mlm_precision_by_type(all_predictions, all_labels, all_token_types)
        avg_loss = val_loss / len(data_loader)
        avg_mlm_loss = val_mlm_loss / len(data_loader)
        avg_value_pred_loss = val_value_pred_loss / len(data_loader)


        accelerator.wait_for_everyone()
    if args.use_discriminator:
        return avg_loss, val_mlm_loss/len(data_loader), val_discriminator_loss/len(data_loader), precision
    else:
        return avg_loss, avg_mlm_loss,avg_value_pred_loss, precision, precision_by_type
        # return avg_loss, avg_mlm_loss, val_value_pred_loss, precision, precision_by_type
    
