# Usage: bash repretrain/re_run_pretrain.sh


CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/home/DAHS2/DAHS_EHR:$PYTHONPATH \
WANDB_MODE=online \
/home/DAHS2/anaconda3/envs/sj/bin/torchrun --nproc_per_node=1 /home/DAHS2/DAHS_EHR/repretrain/re_pretrain_normal.py \
    --num_hidden_layers 6 \
    --num_attention_heads 8 \
    --intermediate_size 1024 \
    --embedding_size 512 \
    --batch_size 16 \
    --acc 8 \
    --exp_name best_pretrain_mlm_MEP+VP0.01 \
    --learning_rate 1e-4 \
    --weight_decay 0.05 \
    --epochs 30 \
    --dropout_prob 0.3 \
    --mask_mode mlm \
    --mask_ratio 0.3 0.3 0.3 \
    --value_mask_ratio 0.05 \
    --value_loss_weight 0.01 \
    --num_workers 0 \
    --value_embedding_type continuous \
    --data_path /home/DAHS2/DAHS_EHR/new_data_preparation