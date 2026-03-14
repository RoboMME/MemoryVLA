#!/bin/bash

pretrained_ckpt='./pretrained/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt'
hf_token='YOUR_HF_TOKEN'

data_root_dir='./data'
data_mix='robomme'

n_gpu=2
bs=32
shuffle_buffer_size=128_000 # if your memory is limited, try smaller value

save_interval=10000
dp_step=4
future_action_window_size=15

image_aug=True
run_root_dir='./log/robomme'
run_id='memvla_robomme'

is_resume=False
resume_step=0
resume_epoch=0

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29700 train.py \
  --pretrained_checkpoint ${pretrained_ckpt} \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix ${data_mix} \
  --vla.expected_world_size ${n_gpu} \
  --vla.per_device_batch_size ${bs} \
  --vla.global_batch_size $((n_gpu * bs)) \
  --vla.learning_rate 2e-5 \
  --vla.max_steps 160000 \
  --data_root_dir ${data_root_dir} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --image_aug ${image_aug} \
  --save_interval ${save_interval} \
  --repeated_diffusion_steps ${dp_step} \
  --future_action_window_size ${future_action_window_size} \
  --action_model_type 'DiT-L' \
  --dataloader_type 'group' \
  --is_resume ${is_resume} \
  --resume_step ${resume_step} \
  --resume_epoch ${resume_epoch} \
  --wandb_project 'memvla' \
  --wandb_entity 'YOUR_WANDB_ENTITY' \
  --hf_token ${hf_token} \
  --vla.shuffle_buffer_size ${shuffle_buffer_size} \
