#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6081
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8

config_dir=./
config_name=pretrain_vl_3B
save_dir=../../checkpoints/pretrain/one_peace_vl_3B

torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../train.py \
    --config-dir=${config_dir} \
    --config-name=${config_name} \
    checkpoint.save_dir=${save_dir}