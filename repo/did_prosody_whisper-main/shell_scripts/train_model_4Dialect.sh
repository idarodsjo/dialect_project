#!/bin/bash

## CHNAGE THESE ################
train_file=/home/idatro/repo/did_prosody_whisper-main/datasets/nordia_data/train_data.pkl
eval_file=/home/idatro/repo/did_prosody_whisper-main/datasets/nordia_data/validation_data.pkl
label_column=cardinal_four # change as necessary

unmod_model_output_dir=/home/idatro/repo/did_prosody_whisper-main/model_output/nordia_unmod
low_pass_model_output_dir=/path/to/model_output/low_pass
monotonize_model_output_dir=/path/to/model_output/monotonize
output_dirs=($unmod_model_output_dir $low_pass_model_output_dir $monotonize_model_output_dir)

unmod_audio_column=audio_path
low_pass_audio_column=low_pass_audio
monotonize_audio_column=monotonize_audio
audio_columns=($unmod_audio_column $low_pass_audio_column $monotonize_audio_column)

for ((i=0 ; i < 1 ; i++)); do
    model=${output_dirs[$i]}
    echo "Creating model(s) in directory: ${model}"    

    if [ ! -d $model ]; then
        mkdir $model
    else
        echo "${model} already exists. Continuing..."
    fi

    CUDA_VISIBLE_DEVICES=0,1 deepspeed /home/idatro/repo/did_prosody_whisper-main/whisper_did_training.py \
        --deepspeed original_ds_config.json \
        --model_name_or_path NbAiLab/nb-whisper-medium \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --audio_column_name ${audio_columns[$i]} \
        --label_column_name ${label_column} \
        --output_dir ${model} \
        --overwrite_output_dir \
        --remove_unused_columns False \
        --do_train \
        --do_eval \
        --fp16 \
        --learning_rate 3e-5 \
        --max_length_seconds 30 \
        --attention_mask False \
        --warmup_ratio 0.1 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 2 \
        --gradient_checkpointing True \
        --per_device_eval_batch_size 32 \
        --dataloader_num_workers 8 \
        --logging_strategy steps \
        --logging_steps 25 \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --load_best_model_at_end True \
        --metric_for_best_model accuracy \
        --seed 0 \
        --freeze_feature_encoder False 

done