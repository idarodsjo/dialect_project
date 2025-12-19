#!/bin/bash

	
normal_model=/home/idatro/repo/did_prosody_whisper-main/model_output/nordia_unmod/checkpoint-24
models=($normal_model)

normal_run_dir=/home/idatro/repo/did_prosody_whisper-main/model_output
output_dirs=($normal_run_dir)

# ### NVOS constants
nvos_full_dataset=/home/idatro/repo/did_prosody_whisper-main/datasets/nordia_data/test_data.pkl
nvos_label_column=cardinal_four
nvos_audio_column=audio_path

for ((i=0 ; i < 1 ; i++)); do
    model=${models[$i]}
    echo "Using model: ${model}"
    run_output_dir="${output_dirs[$i]}"    

    if [ ! -d $run_output_dir ]; then
        mkdir $run_output_dir
    else
        echo "${run_output_dir} already exists. Continuing..."
    fi
    
    echo "--- NVOS (normal) ---"
    python /home/idatro/repo/did_prosody_whisper-main/whisper_did_eval.py --model_path ${model} --save_path "${run_output_dir}/nvos_full/" --test_dataset $nvos_full_dataset --label_column $nvos_label_column --audio_column $nvos_audio_column
done
