#!/bin/bash

### eval both all trained models using NVOS ###########
	
normal_model=/path/to/trained_models/unmodified/checkpoint
low_pass_model=/path/to/trained_models/low_pass/checkpoint
monotonized_model=/path/to/trained_models/monotonize/checkpoint
models=($normal_model $low_pass_model $monotonized_model)

normal_run_dir=/path/to/results/unmodified
low_pass_run_output_dir=/path/to/results/unmodified
monotonized_run_output_dir=/path/to/results/unmodified
output_dirs=($normal_run_dir $low_pass_run_output_dir $monotonized_run_output_dir)

# ### NVOS constants
nvos_full_dataset=audio_path
nvos_label_column=dialect
# nvos_audio_column=full_audio_path

for ((i=0 ; i < 3 ; i++)); do
    model=${models[$i]}
    echo "Using model: ${model}"
    run_output_dir="${output_dirs[$i]}"    

    if [ ! -d $run_output_dir ]; then
        mkdir $run_output_dir
    else
        echo "${run_output_dir} already exists. Continuing..."
    fi
    
    echo "--- NVOS (normal) ---"
    python ../whisper_did_eval.py --model_path ${model} --save_path "${run_output_dir}/nvos_full/" --test_dataset $nvos_full_dataset --label_column $nvos_label_column --audio_column full_audio_path
    
    echo "--- NVOS (low-pass) ---"
    python ../whisper_did_eval.py --model_path ${model} --save_path "${run_output_dir}/nvos_full_low_pass/" --test_dataset $nvos_full_dataset --label_column $nvos_label_column --audio_column low_pass_audio
    
    echo "--- NVOS (monotonized) ---"
    python ../whisper_did_eval.py --model_path ${model} --save_path "${run_output_dir}/nvos_full_monotonized/" --test_dataset $nvos_full_dataset --label_column $nvos_label_column --audio_column monotonize_audio
done
