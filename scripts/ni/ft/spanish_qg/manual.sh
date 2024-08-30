#!/bin/bash

for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --pretrained \
        --train_data_dir ~/natural-instructions \
        --session_id manual/ \
	    --val_data_dir ~/natural-instructions \
        --dev_split_path ./aux_data/xlingual_dev_split_map.pkl \
        --ni_task_info_path ./aux_data/ni_xlingual_task_info.pkl \
        --selection_seed ${SEED} \
        --max_steps 600 \
        --xlingual \
        --slice_list question_answering english english question_answering spanish spanish question_generation english english question_generation spanish spanish \
        --k 4 \
        --slicer task_category input_language output_language \
        --sample_rule mixture \
        --target_mask 0 0 0 1 \
        --proportions_schedule 0.25 0.25 0.25 0.25 \ 0.15 0.35 0.15 0.35 \ 0.10 0.40 0.10 0.40 \ 0.05 0.45 0.05 0.45 \ 0.00 0.50 0.00 0.50 \ 0.00 0.40 0.00 0.60 \
        --eta 0.8 \
        --mw_window 3 \
        --update_steps 100 \
        --filter_val_skills \
        --num_ckpts 6
done 


# copy back original file
# mv ~/natural-instructions/tasks/{$task_name}_tmp.json ~/natural-instructions/tasks/{$task_name}.json
# mv ~/natural-instructions/tasks/{$task_name2}_tmp.json ~/natural-instructions/tasks/{$task_name2}.json