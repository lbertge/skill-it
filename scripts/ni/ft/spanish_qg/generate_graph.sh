#!/bin/bash
for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --train_data_dir ~/natural-instructions \
        --pretrained \
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
        --proportions 0 1 0 1 \
        --update_steps 600 \
        --filter_val_skills \
        --num_ckpts 1

    python3 main.py \
        --task_name ni \
        --train_data_dir ~/natural-instructions \
        --pretrained \
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
        --proportions 0 0 0 1 \
        --update_steps 600 \
        --filter_val_skills \
        --num_ckpts 1
done 
