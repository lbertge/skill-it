#!/bin/bash
SEED=0

for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --train_data_dir ~/natural-instructions \
        --pretrained \
        --session_id 5050_bsize_16/ \
	    --val_data_dir ~/natural-instructions \
        --dev_split_path ./aux_data/xlingual_dev_split_map.pkl \
        --ni_task_info_path ./aux_data/ni_xlingual_task_info.pkl \
        --selection_seed ${SEED} \
        --max_steps 600 \
        --xlingual \
        --slice_list question_answering spanish spanish question_generation spanish spanish \
        --slicer task_category input_language output_language \
        --k 2 \
        --sample_rule mixture \
        --proportions 0.5 0.5 \
        --filter_val_skills \
        --num_ckpts 6 \
        --batch_size 16
done 
