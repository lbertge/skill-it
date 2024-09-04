#!/bin/bash
for SEED in 0
do 
    python3 main.py \
        --task_name ni \
        --train_data_dir ~/natural-instructions \
        --pretrained \
	    --val_data_dir ~/natural-instructions \
        --dev_split_path ./aux_data/xlingual_dev_split_map.pkl \
        --ni_task_info_path ./aux_data/ni_xlingual_task_info.pkl \
        --selection_seed ${SEED} \
        --max_steps 100 \
        --xlingual \
        --slice_list question_answering spanish spanish question_generation spanish spanish \
        --k 2 \
        --slicer task_category input_language output_language \
        --sample_rule mixture \
        --target_mask 0 1 \
        --mw \
        --eta 0.8 \
        --mw_window 3 \
        --update_steps 100 \
        --graph_path ./ni_graphs/spanish_qa_qg_only.npy \
        --filter_val_skills \
        --num_ckpts 1
done 
