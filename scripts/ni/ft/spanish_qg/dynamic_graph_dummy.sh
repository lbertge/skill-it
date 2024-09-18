#!/bin/bash
for SEED in 0
do 
    python3 main.py \
        --task_name ni \
        --train_data_dir /data/albert/natural-instructions \
        --pretrained \
	    --val_data_dir /data/albert/natural-instructions \
        --dev_split_path ./aux_data/xlingual_dummy_dev_split_map.pkl \
        --ni_task_info_path ./aux_data/ni_xlingual_task_info.pkl \
        --selection_seed ${SEED} \
        --max_steps 600 \
        --xlingual \
        --slice_list question_answering english english question_answering spanish spanish question_generation english english question_generation spanish spanish \
        --k 4 \
        --slicer task_category input_language output_language \
        --sample_rule mixture \
        --target_mask 0 0 0 1 \
        --dynamic_graph \
        --eta 0.8 \
        --mw_window 3 \
        --update_steps 10 \
        --graph_path ./ni_graphs/spanish_qg_normalized.npy \
        --filter_val_skills \
        --num_ckpts 60
done 