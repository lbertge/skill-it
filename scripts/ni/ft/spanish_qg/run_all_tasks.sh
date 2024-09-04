#!/bin/bash

# for English QA

for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --train_data_dir ~/natural-instructions \
        --pretrained \
        --session_id 5050_english_qa/ \
	    --val_data_dir ~/natural-instructions \
        --dev_split_path ./aux_data/xlingual_dev_split_map.pkl \
        --ni_task_info_path ./aux_data/ni_xlingual_task_info.pkl \
        --selection_seed ${SEED} \
        --max_steps 600 \
        --xlingual \
        --slice_list question_answering english english question_generation english english \
        --slicer task_category input_language output_language \
        --k 2 \
        --sample_rule mixture \
        --proportions 0.5 0.5 \
        --filter_val_skills \
        --num_ckpts 6
done 

for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --pretrained \
        --train_data_dir ~/natural-instructions \
        --session_id 5050_english_qa/ \
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
        --target_mask 1 0 0 0 \
        --mw \
        --eta 0.8 \
        --mw_window 3 \
        --update_steps 100 \
        --graph_path ./ni_graphs/spanish_qg_normalized.npy \
        --filter_val_skills \
        --num_ckpts 6
done 

# for Spanish QA
for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --train_data_dir ~/natural-instructions \
        --pretrained \
        --session_id 5050_spanish_qa/ \
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
        --num_ckpts 6
done 

for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --pretrained \
        --train_data_dir ~/natural-instructions \
        --session_id 5050_spanish_qa/ \
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
        --target_mask 0 1 0 0 \
        --mw \
        --eta 0.8 \
        --mw_window 3 \
        --update_steps 100 \
        --graph_path ./ni_graphs/spanish_qg_normalized.npy \
        --filter_val_skills \
        --num_ckpts 6
done 

# for English QG
for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --train_data_dir ~/natural-instructions \
        --pretrained \
        --session_id 5050_english_qg/ \
	    --val_data_dir ~/natural-instructions \
        --dev_split_path ./aux_data/xlingual_dev_split_map.pkl \
        --ni_task_info_path ./aux_data/ni_xlingual_task_info.pkl \
        --selection_seed ${SEED} \
        --max_steps 600 \
        --xlingual \
        --slice_list question_answering english english question_generation english english \
        --slicer task_category input_language output_language \
        --k 2 \
        --sample_rule mixture \
        --proportions 0.5 0.5 \
        --filter_val_skills \
        --num_ckpts 6
done 

for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --pretrained \
        --train_data_dir ~/natural-instructions \
        --session_id 5050_english_qg/ \
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
        --target_mask 0 0 1 0 \
        --mw \
        --eta 0.8 \
        --mw_window 3 \
        --update_steps 100 \
        --graph_path ./ni_graphs/spanish_qg_normalized.npy \
        --filter_val_skills \
        --num_ckpts 6
done 

# for Spanish QG
for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --train_data_dir ~/natural-instructions \
        --pretrained \
        --session_id 5050_spanish_qg/ \
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
        --num_ckpts 6
done 

for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --pretrained \
        --train_data_dir ~/natural-instructions \
        --session_id 5050_spanish_qg/ \
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
        --mw \
        --eta 0.8 \
        --mw_window 3 \
        --update_steps 100 \
        --graph_path ./ni_graphs/spanish_qg_normalized.npy \
        --filter_val_skills \
        --num_ckpts 6
done 