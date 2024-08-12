#!/bin/bash

# copy synthetic replacement to the appropriate location first

task_name=task1610_xquad_es_answer_generation
dataset_version=v4
mv ~/natural-instructions/tasks/${task_name}.json ~/natural-instructions/tasks/${task_name}_tmp.json
cp ~/skilldiscovery/dataset/v4/${task_name}.json ~/natural-instructions/tasks/${task_name}.json

task_name2=task1334_sqac_answer_generation
mv ~/natural-instructions/tasks/${task_name2}.json ~/natural-instructions/tasks/${task_name2}_tmp.json
cp ~/skilldiscovery/dataset/v4/${task_name2}.json ~/natural-instructions/tasks/${task_name2}.json

echo "Finished copying synthetic replacement to ~/natural-instructions/tasks/"

for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --train_data_dir ~/natural-instructions \
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


# copy back original file
# mv ~/natural-instructions/tasks/{$task_name}_tmp.json ~/natural-instructions/tasks/{$task_name}.json
# mv ~/natural-instructions/tasks/{$task_name2}_tmp.json ~/natural-instructions/tasks/{$task_name2}.json