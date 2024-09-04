import os 
from datetime import datetime
from utils import make_output_dir
import subprocess


pretrained_script = """for SEED in 0 1 2 3 4
do 
    python3 main.py \
        --task_name ni \
        --pretrained \
        --session_id {output_foldername} \
        --output_folder_method {dataset_version} \
        --train_data_dir {dataset_folder} \
	    --val_data_dir {dataset_folder} \
        --dev_split_path ./aux_data/xlingual_dev_split_map.pkl \
        --ni_task_info_path ./aux_data/ni_xlingual_task_info.pkl \
        --selection_seed ${{SEED}} \
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
        --update_steps 600 \
        --graph_path ./ni_graphs/spanish_qg_normalized.npy \
        --filter_val_skills \
        --num_ckpts 1
done 
"""

dataset_versions = [
    'mixed_00',
    'mixed_01',
    'mixed_02',
    'mixed_03',
    'mixed_04',
    'mixed_05',
    'mixed_06',
    'mixed_07',
    'mixed_08',
    'mixed_09',
    'mixed_10',
]

for dataset_version in dataset_versions:
    output_foldername = f'mixed_data/'
    dataset_folder = f'~/skilldiscovery/dataset/{dataset_version}'
    subprocess.run(pretrained_script.format(output_foldername=output_foldername, dataset_version=dataset_version, dataset_folder=dataset_folder), shell=True, executable='/bin/bash')