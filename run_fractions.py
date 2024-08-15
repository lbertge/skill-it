import os 
from datetime import datetime
from utils import make_output_dir

script = """for SEED in 0 1 2 3 4
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
"""

task_names = ['task1610_xquad_es_answer_generation', 'task1334_sqac_answer_generation']

dataset_versions = [
    'real_01_split',
    'real_02_split',
    'real_04_split',
    'real_095_split',
    'synthetic_01_split',
    'synthetic_02_split',
    'synthetic_04_split',
    'synthetic_095_split',
]

def find_first_folder_in_folder(folder_path):
    # Traverse the directory
    for root, dirs, files in os.walk(folder_path):
        if dirs:
            # Return the full path to the first folder found
            return os.path.join(root, dirs[0])
    # Return None if no folders are found
    return None

for dataset_version in dataset_versions:
    for task_name in task_names:
        os.system(f"cp ~/skilldiscovery/dataset/{dataset_version}/{task_name}.json ~/natural-instructions/tasks/{task_name}.json")
    print(f"Finished copying synthetic data {dataset_version}")

    run_id = datetime.now().strftime("%m%d%Y")

    os.system(script)

    output_dir_path = make_output_dir('./output/', None, run_id)

    # create a folder named ./output/synthetic_fractions 
    output_foldername = './output/synthetic_fractions'
    if not os.path.exists(output_foldername):
        os.mkdir(output_foldername)

    # find the generated output folder in output_dir_path
    generated_output = find_first_folder_in_folder(output_dir_path)    
    new_output_name = output_foldername + '/' + dataset_version
    print(f"Copying {generated_output} to {new_output_name}")
    os.system(f"cp -r {generated_output} {new_output_name}")






    

    
