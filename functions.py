import os
import json
import sys

import numpy as np
import datetime
from pathlib import Path
import tensorflow as tf
import subprocess as sp

def validate_result_directory(model_path, results_folder):
    """Creating a new folder and validating if results folder exists """
    folder_name = os.path.basename(model_path).split('.')[0]
    save_folder = os.path.join(results_folder, folder_name)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    return save_folder, folder_name


def myconverter(obj):
    """Converting all results to be in type for json dump after prediction"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()


def save_results_json(image_path, save_folder, folder_name, class_names, r):
    file_name = os.path.basename(image_path).split('.')[0] + '_' + folder_name
    save_path = os.path.join(save_folder, file_name)

    new_class = [class_names[str(i)] for i in r['class_ids']]
    masks = [np.where(r['masks'][:, :, i] > 0) for i in range(np.shape(r['masks'])[-1])]
    x_mask = [masks[i][1] for i in range(len(masks))]
    y_mask = [masks[i][0] for i in range(len(masks))]

    results_dict = {'class_name': new_class, 'class_ids': r['class_ids'],
                    'scores': r['scores'], 'rois [y_min, x_min, y_max, x_max]': list(r['rois']),
                    'x_mask': x_mask, 'y_mask': y_mask}

    new_dict = {'image_path': os.path.abspath(image_path), 'class_names': class_names, 'results': results_dict}

    with open(save_path + '.json', 'w') as fp:
        json.dump(new_dict, fp, default=myconverter)


def readJson(model_path, json_path, k=7):
    if json_path is None:
        json_path = ''
        my_json = "config.json"
        model_dir = os.path.dirname(os.path.abspath(model_path))

        for file in os.listdir(model_dir):
            if my_json in file:
                json_path = os.path.join(model_dir, file)

    if os.path.exists(json_path):
        with open(json_path) as js:
            config_data = json.load(js)
    else:
        print("config file not found")
        sys.exit()
        # new_values = [' ' for i in range(k)]
        # new_keys = [str(i) for i in range(5)]
        # class_names = dict(zip(new_keys, new_values))

    return config_data


def gpu_memory_usage(memory_th=700, gpu_id=0):
    if not tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
        print("No GPU Found")
        return False

    command = f"nvidia-smi --id={gpu_id} --query-gpu=memory.used --format=csv"
    output_cmd = sp.check_output(command.split())

    memory_used = output_cmd.decode("ascii").split("\n")[1]
    # Get only the memory part as the result comes as '10 MiB'
    memory_used = int(memory_used.split()[0])

    if memory_used > memory_th:
        print("GPU is occupied")
        return False

    return True
