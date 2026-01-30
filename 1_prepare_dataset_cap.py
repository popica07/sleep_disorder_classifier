import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
import argparse
from loguru import logger
from tqdm import tqdm
import multiprocessing
import random
import config
from config import LABEL_MAP, PATH_TO_PROCESSED_DATA, LABELS_DICT

INT_TO_LABEL_STR = {v: k for k, v in LABELS_DICT.items()}

def parallel_prepare_data(args):
    mrns = args[0]
    dataset_dir = args[1]
    mrn_pretrain = args[2]
    mrn_train = args[3]
    mrn_valid = args[4]
    mrn_test = args[5]

    data_dict = {
        "pretrain": [],
        "train": [], 
        "valid": [],
        "test": []
    }

    empty_label_dict_counts = 0
    path_to_Y = os.path.join(dataset_dir, "Y")
    
    for mrn in tqdm(mrns):
        one_patient_dict = {
            mrn: {}
        }

        path_to_X = os.path.join(dataset_dir, "X")
        path_to_patient = os.path.join(path_to_X, mrn)
        path_to_label = os.path.join(path_to_Y, f"{mrn}.pickle")

        assigned_splits = []
        if mrn in mrn_pretrain: assigned_splits.append("pretrain")
        if mrn in mrn_train: assigned_splits.append("train")
        if mrn in mrn_valid: assigned_splits.append("valid")
        if mrn in mrn_test: assigned_splits.append("test")
        
        if not assigned_splits:
            continue

        if os.path.exists(path_to_label):
            with open(path_to_label, 'rb') as file:
                labels_dict = pickle.load(file)
        else:
            continue
        
        if len(labels_dict) == 0:
            empty_label_dict_counts += 1
            continue
        
        if not os.path.exists(path_to_patient):
            continue

        for event_data_name in os.listdir(path_to_patient):
            event_data_path = os.path.join(path_to_patient, event_data_name)

            if event_data_name not in labels_dict:
                continue

            label = labels_dict[event_data_name]
            
            if isinstance(label, int):
                if label in INT_TO_LABEL_STR:
                    label = INT_TO_LABEL_STR[label] 
                else:
                    continue
            elif isinstance(label, dict):
                label = list(label.keys())[0]
            
    
            if label in LABELS_DICT:
                pass 
            elif label in LABEL_MAP:
                label = LABEL_MAP[label] 
            else:
                continue 

            if label not in one_patient_dict[mrn]:
                one_patient_dict[mrn][label] = []
            
            one_patient_dict[mrn][label].append(event_data_path)

        for split_name in assigned_splits:
            data_dict[split_name].append(one_patient_dict)
    
    return data_dict

def main():
    parser = argparse.ArgumentParser(description="Process data and create a dataset")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to the data directory")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for train-test split")
    parser.add_argument("--test_size", type=float, default=0.15, help="Percentage size of test set (0.0-1.0)")
    parser.add_argument("--valid_size", type=float, default=0.15, help="Percentage size of validation set")
    parser.add_argument("--debug", action="store_true", help="Debugging")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for parallel processing")
    parser.add_argument("--min_sample", type=int, default=-1, help="Sample dataset")

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        dataset_dir = PATH_TO_PROCESSED_DATA

    random_state = args.random_state
    num_threads = args.num_threads
    
    path_to_X = os.path.join(dataset_dir, "X")
    if not os.path.exists(path_to_X):
        logger.error(f"Data directory not found: {path_to_X}")
        return

    mrns = sorted(os.listdir(path_to_X))

    if args.debug:
        logger.info("Running in Debug Mode")
        mrns = mrns[:10]
        
    logger.info(f"Number of Patients being processed: {len(mrns)}")

    mrn_train_full, mrn_test = train_test_split(mrns, test_size=args.test_size, random_state=random_state)
    
    relative_valid_size = args.valid_size / (1.0 - args.test_size)
    mrn_train, mrn_valid = train_test_split(mrn_train_full, test_size=relative_valid_size, random_state=random_state)

    mrn_pretrain = mrn_train

    mrn_pretrain_set = set(mrn_pretrain)
    mrn_train_set = set(mrn_train)
    mrn_valid_set = set(mrn_valid)
    mrn_test_set = set(mrn_test)

    logger.info(f"Splits - Train: {len(mrn_train)}, Valid: {len(mrn_valid)}, Test: {len(mrn_test)}")

    mrns_per_thread = np.array_split(mrns, num_threads)
    tasks = [(mrns_one_thread, dataset_dir, mrn_pretrain_set, mrn_train_set, mrn_valid_set, mrn_test_set) 
             for mrns_one_thread in mrns_per_thread]

    with multiprocessing.Pool(num_threads) as pool:
        preprocessed_results = list(pool.imap_unordered(parallel_prepare_data, tasks))

    dataset = {}
    for data_dict in preprocessed_results:
        for key, value in data_dict.items():
            if key not in dataset:
                dataset[key] = value
            else:
                dataset[key].extend(value)

    for key in dataset:
        dataset[key] = sorted(dataset[key], key=lambda x: list(x.keys())[0])

    logger.info(f"Saving dataset map to: {dataset_dir}")
    with open(os.path.join(dataset_dir, f"dataset.pickle"), 'wb') as file:
        pickle.dump(dataset, file)
        
    dataset_event = {}
    for split, split_data in tqdm(dataset.items(), total=len(dataset), desc="Generating event lists"):
        sampled_data = []
        for item in split_data:
            mrn = list(item.keys())[0]
            patient_data = item[mrn]
            for event, event_data in patient_data.items():
                if args.min_sample == -1:
                    sampled_events = event_data
                else:
                    random.seed(args.random_state)
                    sampled_events = random.sample(event_data, args.min_sample) if len(event_data) > args.min_sample else event_data
                
                sampled_events = [(path, event) for path in sampled_events]
                sampled_data.extend(sampled_events)
        
        random.seed(args.random_state)
        random.shuffle(sampled_data)
        dataset_event[split] = sampled_data

    with open(os.path.join(dataset_dir, f"dataset_events_{args.min_sample}.pickle"), 'wb') as file:
        pickle.dump(dataset_event, file)
        
    logger.info("Dataset preparation complete.")

if __name__ == "__main__":
    main()