import pandas as pd
from tqdm import tqdm
import pickle
import os
import torch
from loguru import logger
import matplotlib.pyplot as plt
import argparse
import numpy as np
from collections import Counter

import sys
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score

import sys
sys.path.append("model")

import models
import config
from config import (MODALITY_TYPES, CLASS_LABELS, 
                    LABELS_DICT, PATH_TO_PROCESSED_DATA)
from utils import train_model
from dataset import EventDataset as Dataset 


def main(args):

    dataset_dir = args.dataset_dir

    if dataset_dir == None:
        dataset_dir = PATH_TO_PROCESSED_DATA

    output_file = args.output_file
    path_to_output = os.path.join(dataset_dir, f"{output_file}")
    
    modality_type = args.modality_type
    num_per_event = args.num_per_event
    model_name = args.model_name

    path_to_figures = os.path.join(path_to_output, f"figures")
    path_to_models = os.path.join(path_to_output, f"models")
    path_to_probs = os.path.join(path_to_output, f"probs")

    os.makedirs(path_to_figures, exist_ok=True)
    os.makedirs(path_to_models, exist_ok=True)
    os.makedirs(path_to_probs, exist_ok=True)
    
    dataset_prefix = "dataset_events_-1" 
    path_to_eval_data = os.path.join(path_to_output, "eval_data")

    def load_emb_y(split):
        emb_path = os.path.join(path_to_eval_data, f"{split}_{dataset_prefix}_emb.pickle")
        y_path = os.path.join(path_to_eval_data, f"{split}_{dataset_prefix}_y.pickle")
        
        if not os.path.exists(emb_path):
            logger.error(f"Embedding file not found: {emb_path}")
            return None, None

        with open(emb_path, 'rb') as f:
            emb = pickle.load(f)
        with open(y_path, 'rb') as f:
            y = pickle.load(f)
            
        if modality_type == "respiratory":
            X = emb[0]
        elif modality_type == "sleep_stages":
            X = emb[1]
        elif modality_type == "ekg":
            X = emb[2]
        elif modality_type == "combined":
            X = torch.cat(emb, dim=1)
            
        return X.numpy(), y.numpy()

    logger.info("Loading Training Data...")
    X_train, y_train = load_emb_y("train")
    
    logger.info("Loading Test Data...")
    X_test, y_test = load_emb_y("test")
    
    if X_train is None or X_test is None:
        logger.error("Failed to load data. Ensure Step 3 completed successfully.")
        return

    if num_per_event != -1:
        X_train = X_train[:num_per_event]
        y_train = y_train[:num_per_event]

    path_to_save = path_to_figures
    
    logger.info(f"Training Classifier ({model_name}) on {modality_type} embeddings...")
    logger.info(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    model, y_probs, class_report = train_model(X_train, X_test, y_train, y_test, path_to_save, 
                list(CLASS_LABELS), model_name=model_name, max_iter=args.max_iter)

    logger.info(f"Saving model...")
    with open(os.path.join(path_to_models, f"{modality_type}_model.pickle"), 'wb') as file:
        pickle.dump(model, file)

    logger.info(f"Saving probabilities...")
    with open(os.path.join(path_to_probs, f"{modality_type}_y_probs.pickle"), 'wb') as file:
        pickle.dump(y_probs, file)

    logger.info(f"Saving class report...")
    with open(os.path.join(path_to_probs, f"{modality_type}_class_report.pickle"), 'wb') as file:
        pickle.dump(class_report, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process log files and generate plots.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file name (directory inside processed)")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to preprocessed data")
    parser.add_argument("--modality_type", type=str, help="Target Types", choices=["respiratory", "sleep_stages", "ekg", "combined"], default="combined")
    parser.add_argument("--num_per_event", type=int, default=-1, help="Number of events from start")
    parser.add_argument("--model_name", type=str, default="logistic", help="Model name")
    parser.add_argument("--max_iter", type=int, default=10000, help="Max iterations for logistic regression")

    args = parser.parse_args()
    main(args)