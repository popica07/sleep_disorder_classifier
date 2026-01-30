import pandas as pd
import json
import sys
from collections import Counter
from pathlib import Path
import os
import csv
import mne
import numpy as np
from tqdm import tqdm
from loguru import logger
import pickle
import matplotlib.pyplot as plt
from typing import Any, Union
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


def save_data(data: Any, filename: str) -> None:

    if filename.endswith(('.pkl', '.pickle')):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    elif filename.endswith('.json'):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        raise ValueError("Filename must end with .pkl, .pickle, or .json")


def load_data(filename: str) -> Any:
    if filename.endswith(('.pkl', '.pickle')):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        raise ValueError("Filename must end with .pkl, .pickle, or .json")


def getEDFFilenames(path2check):
    edfFiles = getEDFFiles(path2check)
    return [str(i) for i in edfFiles]


def getEDFFiles(path2check):
    p = Path(path2check)
    if p.is_dir():

        edfFiles = list(p.glob("**/*.[EeRr][DdEe][FfCc]"))  
        edfFiles = [edf for edf in edfFiles if not 'mslt' in edf.stem.lower()]
    else:
        print(path2check, " is not a valid directory.")
        edfFiles = []
    return edfFiles


def getSignalHeaders(edfFilename):
    try:
        try:
            raise ImportError("EdfReader not found, using MNE")
        except:
            edfR = mne.io.read_raw_edf(str(edfFilename), verbose=False)
            return edfR.ch_names
    except:
        print("Could not read headers from {}".format(edfFilename))
        return []


def getChannelLabels(edfFilename):
    channelHeaders = getSignalHeaders(edfFilename)
    try:
        return [fields["label"] for fields in channelHeaders]
    except:
        return channelHeaders


def displaySetSelection(label_set):
    numCols = 4
    curItem = 0
    width = 30
    rowStr = ""
    for label, count in sorted(label_set.items()):
        rowStr += (f"{curItem}.".ljust(4) + f"{count}".rjust(4).ljust(5) + f"{label}").ljust(width)
        curItem = curItem + 1
        if curItem % numCols == 0:
            print(rowStr)
            rowStr = ""
    if len(rowStr) > 0:
        print(rowStr)


def getAllChannelLabels(path2check):
    edfFiles = getEDFFilenames(path2check)
    num_edfs = len(edfFiles)
    if num_edfs == 0:
        label_list = []
    else:
        label_set = getLabelSet(edfFiles)
        label_list = sorted(label_set)
    return label_set, num_edfs


def getAllChannelLabelsWithCounts(edfFiles):
    num_edfs = len(edfFiles)
    if num_edfs == 0:
        label_list = []
    else:
        label_list = []
        for edfFile in tqdm(edfFiles):
            [label_list.append(l) for l in getChannelLabels(edfFile)]
        label_set_counts = Counter(label_list)
    return label_set_counts, num_edfs


def getLabelSet(edfFiles):
    label_set = set()
    for edfFile in edfFiles:
        label_set = label_set.union(set(getChannelLabels(edfFile)))
    return label_set


def read_events_file_as_df(file_name):
    df = pd.read_csv(file_name, sep='\t', header=None, encoding_errors='ignore')
    events = df[0].values
    header = events[1].split(',') 
    rows = []
    for line in events[2:]:
        rows.append(next(csv.reader([line])))
    df = pd.DataFrame(rows, columns=header)
    return df, events[0]


def get_split(file_names, test_size=0.20, random_state=1):
    train_files, test_files = train_test_split(file_names, test_size=test_size, random_state=random_state)
    split = []
    return split, train_files, test_files


def get_all_edf_and_events_file_pair(path_to_dir: str):
    temp_dict = {}
    data_dict = {}

    for file_name in os.listdir(path_to_dir):
        file_prefix = file_name.split(".")[0]
        file_suffix = file_name.split(".")[-1]
        if file_suffix.upper() not in {"EDF", "EVTS", "EVTS_HUNEO", "TXT"}:
            continue
            
        if file_prefix not in temp_dict:
            temp_dict[file_prefix] = {}
        
        if file_suffix.upper() == "EDF":
            temp_dict[file_prefix]["edf"] = os.path.join(path_to_dir, file_name)
        else:
            temp_dict[file_prefix]["evts"] = os.path.join(path_to_dir, file_name)
            
        if len(temp_dict[file_prefix]) >= 2:
            data_dict[file_prefix] = temp_dict[file_prefix]
    
    return data_dict


def filter_edf_events_file_pair(data_dict, ALL_CHANNELS, num_of_files):
    edf_events_files_pruned = []

    for id, values in tqdm(data_dict.items()):
        edf_filename = values["edf"]
        event_filename = values["evts"]
            
        try:
            edf_raw = mne.io.read_raw_edf(edf_filename, verbose=False)  
            channel_names = set(edf_raw.ch_names)
            if len(set(ALL_CHANNELS) - set(channel_names)) == 0:
                edf_events_files_pruned.append((edf_filename, event_filename))
        except:
            continue
        
        if num_of_files != -1 and len(edf_events_files_pruned) == num_of_files:
            break

    return edf_events_files_pruned


def train_model(X_train, X_test, y_train, y_test, path_to_save, class_labels, model_name="logistic", n_bootstrap=100, alpha=0.95, max_iter=100):
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    logger.info(f"Starting training model {model_name}")
    
    if model_name == "logistic":
        model = LogisticRegression(class_weight="balanced", max_iter=max_iter, solver='lbfgs')
    elif model_name == "xgb":
        model = XGBClassifier()

    model.fit(X_train, y_train)
        
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.2f}")

    class_report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(path_to_save, "confusion_matrix.png"))
    plt.close()

    row_sums = conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-10
    norm_conf_matrix = conf_matrix.astype('float') / row_sums

    plt.figure(figsize=(8, 6))
    sns.heatmap(norm_conf_matrix, annot=True, fmt=".2%", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (row-normalized)')
    plt.savefig(os.path.join(path_to_save, "confusion_matrix_normalized.png"))
    plt.close()

    n_classes = len(class_labels)
    all_auroc = []
    all_auprc = []
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        class_label = class_labels[i]
        binary_y_test = (y_test == i).astype(int)
        
        fpr[class_label], tpr[class_label], _ = roc_curve(binary_y_test, y_probs[:, i])
        roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])
        
        precision[class_label], recall[class_label], _ = precision_recall_curve(binary_y_test, y_probs[:, i])
        average_precision[class_label] = average_precision_score(binary_y_test, y_probs[:, i])

    for i in range(n_classes):
        auroc_scores = []
        auprc_scores = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_test), len(y_test), replace=True)
            binary_y_bootstrap = (y_test[indices] == i).astype(int)
            
            if len(np.unique(binary_y_bootstrap)) < 2:
                continue

            auc_score = roc_auc_score(binary_y_bootstrap, y_probs[indices, i])
            auroc_scores.append(auc_score)

            prc_score = average_precision_score(binary_y_bootstrap, y_probs[indices, i])
            auprc_scores.append(prc_score)

        auroc_scores.sort()
        auprc_scores.sort()

        if not auroc_scores:
            lower_auroc, upper_auroc = 0.0, 0.0
        else:
            lower_idx = int((1 - alpha) / 2 * len(auroc_scores))
            upper_idx = int((alpha + (1 - alpha) / 2) * len(auroc_scores))
            lower_auroc = round(auroc_scores[min(lower_idx, len(auroc_scores)-1)], 3)
            upper_auroc = round(auroc_scores[min(upper_idx, len(auroc_scores)-1)], 3)

        if not auprc_scores:
            lower_auprc, upper_auprc = 0.0, 0.0
        else:
            lower_idx = int((1 - alpha) / 2 * len(auprc_scores))
            upper_idx = int((alpha + (1 - alpha) / 2) * len(auprc_scores))
            lower_auprc = round(auprc_scores[min(lower_idx, len(auprc_scores)-1)], 3)
            upper_auprc = round(auprc_scores[min(upper_idx, len(auprc_scores)-1)], 3)

        all_auroc.append((lower_auroc, upper_auroc))
        all_auprc.append((lower_auprc, upper_auprc))

        auroc = round(roc_auc[class_labels[i]], 3)
        auprc = round(average_precision[class_labels[i]], 3)
        class_report[class_labels[i]]['auroc'] = f"{auroc} ({lower_auroc}, {upper_auroc})"
        class_report[class_labels[i]]['auprc'] = f"{auprc} ({lower_auprc}, {upper_auprc})"

    macro_auroc = np.mean([roc_auc[l] for l in class_labels])
    macro_auprc = np.mean([average_precision[l] for l in class_labels])
    
    weights = [np.sum(y_test == i) for i in range(n_classes)]
    if sum(weights) > 0:
        weighted_auroc = np.average([roc_auc[l] for l in class_labels], weights=weights)
        weighted_auprc = np.average([average_precision[l] for l in class_labels], weights=weights)
    else:
        weighted_auroc, weighted_auprc = 0.0, 0.0

    class_report['macro avg']['auroc'] = macro_auroc
    class_report['weighted avg']['auroc'] = weighted_auroc
    class_report['macro avg']['auprc'] = macro_auprc
    class_report['weighted avg']['auprc'] = weighted_auprc

    print(f"Classification Report:\n {class_report}")

    return model, y_probs, class_report