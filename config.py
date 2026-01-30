import configparser
import os
import types

_FILENAME = None
_PARAM = {}

CONFIG = types.SimpleNamespace(
    FILENAME=_FILENAME,
    DATASETS=_PARAM.get("datasets", "datasets"),
    OUTPUT=_PARAM.get("output", "output"),
    CACHE=_PARAM.get("cache", ".cache"),
)

PATH_TO_RAW_DATA = "/content/raw"
PATH_TO_PROCESSED_DATA = "/content/processed"


LABELS_DICT = {
    "Wake": 0, 
    "Stage 1": 1, 
    "Stage 2": 2, 
    "Stage 3": 3, 
    "REM": 4
}

MODALITY_TYPES = ["respiratory", "sleep_stages", "ekg"]
CLASS_LABELS = ["Wake", "Stage 1", "Stage 2", "Stage 3", "REM"]
NUM_CLASSES = 5

EVENT_TO_ID = {
    "Wake": 1, 
     "Stage 1": 2, 
     "Stage 2": 3, 
     "Stage 3": 4, 
     "Stage 4": 4, 
     "REM": 5,
}

LABEL_MAP = {
    "Sleep stage W": "Wake", 
    "Sleep stage N1": "Stage 1", 
    "Sleep stage N2": "Stage 2", 
    "Sleep stage N3": "Stage 3", 
    "Sleep stage R": "REM", 
    "W": "Wake", 
    "N1": "Stage 1", 
    "N2": "Stage 2", 
    "N3": "Stage 3", 
    "REM": "REM", 
    "wake": "Wake", 
    "nonrem1": "Stage 1", 
    "nonrem2": "Stage 2", 
    "nonrem3": "Stage 3", 
    "rem": "REM", 
    "SLEEP-S0": "Wake",
    "SLEEP-S1": "Stage 1",
    "SLEEP-S2": "Stage 2",
    "SLEEP-S3": "Stage 3",
    "SLEEP-S4": "Stage 3",
    "SLEEP-REM": "REM"
}


ALL_CHANNELS = [
    'RESP_PLETH_1',   
    'RESP_SaO2',       
    'RESP_PLETH_2',   
    'EEG_F4-C4',      
    'EEG_C4-A1',       
    'EEG_C4-P4',       
    'EEG_P4-O2',       
    'EOG_ROC-LOC',     
    'ECG'              
]

CHANNEL_DATA = {
    "Respiratory": ['RESP_PLETH_1', 'RESP_SaO2', 'RESP_PLETH_2'],

    "Sleep_Stages": ['EEG_F4-C4', 'EEG_C4-A1', 'EEG_C4-P4', 'EEG_P4-O2', 'EOG_ROC-LOC'],
    "EKG": ["ECG"], 
}

CHANNEL_DATA_IDS = {
    "Respiratory": [ALL_CHANNELS.index(item) for item in CHANNEL_DATA["Respiratory"]], 
    "Sleep_Stages": [ALL_CHANNELS.index(item) for item in CHANNEL_DATA["Sleep_Stages"]], 
    "EKG": [ALL_CHANNELS.index(item) for item in CHANNEL_DATA["EKG"]], 
}