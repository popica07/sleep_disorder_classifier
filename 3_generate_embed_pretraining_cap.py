import time
import torch
import torchvision
import os
import click
import tqdm
import math
import shutil
import datetime
import numpy as np
from loguru import logger
import pickle
import sys

sys.path.append("model") 

import models
from config import (CONFIG, CHANNEL_DATA, 
                    ALL_CHANNELS, CHANNEL_DATA_IDS, 
                    PATH_TO_PROCESSED_DATA)

from dataset import EventDataset as Dataset 

@click.command("generate_eval_embed")
@click.argument("output_file", type=click.Path())
@click.option("--dataset_dir", type=str, default=None)
@click.option("--dataset_file", type=str, default="dataset_events_-1.pickle")
@click.option("--batch_size", type=int, default=32)
@click.option("--num_workers", type=int, default=2)
@click.option("--splits", type=click.STRING, default="train,valid,test", help='Specify the data splits (train, valid, test).')
def generate_eval_embed(
    output_file,
    dataset_dir,
    dataset_file,
    batch_size,
    num_workers,
    splits
):
    if dataset_dir is None:
        dataset_dir = PATH_TO_PROCESSED_DATA

    output_dir = os.path.join(dataset_dir, f"{output_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    if isinstance(splits, str):
        if splits.strip().startswith("["):
            import ast
            try:
                splits = ast.literal_eval(splits)
            except (ValueError, SyntaxError):
                splits = splits.replace("[", "").replace("]", "").replace("'", "").replace('"', "").split(",")
        else:
            splits = splits.split(",")
    
    splits = [s.strip() for s in splits if s.strip()]

    path_to_data = dataset_dir

    dataset = {
        split: Dataset(os.path.join(path_to_data, dataset_file), split=split)
        for split in splits
    }

    model_resp = models.EffNet(
        in_channel=len(CHANNEL_DATA_IDS["Respiratory"]), 
        stride=2, 
        dilation=1
    )

    model_resp.fc = torch.nn.Identity()
    model_resp.to(device)
 
    model_sleep = models.EffNet(
        in_channel=len(CHANNEL_DATA_IDS["Sleep_Stages"]), 
        stride=2, 
        dilation=1
    )
    model_sleep.fc = torch.nn.Identity()
    model_sleep.to(device)
    
    model_ekg = models.EffNet(
        in_channel=len(CHANNEL_DATA_IDS["EKG"]), 
        stride=2, 
        dilation=1
    )
    model_ekg.fc = torch.nn.Identity()
    model_ekg.to(device)

    checkpoint_path = os.path.join(output_dir, "best.pt")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    def load_state_dict_safe(model, state_dict):
        """Helper to load state dict ignoring 'module.' prefix and shape mismatches."""
        model_state = model.state_dict()
        new_state_dict = {}
        
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            
            if name in model_state:
                if model_state[name].shape == v.shape:
                    new_state_dict[name] = v
                else:
                    logger.warning(f"Skipping layer {name}: Shape mismatch {model_state[name].shape} vs {v.shape}")
            else:
                pass 
                
        model.load_state_dict(new_state_dict, strict=False)


    if "respiratory_state_dict" in checkpoint:
        load_state_dict_safe(model_resp, checkpoint["respiratory_state_dict"])
    elif "resp_state_dict" in checkpoint:
        load_state_dict_safe(model_resp, checkpoint["resp_state_dict"])
    else:
        logger.warning("No respiratory state dict found in checkpoint!")

    if "sleep_stages_state_dict" in checkpoint:
        load_state_dict_safe(model_sleep, checkpoint["sleep_stages_state_dict"])
    elif "sleep_state_dict" in checkpoint:
        load_state_dict_safe(model_sleep, checkpoint["sleep_state_dict"])
    else:
        logger.warning("No sleep state dict found in checkpoint!")

    if "ekg_state_dict" in checkpoint:
        load_state_dict_safe(model_ekg, checkpoint["ekg_state_dict"])
    else:
        logger.warning("No EKG state dict found in checkpoint!")

    model_resp.eval()
    model_sleep.eval()
    model_ekg.eval()

    path_to_save = os.path.join(output_dir, "eval_data")
    os.makedirs(path_to_save, exist_ok=True)

    for split in splits:
        if split not in dataset or len(dataset[split]) == 0:
            logger.warning(f"Skipping empty or missing split: {split}")
            continue

        dataloader = torch.utils.data.DataLoader(
            dataset[split], 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=False, 
            drop_last=False
        )
        
        emb = [[], [], []]
        y = []
        
        with torch.no_grad():
            with tqdm.tqdm(total=len(dataloader), desc=("Embeddings for " + split)) as pbar:
                for (i, (resp, sleep, ekg, label)) in enumerate(dataloader):
                    resp = resp.to(device, dtype=torch.float)
                    sleep = sleep.to(device, dtype=torch.float)
                    ekg = ekg.to(device, dtype=torch.float)

                    feat_resp = model_resp(resp)
                    feat_sleep = model_sleep(sleep)
                    feat_ekg = model_ekg(ekg)

                    emb[0].append(torch.nn.functional.normalize(feat_resp).detach().cpu())
                    emb[1].append(torch.nn.functional.normalize(feat_sleep).detach().cpu())
                    emb[2].append(torch.nn.functional.normalize(feat_ekg).detach().cpu())
                    
                    y.append(label)

                    pbar.update()
        
        if len(emb[0]) > 0:
            emb = list(map(torch.cat, emb))
            y = torch.cat(y)
            
            dataset_prefix = dataset_file.split(".")[0]
            
            save_emb_path = os.path.join(path_to_save, f"{split}_{dataset_prefix}_emb.pickle")
            with open(save_emb_path, 'wb') as f:
                pickle.dump(emb, f)
                
            save_y_path = os.path.join(path_to_save, f"{split}_{dataset_prefix}_y.pickle")
            with open(save_y_path, 'wb') as f:
                pickle.dump(y, f)
                
            logger.info(f"Saved embeddings for {split} to {save_emb_path}")
        else:
            logger.warning(f"No data processed for split {split}")

if __name__ == "__main__":
    generate_eval_embed()