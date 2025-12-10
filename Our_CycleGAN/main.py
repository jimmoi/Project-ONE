import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR 

from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import time
import json
import pickle
from model_training import CustomDataset, Trainer
from CycleGAN_arch import CycleGAN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_FILE = "cyclegan_checkpoint.pth"

def data_preprocessing():
    train_dataset_path = r"Our_CycleGAN\Dataset\LOL\lol_paired_with_filename.csv"
    test_dataset_path = r"Our_CycleGAN\Dataset\LOL-v2\lolv2_paired_with_filename.csv"
    
    df_train = pd.read_csv(train_dataset_path)
    df_test = pd.read_csv(test_dataset_path)
    
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    
    test_dataset = CustomDataset(df_test, transform=Trainer.get_cyclegan_transforms())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return df_train, test_loader
    

def model_evaluation():
    pass

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "cyclegan_XXX_model.pth"
    
    df_train, test_loader = data_preprocessing()
    trainer = Trainer(device = DEVICE, model_name=MODEL_NAME, model=CycleGAN)
    trainer.load_checkpoint()
    trainer.start_train(df_train)
    model_evaluation()


if __name__ == "__main__":
    main()