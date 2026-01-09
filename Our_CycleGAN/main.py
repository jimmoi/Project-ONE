import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os


from model_training import Trainer, Trainer_CBAM_GL, Trainer_CBAM_GL_V2
from custom_dataset import CustomDataset, CustomDataset_CBAM_GL_V2
from CycleGAN_arch import CycleGAN, CycleGAN_CBAM_GL, CycleGAN_CBAM_GL_V2
import data_preparation
from model_evaluation import evaluate_quantitative, evaluate_qualitative
from plot_metric_history import plot_metric_history

from experiment_manager import EXPERIMENT_MANAGER

def data_preprocessing():
    train_dataset_path = r"Our_CycleGAN\Dataset\LOL\lol_paired_with_filename.csv"
    test_dataset_path = r"Our_CycleGAN\Dataset\LOL-v2\lolv2_paired_with_filename.csv"
    
    if not os.path.exists(train_dataset_path) or not os.path.exists(test_dataset_path):
        data_preparation.main(EXPERIMENT_MANAGER.data_path)
        if not os.path.exists(train_dataset_path) or not os.path.exists(test_dataset_path):
            raise ValueError("Dataset not found.")
    
    df_train = pd.read_csv(train_dataset_path)
    df_test = pd.read_csv(test_dataset_path)
    
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    
    test_dataset = EXPERIMENT_MANAGER.dataset(df_test, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # !! Caution: fix batch_size = 1

    return df_train, test_loader
    
def model_evaluation(test_loader, n_sample=7):
    def denormalize_image_tensor(image_tensor):
        """
        Denormalizes the tensor output (C, H, W) from [-1, 1] to [0, 1] and plots it.
        """
        image_tensor = image_tensor * 0.5 + 0.5
        image_tensor = image_tensor.clamp(0.0, 1.0)
        return image_tensor
    
    #--------------------------
    # Model Loading
    #--------------------------
    model, load_model_status = EXPERIMENT_MANAGER.load_model()
    if not load_model_status:
        raise ValueError("Model not found.")
    

    np.random.seed(EXPERIMENT_MANAGER.seed)
    random_idx = np.random.choice(len(test_loader), n_sample, replace=False).tolist()
    np.random.seed()
    compare_image = torch.zeros((len(random_idx), 3, 3, 256, 256))
    
    metrics = []
    
    print("Start inference and evaluation...")
    iter_round = 0
    progress_bar = tqdm(test_loader)
    for i, (low_light_img, normal_light_img) in enumerate(progress_bar):
        low_light_img_device = low_light_img.to(EXPERIMENT_MANAGER.device)  
        generated_img = model.G_A2B(low_light_img_device)
        generated_img = generated_img.detach().cpu().clone()
        
        # Denormalize
        low_light_img = denormalize_image_tensor(low_light_img)
        normal_light_img = denormalize_image_tensor(normal_light_img)
        generated_img = denormalize_image_tensor(generated_img)
        
        #--------------------------
        # Quantitative Evaluation
        #--------------------------
        metrics.append(evaluate_quantitative(generated_img, normal_light_img, EXPERIMENT_MANAGER.device))
        
        if i in random_idx:
            compare_image[iter_round] = torch.cat([low_light_img, normal_light_img, generated_img], dim=0)
            iter_round += 1
            
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(EXPERIMENT_MANAGER.quantitative_metrics_path, index=False)
    metric_dict = metrics.mean().to_dict()
    with open(EXPERIMENT_MANAGER.agg_quantitative_metrics_path, "w") as f:
        json.dump(metric_dict, f)  
    #--------------------------
    # Qualitative Evaluation
    #--------------------------
    evaluate_qualitative(compare_image, EXPERIMENT_MANAGER.qualitative_metrics_path)
    print("Evaluation finished.")
    
def model_training(df_train):
    EXPERIMENT_MANAGER.trainer.load_checkpoint()
    EXPERIMENT_MANAGER.trainer.start_train(df_train)
    
def plot_log_history():
    def get_metric_history():
        collect_metric_history = []
        metric_root = EXPERIMENT_MANAGER.curr_dir
        for history_file in os.listdir(EXPERIMENT_MANAGER.log_dir):
            if history_file.endswith(".json") and history_file.startswith("Training_log"):
                file_path = os.path.join(EXPERIMENT_MANAGER.log_dir, history_file)
                with open(file_path,"r") as f:
                    data = json.load(f)
                    collect_metric_history.append(data)
                    
        if not collect_metric_history:
            raise ValueError("No metric history found.")
                    
        data = []
        for step in collect_metric_history:
            for train_it in step:
                row_temp = train_it["metrics"].copy()
                row_temp["epoch"] = train_it["epoch"]
                row_temp["it"] = train_it["iteration"]
                data.append(row_temp)
                
        return data
    
    data = get_metric_history()
    metric_df = pd.DataFrame(data)
    plot_metric_history(metric_df, n_x_ticks=50)
    
def main():
    
    #--------------------------
    # Experiment Setup
    #--------------------------
    EXPERIMENT_MANAGER.set_experiment_name("cyclegan_CBAM_GL_V2_200_patch64_local8", model=CycleGAN_CBAM_GL_V2)
    EXPERIMENT_MANAGER.setup_experiment()
    EXPERIMENT_MANAGER.setup_dataset(CustomDataset_CBAM_GL_V2.setup_dataset(EXPERIMENT_MANAGER.model.get_image_transforms(), patch_size=64, local_sample_n=16))
    EXPERIMENT_MANAGER.setup_trainer(Trainer_CBAM_GL_V2(model=EXPERIMENT_MANAGER.model, n_epochs=200, history_step=10))
    #--------------------------
    # Data Preprocessing
    #--------------------------
    df_train, test_loader = data_preprocessing()
    
    #--------------------------
    # Tensorboard
    #--------------------------
    EXPERIMENT_MANAGER.launch_tensorboard()
    
    #--------------------------
    # Model Training
    #--------------------------
    model_training(df_train=df_train)
    
    
    #--------------------------
    # Plot Log History
    #--------------------------
    plot_log_history()
    
    #--------------------------
    # Model Evaluation
    #--------------------------
    model_evaluation(test_loader=test_loader)
    
    #--------------------------
    # Shutdown PC
    #--------------------------
    EXPERIMENT_MANAGER.shutdown_pc()
    

if __name__ == "__main__":
    main()