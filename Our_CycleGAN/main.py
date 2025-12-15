import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from model_training import CustomDataset, Trainer
from CycleGAN_arch import CycleGAN
from model_evaluation import evaluate_quantitative, evaluate_qualitative
from tqdm import tqdm
import json
import os

class ExperimentManager:
    experiment_dir = "Our_CycleGAN\experiments"
    model_dir = "Our_CycleGAN\models"
    checkpoint_file = "cyclegan_checkpoint.pth"
    best_checkpoint_file = "cyclegan_best_checkpoint.pth"
    
    quantitative_metrics_file = "quantitative_metrics.csv"
    agg_quantitative_metrics_file = "agg_quantitative_metrics.json"
    qualitative_metrics_file = "qualitative_metrics.png"
    
    log_history_file = "history.png"
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.curr_dir = os.path.join(self.experiment_dir, experiment_name)
        self.full_model_path = os.path.join(self.model_dir, self.experiment_name + "_model.pth")
        self.checkpoint_path = os.path.join(self.curr_dir, self.checkpoint_file)
        self.best_checkpoint_path = os.path.join(self.curr_dir, self.best_checkpoint_file)
        self.quantitative_metrics_path = os.path.join(self.curr_dir, self.quantitative_metrics_file)
        self.agg_quantitative_metrics_path = os.path.join(self.curr_dir, self.agg_quantitative_metrics_file)
        self.qualitative_metrics_path = os.path.join(self.curr_dir, self.qualitative_metrics_file)
        self.log_history_path = os.path.join(self.curr_dir, self.log_history_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup_experiment(self):
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.create_experiment_dir()
        
    def create_experiment_dir(self):
        try:
            os.makedirs(self.curr_dir)
        except FileExistsError:
            print(f"Experiment directory {self.curr_dir} already exists.")
    
    def load_model(self):
        model = CycleGAN(self.device, only_G_A2B=True)
        model.model_init()
        model_path = self.full_model_path if os.path.exists(self.full_model_path) else self.best_checkpoint_path
        status = model.load_model(model_path = model_path)
        model.eval()
        return model, status

def data_preprocessing():
    train_dataset_path = r"Our_CycleGAN\Dataset\LOL\lol_paired_with_filename.csv"
    test_dataset_path = r"Our_CycleGAN\Dataset\LOL-v2\lolv2_paired_with_filename.csv"
    
    df_train = pd.read_csv(train_dataset_path)
    df_test = pd.read_csv(test_dataset_path)
    
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    
    test_dataset = CustomDataset(df_test, transform=CycleGAN.get_image_transforms())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # !! Caution: fix batch_size = 1

    return df_train, test_loader
    
def model_evaluation(test_loader, experiment_manager, n_sample=7):
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
    model, load_model_status = experiment_manager.load_model()
    if not load_model_status:
        raise ValueError("Model not found.")
    

    
    random_idx = np.random.choice(len(test_loader), n_sample, replace=False).tolist()
    compare_image = torch.zeros((len(random_idx), 3, 3, 256, 256))
    
    metrics = []
    
    print("Start inference and evaluation...")
    iter_round = 0
    progress_bar = tqdm(test_loader)
    for i, (low_light_img, normal_light_img) in enumerate(progress_bar):
        low_light_img_device = low_light_img.to(experiment_manager.device)  
        generated_img = model.G_A2B(low_light_img_device)
        generated_img = generated_img.detach().cpu().clone()
        
        # Denormalize
        low_light_img = denormalize_image_tensor(low_light_img)
        normal_light_img = denormalize_image_tensor(normal_light_img)
        generated_img = denormalize_image_tensor(generated_img)
        
        #--------------------------
        # Quantitative Evaluation
        #--------------------------
        metrics.append(evaluate_quantitative(generated_img, normal_light_img, experiment_manager.device))
        
        if i in random_idx:
            compare_image[iter_round] = torch.cat([low_light_img, normal_light_img, generated_img], dim=0)
            iter_round += 1
            
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(experiment_manager.quantitative_metrics_path, index=False)
    metric_dict = metrics.mean().to_dict()
    with open(experiment_manager.agg_quantitative_metrics_path, "w") as f:
        json.dump(metric_dict, f)  
    #--------------------------
    # Qualitative Evaluation
    #--------------------------
    evaluate_qualitative(compare_image, experiment_manager.qualitative_metrics_path)
    print("Evaluation finished.")
    
def model_training(df_train, experiment_manager):
    trainer = Trainer(experiment_manager=experiment_manager, model=CycleGAN)
    trainer.load_checkpoint()
    trainer.start_train(df_train)

def main():
    
    #--------------------------
    # Experiment Setup
    #--------------------------
    experiment_manager = ExperimentManager(experiment_name="cyclegan_XXX")
    experiment_manager.setup_experiment()
    #--------------------------
    # Data Preprocessing
    #--------------------------
    df_train, test_loader = data_preprocessing()
    
    #--------------------------
    # Model Training
    #--------------------------
    model_training(df_train=df_train, experiment_manager=experiment_manager)
    
    #--------------------------
    # Model Evaluation
    #--------------------------
    model_evaluation(test_loader=test_loader, experiment_manager=experiment_manager)


if __name__ == "__main__":
    main()