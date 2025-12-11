import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from model_training import CustomDataset, Trainer
from CycleGAN_arch import CycleGAN
from model_evaluation import evaluate_quantitative, evaluate_qualitative
from tqdm import tqdm
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_FILE = "cyclegan_checkpoint.pth"

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
    

def model_evaluation(test_loader, device, model_path, n_sample=7):
    def denormalize_image_tensor(image_tensor):
        """
        Denormalizes the tensor output (C, H, W) from [-1, 1] to [0, 1] and plots it.
        """
        image_tensor = image_tensor * 0.5 + 0.5
        image_tensor = image_tensor.clamp(0.0, 1.0)
        return image_tensor
    
    
    model = CycleGAN(device, only_G_A2B=True)
    model.model_init()
    model.load_model(model_path = model_path)
    model.eval()
    

    
    random_idx = np.random.choice(len(test_loader), n_sample, replace=False).tolist()
    compare_image = torch.zeros((len(random_idx), 3, 3, 256, 256))
    
    metrics = []
    
    print("Start inference and evaluation...")
    iter_round = 0
    progress_bar = tqdm(test_loader)
    for i, (low_light_img, normal_light_img) in enumerate(progress_bar):
        low_light_img_device = low_light_img.to(device)  
        generated_img = model.G_A2B(low_light_img_device)
        generated_img = generated_img.detach().cpu().clone()
        
        # Denormalize
        low_light_img = denormalize_image_tensor(low_light_img)
        normal_light_img = denormalize_image_tensor(normal_light_img)
        generated_img = denormalize_image_tensor(generated_img)
        
        #--------------------------
        # Quantitative Evaluation
        #--------------------------
        metrics.append(evaluate_quantitative(generated_img, normal_light_img, device))
        
        if i in random_idx:
            compare_image[iter_round] = torch.cat([low_light_img, normal_light_img, generated_img], dim=0)
            iter_round += 1
            
    metrics = pd.DataFrame(metrics)
    metrics.to_csv("Our_CycleGAN/metrics.csv", index=False)
    metric_dict = metrics.mean().to_dict()
    with open("Our_CycleGAN/metric_dict.json", "w") as f:
        json.dump(metric_dict, f)
    
            
    #--------------------------
    # Qualitative Evaluation
    #--------------------------
    evaluate_qualitative(compare_image)
    print("Evaluation finished.")
    

    

def model_training(df_train, device):
    trainer = Trainer(device = device, model_name=MODEL_NAME, model=CycleGAN)
    trainer.load_checkpoint()
    trainer.start_train(df_train)

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "cyclegan_XXX_model.pth"
    MODEL_PATH = r"Our_CycleGAN\models\trained_cyclegan.pth"
    
    df_train, test_loader = data_preprocessing()
    # model_training(df_train=df_train, device=DEVICE)
    model_evaluation(test_loader=test_loader, device=DEVICE, model_path=MODEL_PATH)


if __name__ == "__main__":
    main()