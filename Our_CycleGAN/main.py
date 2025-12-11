import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from model_training import CustomDataset, Trainer
from CycleGAN_arch import CycleGAN
from model_evaluation import evaluate_quantitative

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
    model = CycleGAN(device, only_G_A2B=True)
    model.model_init()
    model.load_model(model_path = model_path)
    model.eval()
    
    random_idx = np.random.choice(len(test_loader), n_sample, replace=False).tolist()
    compare_image = torch.zeros((len(random_idx), 3, 3, 256, 256))
    
    print("Start inference...")
    iter_round = 0
    for i, (low_light_img, normal_light_img) in enumerate(test_loader):
        if i in random_idx:
            random_idx.remove(i)
            low_light_img_device = low_light_img.to(device)  
            predicted_img = model.G_A2B(low_light_img_device)
            predicted_img = predicted_img.detach().cpu().clone()
            compare_image[iter_round] = torch.cat([low_light_img, normal_light_img, predicted_img], dim=0)
            if len(random_idx) == 0:
                break
            iter_round += 1
            
    print("Start evaluation...")
    evaluate_quantitative(compare_image)
    print("End evaluation...")

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