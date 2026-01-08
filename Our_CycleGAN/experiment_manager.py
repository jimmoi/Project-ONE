import shutil
import os
import torch
from CycleGAN_arch import CycleGAN
from torch.utils.tensorboard import SummaryWriter

class ExperimentManager:
    experiment_dir = "Our_CycleGAN\experiments"
    model_dir = "Our_CycleGAN\models"
    checkpoint_file = "cyclegan_checkpoint.pth"
    best_checkpoint_file = "cyclegan_best_checkpoint.pth"
    
    quantitative_metrics_file = "quantitative_metrics.csv"
    agg_quantitative_metrics_file = "agg_quantitative_metrics.json"
    qualitative_metrics_file = "qualitative_metrics.png"
    
    log_history_file = "history.png"
    
    def __init__(self, data_path, tensorboard=True):
        self.experiment_name = None
        self.curr_dir = None
        self.full_model_path = None
        self.checkpoint_path = None
        self.best_checkpoint_path = None
        self.quantitative_metrics_path = None
        self.agg_quantitative_metrics_path = None
        self.qualitative_metrics_path = None
        self.log_history_path = None
        self.model = None
        self.verbose_tensorboard = tensorboard
        self.tensorboard_dir = None
        self.tensorboard_writer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = data_path
        self.seed = 42
            
    def setup_experiment(self):
        if self.experiment_name is None:
            raise ValueError("Experiment name not set.")
        
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.create_experiment_dir()
        
    def create_experiment_dir(self):
        try:
            os.makedirs(self.curr_dir)
            
        except FileExistsError:
            print(f"Experiment directory {self.curr_dir} already exists.")
        finally:
            if self.verbose_tensorboard:
                os.makedirs(self.tensorboard_dir, exist_ok=True)
                self.create_tensorboard_writer()
            
    def create_tensorboard_writer(self):
        self.tensorboard_writer = SummaryWriter(os.path.join(self.tensorboard_dir))
        
        tensorboard_bat_path = os.path.join(self.curr_dir, "tensorboard_run.bat")
        with open(tensorboard_bat_path, "w") as f:
            text = f"""
            cd /d %~dp0

            :: Use 'call' to ensure the script continues after activation
            :: activate conda with torch environment
            call conda activate proj

            tensorboard --logdir=runs
            """
            
            f.write(text)
    
    def load_model(self):
        model = self.model(self.device, only_G_A2B=True)
        model.model_init()
        model_path = self.full_model_path if os.path.exists(self.full_model_path) else self.best_checkpoint_path
        status = model.load_model(model_path = model_path)
        model.eval()
        return model, status
    
    def set_experiment_name(self, experiment_name, model):
        self.experiment_name = experiment_name
        self.curr_dir = os.path.join(self.experiment_dir, experiment_name)
        self.full_model_path = os.path.join(self.model_dir, self.experiment_name + "_model.pth")
        self.checkpoint_path = os.path.join(self.curr_dir, self.checkpoint_file)
        self.best_checkpoint_path = os.path.join(self.curr_dir, self.best_checkpoint_file)
        self.quantitative_metrics_path = os.path.join(self.curr_dir, self.quantitative_metrics_file)
        self.agg_quantitative_metrics_path = os.path.join(self.curr_dir, self.agg_quantitative_metrics_file)
        self.qualitative_metrics_path = os.path.join(self.curr_dir, self.qualitative_metrics_file)
        self.log_history_path = os.path.join(self.curr_dir, self.log_history_file)
        self.model = model
        self.tensorboard_dir = os.path.join(self.curr_dir, "runs")

data_path = {
        "lol":"Our_CycleGAN\Dataset\LOL",
        "lolv2":"Our_CycleGAN\Dataset\LOL-v2"
    }

EXPERIMENT_MANAGER = ExperimentManager(data_path, tensorboard = True)