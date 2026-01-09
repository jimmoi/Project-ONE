import shutil
import os
import platform
import subprocess
import webbrowser
import time
import torch
from CycleGAN_arch import CycleGAN
from torch.utils.tensorboard import SummaryWriter

class ExperimentManager:
    """ExperimentManager class for managing experiments.
    usage example:
    EXPERIMENT_MANAGER = ExperimentManager(data_path, tensorboard = True)
    EXPERIMENT_MANAGER.set_experiment_name("cyclegan_CBAM_GL_V2_200_patch64_local8", model=CycleGAN_CBAM_GL_V2)
    EXPERIMENT_MANAGER.setup_experiment()
    EXPERIMENT_MANAGER.setup_dataset(CustomDataset_CBAM_GL_V2.setup_dataset(EXPERIMENT_MANAGER.model.get_image_transforms(), patch_size=64, local_sample_n=16))
    EXPERIMENT_MANAGER.setup_trainer(Trainer_CBAM_GL_V2(model=EXPERIMENT_MANAGER.model, n_epochs=200, history_step=10))
    """
    
    all_experiment_dir = "Our_CycleGAN\experiments"
    all_model_dir = "Our_CycleGAN\models"
    
    visualization_dir = "visualization"
    log_dir = "log"
    
    checkpoint_file = "cyclegan_checkpoint.pth"
    best_checkpoint_file = "cyclegan_best_checkpoint.pth"
    quantitative_metrics_file = "quantitative_metrics.csv"
    agg_quantitative_metrics_file = "agg_quantitative_metrics.json"
    qualitative_metrics_file = "qualitative_metrics.png"
    
    log_history_file = "history.png"
    
    def __init__(self, data_path, tensorboard=True):
        self.trainer = None
        self.dataset = None
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
        
        os.makedirs(self.all_experiment_dir, exist_ok=True)
        os.makedirs(self.all_model_dir, exist_ok=True)
        self.create_experiment_dir()
        
    def setup_dataset(self, dataset):
        self.dataset = dataset
        
    def setup_trainer(self, trainer):
        self.trainer = trainer
        
    def create_experiment_dir(self):
        try:
            os.makedirs(self.curr_dir)
            os.makedirs(self.visualization_dir)
            os.makedirs(self.log_dir)
            
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
        
        # directory in each experiment
        self.curr_dir = os.path.join(self.all_experiment_dir, experiment_name)
        self.visualization_dir = os.path.join(self.curr_dir, self.visualization_dir)
        self.log_dir = os.path.join(self.curr_dir, self.log_dir)
        
        # path to model file in each experiment
        self.full_model_path = os.path.join(self.all_model_dir, self.experiment_name + "_model.pth")
        self.checkpoint_path = os.path.join(self.curr_dir, self.checkpoint_file)
        self.best_checkpoint_path = os.path.join(self.curr_dir, self.best_checkpoint_file)
        
        # metrics file path in each experiment
        self.quantitative_metrics_path = os.path.join(self.visualization_dir, self.quantitative_metrics_file)
        self.agg_quantitative_metrics_path = os.path.join(self.visualization_dir, self.agg_quantitative_metrics_file)
        self.qualitative_metrics_path = os.path.join(self.visualization_dir, self.qualitative_metrics_file)
        self.log_history_path = os.path.join(self.visualization_dir, self.log_history_file)
        self.model = model
        
        # directory to tensorboard in each experiment
        self.tensorboard_dir = os.path.join(self.curr_dir, "runs")
        
    def launch_tensorboard(self, port=6006):
        logdir = os.path.abspath(self.tensorboard_dir)
        url = f"http://localhost:{port}"
    
        # Base command for TensorBoard
        tb_cmd = f"python -m tensorboard.main --logdir \"{logdir}\" --port {port}"

        print(f"--- Starting TensorBoard in a new terminal ---")
        print(f"--- Monitoring: {logdir} ---")

        system = platform.system()
        
        if system == "Windows":
            # 'start' opens a new cmd window
            subprocess.Popen(f"start cmd /k {tb_cmd}", shell=True)
            
        elif system == "Darwin":  # macOS
            # Uses AppleScript to open Terminal and run the command
            osascript = f'tell application "Terminal" to do script "{tb_cmd}"'
            subprocess.Popen(["osascript", "-e", osascript])
            
        else:  # Linux (requires gnome-terminal, xterm, etc.)
            # gnome-terminal is standard on Ubuntu
            subprocess.Popen(["gnome-terminal", "--", "bash", "-c", f"{tb_cmd}; exec bash"])

        # Wait for the server to spin up and open browser
        time.sleep(3)
        webbrowser.open(url)
        print(f"Browser opened to {url}. You can continue using this script.")

data_path = {
        "lol":"Our_CycleGAN\Dataset\LOL",
        "lolv2":"Our_CycleGAN\Dataset\LOL-v2"
    }

EXPERIMENT_MANAGER = ExperimentManager(data_path, tensorboard = True)