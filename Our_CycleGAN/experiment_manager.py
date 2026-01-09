import shutil
import torch
import platform
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import threading
import subprocess
import webbrowser
import time

class ExperimentManager:
    """ExperimentManager class for managing experiments.
    usage example:
    EXPERIMENT_MANAGER = ExperimentManager(data_path, tensorboard = True)
    EXPERIMENT_MANAGER.set_experiment_name("cyclegan_CBAM_GL_V2_200_patch64_local8", model=CycleGAN_CBAM_GL_V2)
    EXPERIMENT_MANAGER.setup_experiment()
    EXPERIMENT_MANAGER.setup_dataset(CustomDataset_CBAM_GL_V2.setup_dataset(EXPERIMENT_MANAGER.model.get_image_transforms(), patch_size=64, local_sample_n=16))
    EXPERIMENT_MANAGER.setup_trainer(Trainer_CBAM_GL_V2(model=EXPERIMENT_MANAGER.model, n_epochs=200, history_step=10))
    """
    
    # Use Path objects for base directories
    all_experiment_dir = Path("Our_CycleGAN/experiments")
    all_model_dir = Path("Our_CycleGAN/models")
    
    # Constants for sub-folder and file names
    visualization_subdir = "visualization"
    log_subdir = "log"
    
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
        
        # pathlib.mkdir handles nested creation easily
        self.all_experiment_dir.mkdir(parents=True, exist_ok=True)
        self.all_model_dir.mkdir(parents=True, exist_ok=True)
        self.create_experiment_dir()
        
    def setup_dataset(self, dataset):
        self.dataset = dataset
        
    def setup_trainer(self, trainer):
        self.trainer = trainer
        
    def create_experiment_dir(self):
        # Create main experiment folder and subfolders
        # parents=True ensures the whole tree is built
        self.curr_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose_tensorboard:
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.create_tensorboard_writer()
            
    def create_tensorboard_writer(self):
        # SummaryWriter accepts Path objects
        self.tensorboard_writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        
        # Clean way to define a file path inside curr_dir
        tensorboard_bat_path = self.curr_dir / "tensorboard_run.bat"
        
        text = f"""
        cd /d %~dp0
        call conda activate proj
        tensorboard --logdir=runs
        """
        tensorboard_bat_path.write_text(text)
    
    def load_model(self):
        model = self.model(self.device, only_G_A2B=True)
        model.model_init()
        
        # .exists() is much cleaner than os.path.exists()
        model_path = self.full_model_path if self.full_model_path.exists() else self.best_checkpoint_path
        status = model.load_model(model_path=str(model_path))
        model.eval()
        return model, status
    
    def set_experiment_name(self, experiment_name, model):
        self.experiment_name = experiment_name
        
        # Path chaining with / operator
        self.curr_dir = self.all_experiment_dir / experiment_name
        self.visualization_dir = self.curr_dir / self.visualization_subdir
        self.log_dir = self.curr_dir / self.log_subdir
        
        # File paths
        self.full_model_path = self.all_model_dir / f"{self.experiment_name}_model.pth"
        self.checkpoint_path = self.curr_dir / self.checkpoint_file
        self.best_checkpoint_path = self.curr_dir / self.best_checkpoint_file
        
        # Metrics paths
        self.quantitative_metrics_path = self.visualization_dir / self.quantitative_metrics_file
        self.agg_quantitative_metrics_path = self.visualization_dir / self.agg_quantitative_metrics_file
        self.qualitative_metrics_path = self.visualization_dir / self.qualitative_metrics_file
        self.log_history_path = self.visualization_dir / self.log_history_file
        
        self.model = model
        self.tensorboard_dir = self.curr_dir / "runs"
        
    def launch_tensorboard(self, port=6006):
        logdir = self.tensorboard_dir.resolve()
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
            
    def shutdown_pc(self, timeout = 300):
        def shutdown_windows():
            print("\n--- NO INPUT DETECTED ---")
            print("Shutting down the PC now...")
            # /s = shutdown, /t 0 = zero second delay
            os.system("shutdown /s /t 0")

            # Create the timer thread
        timer = threading.Timer(timeout, shutdown_windows)
        
        print("="*40)
        print("SYSTEM SHUTDOWN INITIATED")
        print(f"You have {timeout // 60} minutes to cancel this action.")
        print("To cancel: Type anything and press ENTER.")
        print("="*40)

        # Start the countdown in the background
        timer.start()

        # Wait for user input in the main terminal
        user_input = input("Action: ")

        # If the user types anything, cancel the timer
        if timer.is_alive():
            timer.cancel()
            print("\nShutdown cancelled. You may continue working.")
        else:
            # If the timer already finished, the PC is likely already shutting down
            print("Too late! Shutdown already triggered.")

# Usage remains similar
data_path = {
    "lol": Path("Our_CycleGAN/Dataset/LOL"),
    "lolv2": Path("Our_CycleGAN/Dataset/LOL-v2")
}

EXPERIMENT_MANAGER = ExperimentManager(data_path, tensorboard=True)