from .D_net import *
from .G_net import *

class CycleGAN():
    def __init__(self, device):
        # model
        self.G_A2B = None
        self.G_B2A = None
        self.D_A = None
        self.D_B = None
        self.device = device
        self.model_init()
        
    def model_init(self, init_weights=True):
        # --- MODEL INITIALIZATION (User's original structure) ---
        # discriminator
        self.D_A = Discriminator(input_nc=3).to(self.device) # D_A is Discriminator for Domain A (Low Light)
        self.D_B = Discriminator(input_nc=3).to(self.device) # D_B is Discriminator for Domain B (Normal Light)

        # generator
        self.G_A2B = Generator(input_nc=3, output_nc=3).to(self.device) # G_B(A) -> B
        self.G_B2A = Generator(input_nc=3, output_nc=3).to(self.device)  # G_A(B) -> A
        
    def load_model(self, model_path):
        """Loads a pre-trained model from a checkpoint file."""
        if os.path.exists(model_path):
            try:
                print(f"Loading model from {model_path}...")
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Load states
                self.G_A2B.load_state_dict(checkpoint['G_A2B_state_dict'])
                self.G_B2A.load_state_dict(checkpoint['G_B2A_state_dict'])
                self.D_A.load_state_dict(checkpoint['D_A_state_dict'])
                self.D_B.load_state_dict(checkpoint['D_B_state_dict'])
                
                print(f"Model loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print(f"No model found at {model_path}")
            return False