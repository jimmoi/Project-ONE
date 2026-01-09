import torch
import torch.nn as nn

# PatchGAN
class Discriminator(nn.Module): 
    def __init__(self, input_nc):
        super().__init__()
        
        # C64 - C128 - C256 - C512
        model = [
            # C64 (No normalization for the first layer)
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # C128
        model += [
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # C256
        model += [
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # C512
        model += [
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1), # Stride 1 in the penultimate layer
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Output layer (1-channel output: the 'patch' score)
        model += [nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # We don't apply Sigmoid here; it's handled in the loss function (MSELoss in this case)
        return self.model(x)
    
class Discriminator_GL(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        
        # Global Discriminator
        # Input: 256 x 256
        self.global_conv = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.global_fc = nn.Linear(512 * 4 * 4, 1024)
        
        # Local Discriminator
        # Input: 128 x 128
        self.local_conv = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.local_fc = nn.Linear(512 * 4 * 4, 1024)
        
        # Concatenation layer
        self.feat_fc = nn.Linear(2048, 1)

    def forward(self, x_global, x_local):
        # Global pathway
        x_g = self.global_conv(x_global)
        x_g = x_g.view(x_g.size(0), -1)
        x_g = self.global_fc(x_g)
        
        # Local pathway
        x_l = self.local_conv(x_local)
        x_l = x_l.view(x_l.size(0), -1)
        x_l = self.local_fc(x_l)
        
        # Combine
        out = torch.cat((x_g, x_l), 1)
        out = self.feat_fc(out)
        
        return out
    
class Discriminator_GL_V2(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        
        # Global Discriminator
        # Input: 256 x 256
        self.global_conv = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.global_fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.global_fc2 = nn.Linear(1024, 1)
        
        # Local Discriminator
        # Input: 128 x 128
        self.local_conv = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.local_fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.local_fc2 = nn.Linear(1024, 1)
        

    def forward(self, x_global, x_locals):
        # Global pathway
        x_g = self.global_conv(x_global)
        x_g = x_g.view(x_g.size(0), -1)
        x_g = self.global_fc1(x_g)
        x_g = self.global_fc2(x_g) # size: (batch_size, 1)
        
        # Local pathway
        x_l = self.local_conv(x_locals)
        x_l = x_l.view(x_l.size(0), -1)
        x_l = self.local_fc1(x_l)
        x_l = self.local_fc2(x_l) # size: (batch_size, 1)
        
        #concat
        out = torch.cat((x_g, x_l))
        
        return out