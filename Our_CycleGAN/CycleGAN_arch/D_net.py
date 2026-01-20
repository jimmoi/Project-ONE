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
    """
    Global Local Discriminator with only one patch 128 x 128.
    concat global and local features.
    output is 1 channel.
    """
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
    """
    Global Local Discriminator with n patch.
    output is n+1 channel.
    *** 4 patch = 128 x 128
    *** 16 patch = 64 x 64
    """
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
            
            ## for local patch size = 128
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
        # x_locals size: (n_patches, 3, 128, 128)
        x_l = self.local_conv(x_locals) # size: (n_patches, 512, 4, 4)
        x_l = x_l.view(x_l.size(0), -1)
        x_l = self.local_fc1(x_l) # size: (n_patches, 1024)
        x_l = self.local_fc2(x_l) # size: (n_patches, 1)
        
        #concat
        out = torch.cat((x_g, x_l)) # size: (batch_size, n_patches+1)
        
        return out
  
class Discriminator_GL_V3(nn.Module):
    class Discriminator(nn.Module):
        def __init__(self, in_channels=3, num_downsamples=3):
            super().__init__()
            layers = []
            
            # Initial extraction (Stride 2)
            layers.append(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Dynamic number of downsampling layers to match sizes
            curr_channels = 64
            for i in range(num_downsamples - 1):
                out_channels = curr_channels * 2
                layers.append(nn.Conv2d(curr_channels, out_channels, kernel_size=4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                curr_channels = out_channels
                
            # Final Integration (Stride 1) - Results in 31x31 feature map
            layers.append(nn.Conv2d(curr_channels, 512, kernel_size=4, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(512))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    def __init__(self, in_channels=3):
        super().__init__()
        self.global_disc = self.Discriminator(in_channels=in_channels, num_downsamples=3)
        self.local_disc = self.Discriminator(in_channels=in_channels, num_downsamples=2)
        
        # Final classifier: 1x1 convolution to generate N x N Patch score matrix 
        self.classifier = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x_global, x_local):
        """
        global_img: [B, 3, 256, 256]
        local_patches: [B*5, 3, 128, 128] (5 patches per image in the batch) 
        """
        # Process global image
        global_features = self.global_disc(x_global)
        
        # Process local patches and reshape to combine their features
        # Assuming we average or sum the 5 patches per image to match batch size
        local_features = self.local_disc(x_local)
        
        # Expand global features to match the 5 local patches
        global_features_expanded = global_features.expand(local_features.size(0), -1, -1, -1) # Shape: [5, 512, 31, 31]
        
        # Resize/process local features to match global spatial dimensions if necessary
        # and perform element-wise sum [cite: 148]
        # (The paper aligns these via the specific strides to ensure same-size feature maps)
        fused_features = global_features + local_features 
        
        out = self.classifier(fused_features)
        
        return out