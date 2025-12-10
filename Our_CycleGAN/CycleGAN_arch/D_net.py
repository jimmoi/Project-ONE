import torch.nn as nn

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