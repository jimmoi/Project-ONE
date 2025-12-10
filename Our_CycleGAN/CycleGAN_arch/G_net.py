import torch.nn as nn
from .res_block import ResidualBlock


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, num_residual_blocks=9):
        super().__init__()

        # c7s1-64, d128, d256, R256×9, u128, u64, c7s1-3

        # Initial layers
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling layers (Encoder)
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks (Transformer)
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling layers (Decoder)
        out_features = in_features // 2
        for _ in range(2):
            model += [
                # Use ConvTranspose2d for upsampling
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7),
            nn.Tanh() # Tanh ensures output pixel values are between -1 and 1
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)