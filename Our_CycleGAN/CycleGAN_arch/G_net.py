import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Pad, Conv, Norm, ReLU, Pad, Conv, Norm
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        # x + conv_block(x) is the residual connection
        return x + self.conv_block(x)

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
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP ใช้สำหรับหาความสัมพันธ์ระหว่าง Channel
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        # Conv layer เพื่อหาความสัมพันธ์เชิงพื้นที่
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # หาค่าเฉลี่ยและค่าสูงสุดในแกน Channel (dim=1)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# ==========================================
# ส่วนที่ 2: Residual Block (รวม CBAM)
# ==========================================

class ResidualBlock_CBAM(nn.Module):
    def __init__(self, dim, use_cbam=True):
        super().__init__()
        self.use_cbam = use_cbam
        
        # Pad, Conv, Norm, ReLU, Pad, Conv, Norm
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim)
        )

        if self.use_cbam:
            self.ca = ChannelAttention(dim)
            self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv_block(x)
        
        if self.use_cbam:
            # Apply Channel Attention
            out = self.ca(out) * out
            # Apply Spatial Attention
            out = self.sa(out) * out
            
        # x + conv_block(x) [processed by CBAM] is the residual connection
        return x + out

# ==========================================
# ส่วนที่ 3: Generator (Main Architecture)
# ==========================================

class Generator_CBAM(nn.Module):
    def __init__(self, input_nc, output_nc, num_residual_blocks=9):
        super().__init__()

        # Structure: c7s1-64, d128, d256, R256×9, u128, u64, c7s1-3

        # 1. Initial layers (c7s1-64)
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # 2. Downsampling layers (Encoder) -> d128, d256
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

        # 3. Residual blocks with CBAM (Transformer) -> R256x9
        # in_features ตอนนี้คือ 256
        for _ in range(num_residual_blocks):
            model += [ResidualBlock_CBAM(in_features, use_cbam=True)]

        # 4. Upsampling layers (Decoder) -> u128, u64
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # 5. Output layer (c7s1-3)
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7),
            nn.Tanh() # Output -1 to 1
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)