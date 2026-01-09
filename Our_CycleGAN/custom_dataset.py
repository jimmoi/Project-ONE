import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class CustomDataset(Dataset):
    def __init__(self, dataframe, is_test=False):
        self.dataframe = dataframe
        self.is_test = is_test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        A_path = self.dataframe.iloc[idx, 1]
        B_path = self.dataframe.iloc[idx, 3]

        A_image = Image.open(A_path).convert("RGB")
        B_image = Image.open(B_path).convert("RGB")

        if self.transform:
            A_image = self.transform(A_image)
            B_image = self.transform(B_image)

        return A_image, B_image
    
    @staticmethod
    def setup_dataset(transform):
        CustomDataset.transform = transform
        return CustomDataset
    
class CustomDataset_CBAM_GL_V2(CustomDataset):
    def __init__(self, dataframe, is_test=False):
        super().__init__(dataframe)
        self.patch_size = CustomDataset_CBAM_GL_V2.patch_size
        self.local_sample_n = CustomDataset_CBAM_GL_V2.local_sample_n
        self.is_test = is_test

    def __getitem__(self, idx):
        A_path = self.dataframe.iloc[idx, 1]
        B_path = self.dataframe.iloc[idx, 3]

        A_image = Image.open(A_path).convert("RGB")
        B_image = Image.open(B_path).convert("RGB")

        if self.transform:
            A_image = self.transform(A_image)
            A_patches = self.sample_local_patch(A_image)
            B_image = self.transform(B_image)
            B_patches = self.sample_local_patch(B_image)

        if self.is_test:
            return A_image, B_image
        else:
            return A_image, A_patches, B_image, B_patches
    
    def sample_local_patch(self, img, is_train=False):
        """
        Randomly crops a patch of size (patch_size, patch_size) from the input image.
        Assumes img is (B, C, H, W).
        Returns patch (B, C, patch_size, patch_size).
        """
        local_patches = []
        for _ in range(self.local_sample_n):
            if is_train:
                img = img.squeeze(0)
            h, w = img.shape[-2:]
            i = random.randint(0, h - self.patch_size)
            j = random.randint(0, w - self.patch_size)
            local_patches.append(img[..., i:i+self.patch_size, j:j+self.patch_size])
        local_patches = torch.stack(local_patches, dim=0)
        local_patches = local_patches.squeeze(0)
        return local_patches
    
    @staticmethod
    def setup_dataset(transform, patch_size=128, local_sample_n=4):
        CustomDataset_CBAM_GL_V2.transform = transform
        CustomDataset_CBAM_GL_V2.patch_size = patch_size
        CustomDataset_CBAM_GL_V2.local_sample_n = local_sample_n
        return CustomDataset_CBAM_GL_V2