import numpy as np
import torch
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr_ski
from skimage.metrics import structural_similarity as ssim_ski
import pyiqa

import sys
import os
from contextlib import contextmanager


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_model = pyiqa.create_metric('lpips', device=device)
niqe_model = pyiqa.create_metric('niqe', device=device)


def save_fig_png(fig, save_path):
    fig.savefig(save_path, dpi=300, bbox_inches='tight')


def evaluate_quantitative(generated_image_tensor, real_image_tensor):
    """
    Assumes tensors are in range [0, 1] or [-1, 1].
    Shape: (1, C, H, W)
    """
    
    # 1. Ensure tensors are on the correct device and detached
    gen_t = generated_image_tensor.to(device).detach()
    real_t = real_image_tensor.to(device).detach()

    # 2. Handle [-1, 1] to [0, 1] conversion if necessary
    # If your model outputs Tanh (-1 to 1), uncomment the next two lines:
    # gen_t = (gen_t + 1) / 2
    # real_t = (real_t + 1) / 2

    # 3. Prepare Numpy versions for Skimage (H, W, C)
    gen_np = gen_t.squeeze().permute(1, 2, 0).cpu().numpy()
    real_np = real_t.squeeze().permute(1, 2, 0).cpu().numpy()

    # PSNR & SSIM (Standardized to 1.0 range)
    psnr_score = psnr_ski(real_np, gen_np, data_range=1.0)
    ssim_score = ssim_ski(real_np, gen_np, data_range=1.0, channel_axis=-1)

    # LPIPS & NIQE (Using pre-loaded models)
    with torch.no_grad():
        lpips_score = lpips_model(gen_t, real_t).item()
        niqe_score = niqe_model(gen_t).item()

    return {
        'PSNR': psnr_score, 
        'SSIM': ssim_score, 
        'LPIPS': lpips_score, 
        'NIQE': niqe_score
    }

def evaluate_qualitative(compare_image, save_path):
        
    def compare_plot(compare_image, titles="Compare Images (Low Light, Normal Light, Generated)"):
        """
        Plot the compare image (Low Light, Normal Light, Generated).
        compare_image: (Set, Type, Height, Width, Channels)
        compare_image: (n_Set, 3, 286, 286, 3)
        """
        
        n_coloumn = compare_image.size(0)
        n_row = compare_image.size(1)
        
        compare_image = compare_image.squeeze()
        compare_image = compare_image.permute(0, 1, 3, 4, 2) # (Set, Type, Height, Width, Channels)
                
        title_types = ["Low Light", "Normal Light", "Generated"]
        title_sets = list(range(n_coloumn))

        fig, axes = plt.subplots(n_row, n_coloumn, figsize=(n_coloumn*5, n_row*5), sharex=True, sharey=True)
        for i in range(n_coloumn):
            for j in range(n_row):
                if i == 0:
                    axes[j, i].set_ylabel(title_types[j], fontsize=16)
                if j == n_row-1:
                    axes[j, i].set_xlabel(title_sets[i], fontsize=16)
                    
                axes[j, i].imshow(compare_image[i, j])
        fig.suptitle(titles, fontsize=16)
                
        return fig
    
    result_fig = compare_plot(compare_image)
    save_fig_png(result_fig, save_path)
