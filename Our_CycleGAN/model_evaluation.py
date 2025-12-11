import numpy as np
import torch
from CycleGAN_arch import CycleGAN
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr_ski
from skimage.metrics import structural_similarity as ssim_ski
import pyiqa

import sys
import os
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    """Context manager to temporarily redirect stdout to /dev/null."""
    with open(os.devnull, 'w') as null_file:
        # Save the original stdout
        original_stdout = sys.stdout
        # Redirect stdout to the null file
        sys.stdout = null_file
        try:
            # Yield control back to the 'with' block
            yield
        finally:
            # Restore the original stdout
            sys.stdout = original_stdout


def save_fig_png(fig, save_path):
    fig.savefig(save_path, dpi=300, bbox_inches='tight')


def evaluate_quantitative(generated_image_tensor, real_image_tensor, device):
    # Convert to numpy (H, W, C)
    generated_image_numpy = generated_image_tensor.squeeze().permute(1, 2, 0).numpy()
    real_image_numpy = real_image_tensor.squeeze().permute(1, 2, 0).numpy()
    
    generated_image_tensor = generated_image_tensor.to(device)
    real_image_tensor = real_image_tensor.to(device)    
    
    def calculate_psnr(generated_image_numpy, real_image_numpy):
        """
        Calculate the PSNR, SSIM, and LPIPS metrics.
        """

        psnr_score = psnr_ski(real_image_numpy, generated_image_numpy, data_range=1.0)
    
        return psnr_score
    
    def calculate_ssim(generated_image_numpy, real_image_numpy):
        """
        Calculate the SSIM metric.
        """
        ssim_score = ssim_ski(real_image_numpy, generated_image_numpy, data_range=1.0, multichannel=True, channel_axis=-1)
        
        return ssim_score
    
    def calculate_lpips(generated_image_tensor, real_image_tensor):
        """
        Calculate the LPIPS metric.
        """
        lpips_score = None
        with suppress_stdout():
            lpips_metric = pyiqa.create_metric('lpips', device=device)
            
            with torch.no_grad():
                lpips_score = lpips_metric(generated_image_tensor, real_image_tensor).item() # Note the order: (dist, ref)
            
        return lpips_score
    
    def calculate_niqe(generated_image_tensor):
        """
        Calculate the NIQE metric.
        """
        niqe_metric = pyiqa.create_metric('niqe', device=device)
        with torch.no_grad():
            niqe_score = niqe_metric(generated_image_tensor).item()

        return niqe_score
    
    psnr_score = calculate_psnr(generated_image_numpy, real_image_numpy)
    ssim_score = calculate_ssim(generated_image_numpy, real_image_numpy)
    lpips_score = calculate_lpips(generated_image_tensor, real_image_tensor)
    niqe_score = calculate_niqe(generated_image_tensor)
    
    metrics = {'PSNR': psnr_score, 'SSIM': ssim_score, 'LPIPS': lpips_score, 'NIQE': niqe_score}
    
    return metrics

def evaluate_qualitative(compare_image):
        
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
                    axes[j, i].set_ylabel(title_types[i], fontsize=16)
                if j == n_row-1:
                    axes[j, i].set_xlabel(title_sets[i], fontsize=16)
                    
                axes[j, i].imshow(compare_image[i, j])
        fig.suptitle(titles, fontsize=16)
                
        return fig
    
    result_fig = compare_plot(compare_image)
    save_fig_png(result_fig, "Our_CycleGAN/compare.png")
