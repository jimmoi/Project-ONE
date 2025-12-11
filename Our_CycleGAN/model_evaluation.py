import numpy as np
import torch
from CycleGAN_arch import CycleGAN
import matplotlib.pyplot as plt

def save_fig_png(fig, save_path):
    fig.savefig(save_path, dpi=300, bbox_inches='tight')

def evaluate_qualitative():
    pass

def evaluate_quantitative(compare_image):

    def tensor_to_rgb(image_tensor):
        """
        Denormalizes the tensor output (C, H, W) from [-1, 1] to [0, 1] and plots it.
        """
        image_tensor = image_tensor.squeeze(0)
        image_tensor = image_tensor * 0.5 + 0.5
        image_tensor = image_tensor.clamp(0.0, 1.0)
        return image_tensor
        
    def compare_plot(compare_image, titles="Compare Images (Low Light, Normal Light, Generated)"):
        # compare_image: (Set, Type, Channels, Height, Width)
        # compare_image: (n_Set, 3, 3, 286, 286)
        
        n_coloumn = compare_image.size(0)
        n_row = compare_image.size(1)
        
        for i in range(n_coloumn):
            for j in range(n_row):
                compare_image[i, j] = tensor_to_rgb(compare_image[i, j])
                
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
        # fig.tight_layout()
                
        return fig
    
    result_fig = compare_plot(compare_image)
    save_fig_png(result_fig, "Our_CycleGAN/compare.png")
