import os
import json
from experiment_manager import EXPERIMENT_MANAGER
import matplotlib.pyplot as plt
import numpy as np

def plot_metric_history(df, n_x_ticks=50):

    loss_columns = ['G_Total', 'G_A2B', 'G_B2A', 'Cycle', 'Id', 'D_A', 'D_B']
    summary_df = df.groupby('epoch')[loss_columns].mean().reset_index()
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.flatten()
    
    x_data = summary_df['epoch']
    min_epoch = x_data.min()
    max_epoch = x_data.max()
    
    # Generate tick positions every 50 epochs
    epoch_ticks = np.arange(min_epoch, max_epoch + 1, n_x_ticks)
    
    for ax, col in zip(axes, loss_columns):
        ax.plot(
            x_data,
            summary_df[col],
            label=f'Average {col}',
            marker='o',         
            linestyle='-',      
            linewidth=2,
            markersize=4        # Smaller marker size for clarity
        )
        
        ax.set_title(f'Average Loss: {col} per Epoch', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(f'Average {col} Loss', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        ax.set_xticks(epoch_ticks)
        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout(pad=3.0) 
    plt.suptitle('CycleGAN Average Loss Metrics per Epoch', y=1.02, fontsize=16, fontweight='bold')
    plt.savefig(EXPERIMENT_MANAGER.log_history_path)