import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np

metrices_data = {
    "cyclegan_200": {"PSNR": 23.56176164605599, "SSIM": 0.7775808572769165, "LPIPS": 0.1877647888660431, "NIQE": 5.09457269117504},
    "cyclegan_cbam_200": {"PSNR": 25.444143474908028, "SSIM": 0.8170828819274902, "LPIPS": 0.14077717449516058, "NIQE": 5.186436060372383},
    "cyclegan_cbam_gl_200": {"PSNR": 21.3376044730869, "SSIM": 0.6692161560058594, "LPIPS": 0.28203416846692564, "NIQE": 5.393114638328606},
    "cyclegan_CBAM_GL_V2_200_patch64_local16": {"PSNR": 23.670312002786062, "SSIM": 0.7772073149681091, "LPIPS": 0.1829250954836607, "NIQE": 4.248169059996667},
    "cyclegan_CBAM_GL_V2_200_patch128_local4": {"PSNR": 19.10156551579325, "SSIM": 0.6833591461181641, "LPIPS": 0.3364622661471367, "NIQE": 4.9862198617708655},
    "cyclegan_CBAM_GL_V3_200_patch128_local5": {"PSNR": 28.2633035289971, "SSIM": 0.8428758978843689, "LPIPS": 0.11941311068832874, "NIQE": 4.942464001586857}
}

all_experiment_dir = Path(r"Our_CycleGAN\experiments")
metrices_over_exp_file = all_experiment_dir / "metrices_over_exp.png"
loss_over_exp_file = all_experiment_dir / "loss_over_exp.png"
loss_each_exp_file = "loss_each_exp.png"

def get_data_each_experiment(experiment_dir):
        data = []
        if experiment_dir.is_dir():
            experiment_name = experiment_dir.name
            log_dir = experiment_dir / "log"
            for log_file in log_dir.iterdir():
                if log_file.is_file() and log_file.name.startswith("Training_log_history") and log_file.suffix == ".json":
                    with open(log_file,"r") as f:
                        metric_hist = json.load(f)
                        for train_iteration in metric_hist:
                            row_temp = train_iteration["metrics"].copy()
                            row_temp["epoch"] = train_iteration["epoch"]
                            row_temp["it"] = train_iteration["iteration"]
                            data.append(row_temp)
                        
        if not data:
            raise ValueError("No metric history found.")
        
        df = pd.DataFrame(data)
        return df

def plot_metrices_over_exp(data):
    df = pd.DataFrame(data).T.reset_index().rename(columns={'index': 'Model'})
    
    name_mapper = {
        "cyclegan_200":"CycleGAN Only",
        "cyclegan_cbam_200":"+ CBAM",
        "cyclegan_cbam_gl_200":"CBAM + GL(Lizuka)",
        "cyclegan_CBAM_GL_V2_200_patch64_local16":"CBAM + GL(Tiger (64, 16))",
        "cyclegan_CBAM_GL_V2_200_patch128_local4":"CBAM + GL(Tiger (128, 4))",
        "cyclegan_CBAM_GL_V3_200_patch128_local5":"CBAM + GL(Zhao)",
    }
    
    df['Model'] = df['Model'].map(name_mapper)

    # Set consistent style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")

    # Assign colors to models for consistency across charts
    model_order = df.sort_values('PSNR', ascending=False)['Model'].tolist()
    colors = sns.color_palette("viridis", len(model_order))
    color_map = dict(zip(model_order, colors))

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    metrics = [
        ('PSNR', 'Higher is Better (↑)', True),
        ('SSIM', 'Higher is Better (↑)', True),
        ('LPIPS', 'Lower is Better (↓)', False),
        ('NIQE', 'Lower is Better (↓)', False)
    ]

    for i, (metric, sub_label, higher_is_better) in enumerate(metrics):
        # Sort data for the specific metric to show ranking
        df_sorted = df.sort_values(by=metric, ascending=higher_is_better)
        
        # Horizontal bar plot
        bars = axes[i].barh(df_sorted['Model'], df_sorted[metric], 
                        color=[color_map[m] for m in df_sorted['Model']],
                        edgecolor='white', linewidth=1)
    
        # Title and Subtitle
        axes[i].set_title(f'{metric}', fontsize=18, fontweight='bold', loc='left', pad=20)
        axes[i].text(0, 1.02, sub_label, transform=axes[i].transAxes, fontsize=12, color='gray', style='italic')
        
        # Add data labels
        for bar in bars:
            width = bar.get_width()
            axes[i].text(width + (width * 0.01), bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', va='center', fontsize=12, fontweight='bold')

        # Cleaning up aesthetics
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
        axes[i].grid(axis='x', linestyle='--', alpha=0.4)
        axes[i].xaxis.set_tick_params(labelsize=10)
        axes[i].yaxis.set_tick_params(labelsize=12)

    plt.tight_layout(pad=4.0)
    plt.savefig(metrices_over_exp_file, dpi=300, bbox_inches='tight')
    
def plot_loss_over_experiment():
    def plot_gan_loss_total(data_dict):
        df_combine = pd.concat([df for _, df in data_dict.items()])
        df_combine = df_combine[["G_Total"]]
        y_max = (df_combine.quantile(0.98) * 1.2).values[0]
        y_min = (df_combine.min() * 0.9).values[0]

        plt.figure(figsize=(20, 10))
        
        name_mapper = {
            "cyclegan_200":"CycleGAN Only",
            "cyclegan_cbam_200":"+ CBAM",
            "cyclegan_cbam_gl_200":"CBAM + GL(Lizuka)",
            "cyclegan_CBAM_GL_V2_200_patch64_local16":"CBAM + GL(Tiger (64, 16))",
            "cyclegan_CBAM_GL_V2_200_patch128_local4":"CBAM + GL(Tiger (128, 4))",
            "cyclegan_CBAM_GL_V3_200_patch128_local5":"CBAM + GL(Zhao)",
        }
        
        for experiment, df in data_dict.items():
            x_data = df["epoch"]
            y_data = df["G_Total"]
            
            
            plt.plot(
                x_data, 
                y_data, 
                label=name_mapper[experiment],
                linewidth=1.5
            )
        plt.legend(loc="upper right")
        plt.ylim(y_min, y_max)
        plt.xlim(0)
        plt.title("Compare GAN Loss Total")
        plt.xlabel("Epoch")
        plt.ylabel("GAN Loss Total")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(loss_over_exp_file, dpi=300, bbox_inches='tight')
    
    global all_experiment_dir
    
    data_dict = {}
    for experiment_dir in all_experiment_dir.iterdir():
        if experiment_dir.is_dir():
            df = get_data_each_experiment(experiment_dir)
            df = df[df["epoch"]<=200]
            df = df.groupby('epoch').mean().reset_index()
            df = df[["epoch","G_Total"]]
            data_dict[experiment_dir.name] = df
    
    plot_gan_loss_total(data_dict)

def plot_loss_each_experiment():
    def plot_metric_history(df, experiment_dir, n_x_ticks=50):
        df = df[df["epoch"]<=200]
        summary_df = df.groupby('epoch').mean().reset_index()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        axes = axes.flatten()
        
        x_data = summary_df['epoch']
        min_epoch = x_data.min()
        max_epoch = x_data.max()
        
        # Generate tick positions every 50 epochs
        epoch_ticks = np.arange(min_epoch, max_epoch + 1, n_x_ticks)
        
        
        plots_title = {
            "Total Generator Loss" : ["G_Total"], 
            "Identity & Cycle Consistency Loss" : ["Id", "Cycle"],
            "Generator B2A & Discriminator A Loss" : ["G_B2A", "D_A"],
            "Generator A2B & Discriminator B Loss" : ["G_A2B", "D_B"]
        }
        
        label_mapping = {
            "G_Total": "Total Generator Loss",
            "G_A2B": "Generator A2B",
            "G_B2A": "Generator B2A",
            "D_A": "Discriminator A",
            "D_B": "Discriminator B",
            "Id": "Identity Loss",
            "Cycle": "Cycle Consistency Loss"
        }
        
        for ax, (title, columns) in zip(axes, plots_title.items()):
            df_combine = pd.concat([summary_df[col] for col in columns])
            y_max = df_combine.quantile(0.98) * 1.2 
            y_min = df_combine.min() * 0.9
            
            for col in columns:
                ax.plot(
                    x_data,
                    summary_df[col],
                    label=f'Avg. {label_mapping[col]}',        
                    linewidth=2,
                )
            
            ax.set_title(f'Avg. {title} per Epoch', fontsize=14)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(f'Avg. Loss', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_ylim(y_min, y_max)
            
            ax.set_xticks(epoch_ticks)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout(pad=1.0) 
        plt.suptitle('CycleGAN Average Loss Metrics per Epoch', y=1.02, fontsize=16, fontweight='bold')
        plt.savefig(experiment_dir / "visualization/loss_each_exp_file.png")
    
    global all_experiment_dir
    
    for experiment_dir in all_experiment_dir.iterdir():
        if experiment_dir.is_dir():
            df = get_data_each_experiment(experiment_dir)
            plot_metric_history(df, experiment_dir)
        
        
        

if __name__ == "__main__":
    plot_metrices_over_exp(metrices_data)
    plot_loss_over_experiment()
    plot_loss_each_experiment()
    