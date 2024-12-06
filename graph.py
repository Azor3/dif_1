import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def load_weights(filename):
    try:
        return pd.read_csv(filename, header=None).values
    except:
        print(f"Couldn't load {filename}")
        return None

def load_results(filename):
    try:
        return pd.read_csv(filename, header=None, names=['Algorithm', 'Epoch', 'Loss', 'Time'])
    except:
        print(f"Couldn't load {filename}")
        return None

def plot_tsne_by_algorithm(all_weights, algorithm, start_idx):
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("husl", n_colors=5)
    markers = ['o', 's', '^', 'D', 'v']
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30.0)
    algorithm_weights = all_weights[start_idx:start_idx+250]  # 5 runs * 50 epochs
    weights_2d = tsne.fit_transform(algorithm_weights)
    
    for run in range(5):
        trajectory = weights_2d[run*50:(run+1)*50]
        plt.plot(trajectory[:, 0], trajectory[:, 1],
                color=colors[run],
                label=f'Run {run}',
                linewidth=2,
                marker=markers[run],
                markersize=4,
                markevery=[0, -1],
                alpha=0.8)
        
        # Start point
        plt.plot(trajectory[0, 0], trajectory[0, 1],
                'k*', markersize=15, alpha=0.7)
        # End point
        plt.plot(trajectory[-1, 0], trajectory[-1, 1],
                'ko', markersize=10, alpha=0.7)
    
    plt.title(f'{algorithm} Optimization Trajectories (T-SNE)', fontsize=14)
    plt.xlabel('T-SNE Dimension 1')
    plt.ylabel('T-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{algorithm.lower()}_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_loss_curves_by_algorithm(results_data, algorithm):
    # Time vs Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for run in range(5):
        run_data = results_data[(results_data['Algorithm'] == algorithm) & 
                              (results_data['Run'] == run)]
        plt.plot(run_data['Time'], run_data['Loss'], 
                label=f'Run {run} (w={run_data["Init_Weight"].iloc[0]})')
    
    plt.title(f'{algorithm} Loss vs Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Epoch vs Loss
    plt.subplot(1, 2, 2)
    for run in range(5):
        run_data = results_data[(results_data['Algorithm'] == algorithm) & 
                              (results_data['Run'] == run)]
        plt.plot(run_data['Epoch'], run_data['Loss'],
                label=f'Run {run} (w={run_data["Init_Weight"].iloc[0]})')
    
    plt.title(f'{algorithm} Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{algorithm.lower()}_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def process_results():
    init_weights = [0.0, 0.1, 0.01, 0.001, 0.0001]
    algorithms = ['GD', 'SGD', 'Adam']
    results_data = []
    
    for alg in algorithms:
        for run in range(5):
            filename = f'{alg.lower()}_results_{run}.csv'
            df = pd.read_csv(filename, header=None, 
                           names=['Algorithm', 'Epoch', 'Loss', 'Time'])
            df['Run'] = run
            df['Init_Weight'] = init_weights[run]
            results_data.append(df)
    
    return pd.concat(results_data, ignore_index=True)

def main():
    # Load weights
    all_weights = []
    algorithms = ['GD', 'SGD', 'Adam']
    
    for alg in algorithms:
        for run in range(5):
            weights = load_weights(f'weights_{alg.lower()}_{run}.csv')
            if weights is not None:
                all_weights.append(weights)
    
    all_weights = np.vstack(all_weights)
    
    # Plot individual T-SNE visualizations
    for idx, alg in enumerate(algorithms):
        plot_tsne_by_algorithm(all_weights, alg, idx * 250)
    
    # Process and plot loss curves
    results_data = process_results()
    
    # Plot individual loss curves for each algorithm
    for alg in algorithms:
        plot_loss_curves_by_algorithm(results_data, alg)

if __name__ == "__main__":
    main()