import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

stationary_path = "/home/ubuntu/sbmpc/sbmpc/experiments/ecai/results/stationary/"
output_path = stationary_path + "plots/"

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

def plot_csv_file(csv_path, title, ylabel):
    data = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    
    # Assuming first column is x-axis (samples) and remaining are different methods
    x = data.iloc[:, 0]
    
    # Plot each method
    for i in range(1, len(data.columns)):
        plt.plot(x, data.iloc[:, i], marker='o', label=data.columns[i])
    
    plt.xlabel("Number of samples")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    filename = os.path.basename(csv_path).replace('.csv', '.png')
    plt.savefig(output_path + filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# Plot configurations for each CSV file
plot_configs = {
    "average obstacle dist.csv": ("Average Distance from Obstacles", "Distance (m)"),
    "average target dist.csv": ("Average Distance to Target", "Distance (m)"),
    "control frequency.csv": ("Control Frequency", "Frequency (Hz)"),
    "num collisions.csv": ("Number of Collisions", "Count"),
    "reached target.csv": ("Target Reached", "Success Rate"),
    "rejection rate.csv": ("Sample Rejection Rate", "Rejection Rate (%)"),
    "simulation runtime.csv": ("Simulation Runtime", "Time (s)")
}

# Generate plots for all CSV files
for filename, (title, ylabel) in plot_configs.items():
    csv_path = stationary_path + filename
    if os.path.exists(csv_path):
        plot_csv_file(csv_path, title, ylabel)
    else:
        print(f"File not found: {filename}")

print(f"All plots saved to: {output_path}")