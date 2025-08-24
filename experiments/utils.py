import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

moving = "/home/ubuntu/sbmpc/sbmpc/experiments/ecai/results/moving/"
stationary = "/home/ubuntu/sbmpc/sbmpc/experiments/ecai/results/stationary/"
complex = "/home/ubuntu/sbmpc/sbmpc/experiments/ecai/results/complex/"

output_path = stationary

def make_line_graph(path, plot, unit, method):
    data = np.array(pd.read_csv(path,engine='python'))
    x = data[:,0] # n_obs
    y1 = data[:,1] # traj
    y2 = data[:,2]  # mppi rej
    y3 = data[:,3] # mppi penalty
    y4 = data[:,4] # bc mppi
     
    if (method == "penalty"):
        plt.plot(x,y3, label='MPPI Penalty') # ie.e. baseline
        plt.plot(x,y4, label='BC-MPPI')
    else:
        plt.plot(x,y2, label='MPPI Rejection')
        plt.plot(x,y4, label='BC-MPPI')

    plt.xlabel("Number of obstacles")

    plt.ylabel(plot + f" ({unit})")
    plt.title(plot)

    plt.legend()
    plt.savefig(output_path + f"{plot} ({method}).png")
    plt.close()


make_line_graph(f"{output_path}simulation runtime.csv", "Simulation Runtime", "%", "penalty")