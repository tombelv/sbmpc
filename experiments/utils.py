import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

moving_path  = "experiments/results/trajectory_planning_moving/line graphs/"
stationary_path = "experiments/results/trajectory_planning_stationary/line graphs/"
stylised_path = "experiments/results/stylised_trajectory_planning/line graphs/"

output_path = stationary_path

def make_line_graph(path):
    data = np.array(pd.read_csv(path,engine='python'))
    x = data[:,0]
    y1 = data[:,1]
    y2 = data[:,2]
    y3 = data[:,3]
    
    plt.plot(x,y1, label='MPPI Sampler')
    plt.plot(x,y2, label='GP Sampler')
    plt.plot(x,y3, label='BNN Sampler')

    plt.xlabel("")
    plt.ylabel("")
    plt.title("")

    plt.legend()
    # plt.show()
    plt.savefig(output_path + "num_collisions.png")
    plt.close()


def make_line_graph_2(path):
    data = np.array(pd.read_csv(path,engine='python'))
    x = data[:,0]
    y1 = data[:,1]
    y2 = data[:,2]
    
    plt.plot(x,y1, label='MPPI Sampler')
    plt.plot(x,y2, label='GP Sampler')

    plt.xlabel("")
    plt.ylabel("")
    plt.title("")

    plt.legend()
    # plt.show()
    plt.savefig(output_path + "total_duration.png")
    plt.close()


def get_smoothness(traj):
    x_vel = np.diff(traj[:,0], axis=0) # estimate velocity
    y_vel = np.diff(traj[:,1], axis=0)
    z_vel = np.diff(traj[:,2], axis=0)

    x_acc = np.diff(x_vel, axis=0) # estimate acceleration
    y_acc = np.diff(y_vel, axis=0)
    z_acc = np.diff(z_vel, axis=0)

    return np.sum((x_acc**2 + y_acc**2 + z_acc**2)/3)


# make_line_graph("experiments/results/trajectory_planning_stationary/num collisions.csv")
# make_line_graph_2("experiments/results/stylised_trajectory_planning/total duration.csv")