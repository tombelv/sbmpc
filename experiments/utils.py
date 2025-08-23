import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# moving_path  = "experiments/results/trajectory_planning_moving/line graphs/"
# stationary_path = "experiments/results/trajectory_planning_stationary/line graphs/"
# stylised_path = "experiments/results/stylised_trajectory_planning/line graphs/"

moving = "experiments/more results/moving/"
stationary = "experiments/more results/stationary/"
stylised = "experiments/more results/stylised/"

output_path = stylised

def make_line_graph(path):
    data = np.array(pd.read_csv(path,engine='python'))
    x = data[:,0]
    y1 = data[:,1]
    y2 = data[:,2]
    y3 = data[:,3]
    
    
    plt.plot(x,y1, label='MPPI Sampler')
    plt.plot(x,y2, label='GP Sampler')
    plt.plot(x,y3, label='BNN Sampler')
                
    # plt.ylim([0, 0.5])
    plt.xlabel("Number of samples")

    # plt.ylabel(r"Average acceleration $(m/s^2)$")
    plt.ylabel("Total simulation time (s)")
    plt.title("Total Simulation Time")

    plt.legend()
    # plt.show()
    plt.savefig(output_path + "simiultation_runtime.png")
    plt.close()


def make_line_graph_2(path):
    data = np.array(pd.read_csv(path,engine='python'))
    x = data[:,0]
    y1 = data[:,1]
    y2 = data[:,2]
    
    plt.plot(x,y1, label='GP Sampler')
    plt.plot(x,y2, label='BNN Sampler')

    plt.xlabel("Delta (constraint violation threshold)")

    plt.ylabel(r"Average acceleration $(m/s^2)$")
    plt.title("Average Acceleration")

    plt.legend()
    # plt.show()
    plt.savefig(output_path + "average_acceleration.png")
    plt.close()


def get_smoothness(traj):
    dt = 0.02

    x_vel = np.diff(traj[:,0], axis=0)/dt # estimate velocity
    y_vel = np.diff(traj[:,1], axis=0)/dt
    z_vel = np.diff(traj[:,2], axis=0)/dt
   
    x_acc = np.diff(x_vel, axis=0)/dt # estimate acceleration
    y_acc = np.diff(y_vel, axis=0)/dt
    z_acc = np.diff(z_vel, axis=0)/dt

    av_x_acc = np.mean(x_acc**2) # get average acceleration
    av_y_acc = np.mean(y_acc**2)
    av_z_acc = np.mean(z_acc**2)

    return np.mean([av_x_acc,av_y_acc,av_z_acc]) # get overall average acceleration
