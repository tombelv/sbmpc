import sys

sys.path.extend(["/home/student/mppi_ws/sbmpc"])

import numpy as np
import os
import pinocchio as pin
from sbmpc.geometry import quat2rotm

import matplotlib.pyplot as plt

def get_traj_stats(trajectory, reference, time_arr, constraints_arr, settle_band_meters = 0.3, settle_band_heading = np.pi/12, settle_index_increment: int = 200):
    # calculate settling time
    # we define this as the time when the robot crosses into a certain region bounded around the goal and stays there
    # we will define bounds on this in x, y, bearing
    # calculate steady state error
    # calculate overshoot

    base_xyz, orientation_quat_arr = trajectory[:, 0:3], trajectory[:, 3:7]
    
    change_arrs = []
    diff_from_ref_arr = []
    for i in range(3):
        change = np.abs(base_xyz[1:, i] - base_xyz[0:-1, i])
        change_arrs.append(change)
        diff_ref = base_xyz[:, i] - reference[i]
        diff_from_ref_arr.append(diff_ref)


    orientation_angles = []
    for i in range(orientation_quat_arr.shape[0]):
        quat_ori = orientation_quat_arr[i, 0:4]
        rot = quat2rotm(quat_ori)
        # euler = pin.rpy.matrixToRpy(np.matrix(rot))
        # orientation_angles.append(euler)
    # orientation_angles = np.array(orientation_angles)

    pos_within_band = []
    for i in range(3):
        within_band = np.abs(diff_from_ref_arr[i]) <= settle_band_meters
        pos_within_band.append(within_band)

    # bearing_within_band = np.abs(base_heading) <= settle_band_heading

    settle_indices = [-1] * 3

    for i in range(1, len(pos_within_band[0])):
        for j in range(3):
            within_band = pos_within_band[j]
            if not within_band[i] and settle_indices[j] != -1:
                settle_indices[j] = -1
            elif within_band[i] and settle_indices[j] == -1:
                settle_indices[j] = i

    # we want targets for xyz to be the reference if it got to within settle bands and stayed there
    # or the constraints if it did not.

    settle_index = -1
    if settle_indices[0] == -1 or settle_indices[1] == -1 or settle_indices[2] == -1:
        pass
    else:
        settle_index = max(settle_indices[0], max(settle_indices[1], settle_indices[2]))

    time_to_settle = -1
    if settle_index != -1:
        time_to_settle = time_arr[settle_index]

    settle_index_incremented = settle_index + settle_index_increment if settle_index != -1 else -1
    settle_index_incremented = len(base_xyz) - 1 if settle_index_incremented >= len(base_xyz) else settle_index_incremented

    settle_errors = [None] * 3
    if settle_index_incremented != -1:
        for i in range(3):
            err = np.max(np.abs(base_xyz[settle_index_incremented:, i] - reference[i]))
            settle_errors[i] = err

    # overshoot
    # for this we will define overshoot as oscillation over the target position
    # the first time we cross over, till we go back over the target, we take the max difference for overshoot

    overshoot_indices = [-1] * 3
    pos_to_neg_flags = [None] * 3
    first_overshoots = [-1] * 3
    for i in range(3):
        starting_val = base_xyz[0, i]
        pos_to_neg = (starting_val - reference[i]) > 0
        pos_to_neg_flags[i] = pos_to_neg
        for j in range(len(base_xyz)):
            if pos_to_neg and base_xyz[j, i] < reference[i] and overshoot_indices[i] == -1:
                overshoot_indices[i] = j
                first_overshoots[i] = time_arr[j]
            elif not pos_to_neg and base_xyz[j, i] > reference[i] and overshoot_indices[i] == -1:
                overshoot_indices[i] = j
                first_overshoots[i] = time_arr[j]
    
    overshoot_vals = [None] * 3

    for i in range(3):
        overshoot_idx = overshoot_indices[i]
        if overshoot_idx != -1:
            overshoot_val = np.max(np.abs(base_xyz[overshoot_idx:, i] - reference[i]))
            overshoot_vals[i] = overshoot_val

    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    labels = ["x", "y", "z"]
    legend_labels = [None, None, None, None]
    for i in range(3):
        these_labels = [val for val in legend_labels]
        if i == 0:
            these_labels[0] = "reference"
            these_labels[1] = "settle_bounds"
            these_labels[2] = "settle_point"
            these_labels[3] = "first_overshoot_point"


        axes[i].plot(time_arr, base_xyz[:, i])
        axes[i].set_ylabel(labels[i])
        axes[i].axhline(reference[i], color="r", label=these_labels[0])
        # axes[i].axhline(reference[i] + settle_band_meters, color="r", linestyle="--")
        # axes[i].axhline(reference[i] - settle_band_meters, color="r", linestyle="--", label=these_labels[1])
        axes[i].axvline(time_arr[settle_indices[i]], color="g", label=these_labels[2])
        axes[i].axvline(first_overshoots[i], color="b", label=these_labels[3])
    axes[2].set_xlabel("time")
    fig.legend()
    fig.suptitle("Position vs Reference, with Steady State Error Marked")
    fig.savefig("/home/developer/MPPI_WS/fig.png")
    plt.show()

    for i in range(3):

        print("hi")


    print("hi")


    metrics = {"time_to_settle": time_to_settle,
               "settle_index": settle_index, # if -1, the system didn't settle
               "settle_indices_pos": settle_indices,
               "settle_index_incremented": settle_index_incremented,
               "settle_errors_pos": settle_errors, # if this is 0, the system didn't settle
               "overshoot_indices": overshoot_indices,
               "first_overshoots": first_overshoots,
               "overshoot_vals": overshoot_vals}
    return metrics

def main():

    file_names = ["trajectory.npy", "ctrl_input.npy", "ref.npy", "time_arr.npy", 'constraints_arr.npy']

    in_dir = "/home/developer/MPPI_WS/20hz"

    files = [os.path.join(in_dir, file) for file in file_names]
    arrays = [np.load(file) for file in files]

    metrics = get_traj_stats(arrays[0], arrays[2], arrays[3], arrays[4], settle_band_meters = 0.05, settle_band_heading = np.pi/12, settle_index_increment = 200)
    
    print(metrics)

if __name__ == "__main__":
    main()