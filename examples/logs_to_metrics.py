import os

import numpy as np

def save_state_metrics(in_dir: str, out_dir: str):
     
    print("\nPOM")
    logs_dir_pom = os.path.join(in_dir, 'pom.log')
    fields = ['ts', 'i', 'posp', 'attp', 'velp', 'avelp', 'accp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'ax', 'ay', 'az', 'sx2', 'sy2', 'sz2', 'sr2', 'sp2', 'sh2', 'svx2', 'svy2', 'svz2', 'swx2', 'swy2', 'swz2', 'sax2', 'say2', 'saz2']

    #indexes = np.arange(7,18+1)
    indexes = np.arange(7,21+1) # including linear acceleration
    data = np.loadtxt(logs_dir_pom, usecols=indexes, skiprows=8, dtype='float64')
    time_raw = np.loadtxt(logs_dir_pom, usecols=0, skiprows=8, dtype='float64')

    #print(data)
    print('Lines:',data.shape[0], '\nCols:', data.shape[1])

    t0 = time_raw[0]
    print("Scaling time using t0 =", t0)
    time_s = time_raw - t0
    # Check the time in ms bteween steps, it should be 1 ms (= 1KHz)
    print((time_s[2]-time_s[1])*1000)
    print((time_s[100]-time_s[99])*1000)
    print((time_s[200]-time_s[199])*1000)

    # unpack variables
    pos = data[:,0:3] # x y z
    att = data[:,3:6] # roll pitch yaw
    vel = data[:,6:9] # vx vy vz
    avel = data[:,9:12] # avx avy avz


    with open(os.path.join(out_dir, "trajectory.npy"), "wb") as file_handle:
        np.save(file_handle, data[:,0:12])

    with open(os.path.join(out_dir, "time_arr.npy"), "wb") as file_handle:
        np.save(file_handle, time_s)

def save_ctrl_metrics(in_dir: str, out_dir: str):
    print("\nROTORCRAFT")
    logs_dir_pom = os.path.join(in_dir, 'rotorcraft.log')

    indexes = np.arange(24,24+24) # 16 bcs 8 commanded vel + 8 measured
    #data = np.loadtxt(logs_dir_pom, usecols=indexes, skiprows=45, dtype='float64')
    data = np.genfromtxt(logs_dir_pom,skip_header=45,filling_values=np.nan,usecols=indexes,loose=True)
    # loose=True to ignore errors due to '-', filling_values=np.nan to fill missing data due to '-'
    # commanded and measured vels has different log times!
    time_raw = np.loadtxt(logs_dir_pom, usecols=0, skiprows=45, dtype='float64')

    #print(data)
    print('Lines:',data.shape[0], '\nCols:', data.shape[1])

    t0 = time_raw[0]
    print("Scaling time using t0 =", t0)
    time_s = time_raw - t0
    print((time_s[1]-time_s[0])*1000)

    # unpack variables
    cmd_v = data[:,0:4] # commanded rotor speeds
    meas_v = data[:,8::3] # measured rotor speeds

    with open(os.path.join(out_dir, "ctrl_input.npy"), "wb") as file_handle:
        np.save(file_handle, data[:,0:4])




if __name__ == "__main__":
     in_dir = "/home/student/mppi_ws/src/fmppi_drone/fmppi_drone/logs_20hz"
     out_dir = "/home/student/out_dir/20hz"

     save_state_metrics(in_dir, out_dir)

     save_ctrl_metrics(in_dir, out_dir)

     reference = np.array([1, 2, 1, 0, 0, 0, 1])

     with open(os.path.join(out_dir, "ref.npy"), "wb") as file_handle:
            np.save(file_handle, reference)
     with open(os.path.join(out_dir, "constraints_arr.npy"), "wb") as file_handle:
            np.save(file_handle, np.zeros((3,1)))

# if save_results:
#         traj_arr = sim.state_traj

#         # with open("ctrl_input.npy", "wb") as file_handle:
#         #     np.save(file_handle, sim.input_traj)
#         # with open("ref.npy", "wb") as file_handle:
#         #     np.save(file_handle, sim.const_reference)
#         with open("time_arr.npy", "wb") as file_handle:
#             np.save(file_handle, time_vect)
#         with open("constraints_arr.npy", "wb") as file_handle:
#             np.save(file_handle, constraints)