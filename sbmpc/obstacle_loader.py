import xml.etree.ElementTree as ET
from copy import deepcopy
import random
import numpy as np
import mujoco

basic_sphere =  """<geom size="0.05" type="sphere" rgba = "1 0.1 0.2 1"/>"""
inertial = """<inertial pos="0 0 0"  mass="1.0" fullinertia="0.0031887 0.0032245 0.0013857 -0.0000038 -0.0000881 0.0000846"/>"""
obstacle_path = "examples/bitcraze_crazyflie_2/obstacles.xml"
scene_path = "examples/bitcraze_crazyflie_2/scene.xml"

'"examples/bitcraze_crazyflie_2/obstacles.xml'

class ObstacleLoader():
    def __init__(self, obstacle_path=obstacle_path, scene_path=scene_path):
        self.obstacle_path = obstacle_path
        self.scene_path = scene_path

        tree = ET.parse(self.obstacle_path)  # save original scene and obstacle files for reset
        self.original_obstacles = deepcopy(tree)
        scene_elem = ET.parse(self.scene_path)
        self.original_scene = deepcopy(scene_elem)

        self.obs_pos = []
        self.radius = 0.05 # set on line 7

        self.model = mujoco.MjModel.from_xml_path(scene_path) 
        self.data = mujoco.MjData(self.model)

        self.n_obstacles = 3


    def create_obstacles(self,obstacle_type=basic_sphere):
        tree = ET.parse(self.obstacle_path)
        root = tree.getroot()
        worldbody = root.find('worldbody')
    
        x_range = 0.5  # can scale x, y and z to cover a wider area
        y_range = 0.5
        z_range = 0.5

        random_positions = [] 
        for i in range(self.n_obstacles):   # generate random positions for the obstacles - perhaps should avoid where the model is generated
            x = round(random.uniform(-1*x_range,1*x_range),3)
            y = round(random.uniform(-1*y_range,1*y_range),3)
            z = round(random.uniform(0,1*z_range), 3)  # don't spawn below floor level
            random_positions.append(f"{x} {y} {z}")
            self.obs_pos.append([x, y, z])
        self.obs_pos = np.concatenate(self.obs_pos, axis=0) 
        print(f"Created obstacles at {random_positions}")

        # inertial_props = ET.fromstring(inertial) # add some physical properties for collisions 
        
        for obs in range(self.n_obstacles):   # add obstacles to obstacles.xml
            body = ET.Element("body", {"name" : "obstacle " + str(obs+1)})
            # body.append(inertial_props)
            obstacle_elem = ET.fromstring(obstacle_type)
            obstacle_elem.set("pos", random_positions[obs])
        
            body.append(obstacle_elem)
            worldbody.append(body)
            ET.indent(tree, '   ')
      
        tree.write(self.obstacle_path)

        return

    def load_obstacles(self):    # include obstacles.xml in scene
        scene = self.scene_path
        scene_elem = ET.parse(scene)
        root = scene_elem.getroot()

        include = ET.Element('include')
        include.set("file", "obstacles.xml")

        root.insert(0,include)
        scene_elem.write(scene)
        return

    def reset_xmls(self):  # call to clear xmls i.e. remove obstacles
        self.original_scene.write(self.scene_path)
        self.original_obstacles.write(self.obstacle_path)
        return

    def get_obstacle_trajectory(self, iters, function=None):  # for each iteration generate the position of each obstacle according to the chosen trajectory function
        n = self.n_obstacles
        traj = np.zeros((iters,n,3))
        step = 0.05
        r = 0.4
        ang = 0 
        origin = traj[0]

        if function == "circle": 
            for t in range(iters):
                x = r*np.cos(ang) 
                y = r*np.sin(ang) 
                obstacle_pos = []
                for i in range(n):
                    obstacle_pos.append([origin[i][0] + x, origin[i][1] + y, origin[i][2] + (x+y)/2])
                traj[t] = obstacle_pos
                ang += step

        elif function == "diagonal":
            step = 0.005
            for t in range(iters):
                obstacle_pos = []
                for i in range(n):
                    obstacle_pos.append(origin[i] + step)
                traj[t] = obstacle_pos
                origin = traj[t-1]
        
        elif function == "sine": 
            r = 0.05
            for t in range(iters):
                x = y = z = r*np.sin(ang) 
                obstacle_pos = []
                for i in range(n):
                    obstacle_pos.append([origin[i][0] + x, origin[i][1] + y, origin[i][2] + z])
                traj[t] = obstacle_pos
                origin = traj[t-1]
                ang += step

        else:  
            pass   # if no valid funtion selected then keep obstacles stationary

        traj = np.reshape(traj, (iters,n*3))
        return traj