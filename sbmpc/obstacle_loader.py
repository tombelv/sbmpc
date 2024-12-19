import xml.etree.ElementTree as ET
from copy import deepcopy
import random

basic_sphere =  """<geom size="0.05" type="sphere" rgba = "0 0 1 1"/>"""
inertial = """<inertial pos="0 0 0"  mass="1.0" fullinertia="0.0031887 0.0032245 0.0013857 -0.0000038 -0.0000881 0.0000846"/>"""
obstacle_path = "examples/bitcraze_crazyflie_2/obstacles.xml"
scene_path = "examples/bitcraze_crazyflie_2/scene.xml"


class ObstacleLoader():
    def __init__(self, obstacle_path=obstacle_path, scene_path=scene_path):
        self.obstacle_path = obstacle_path
        self.scene_path = scene_path

        tree = ET.parse(self.obstacle_path)  # save original scene and obstacle files for reset
        self.original_obstacles = deepcopy(tree)
        scene_elem = ET.parse(self.scene_path)
        self.original_scene = deepcopy(scene_elem)

    def create_obstacles(self,obstacle_type=basic_sphere):
        tree = ET.parse(self.obstacle_path)
        root = tree.getroot()
        worldbody = root.find('worldbody')
        body = worldbody.find('body')

        x_range = 0.5  # can scale x,y and z to cover a wider area
        y_range = 0.5
        z_range = 0.5
        n_obstacles = 3

        random_positions = []  # make a set to avoid duplicates
        for i in range(n_obstacles):   # generate random positions for the obstacles - perhaps should avoid where the model is generated
            x = round(random.uniform(-1*x_range,1*x_range),3)
            y = round(random.uniform(-1*y_range,1*y_range),3)
            z = round(random.uniform(0,1*z_range),3)  # don't spawn below floor level
            random_positions.append(f"{x} {y} {z}")
        print(f"Created obstacles at {random_positions}")

        inertial_props = ET.fromstring(inertial) # add some physical properties for collisions TODO - get collisions working
        body.append(inertial_props)
        for obs in range(n_obstacles):   # add obstacles to obstacles.xml
            obstacle_elem = ET.fromstring(obstacle_type)
            obstacle_elem.set("pos", random_positions[obs])
        
            body.append(obstacle_elem)
      
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





