import xml.etree.ElementTree as ET
from copy import deepcopy
from numpy.random import random as rand


basic_sphere = """
<geom size="0.05" type="sphere" rgba = "0 0 1 1"/>
  """
original_obstacles = None
original_scene = None

obstacle_path = "examples/bitcraze_crazyflie_2/obstacles.xml"
scene_path = "examples/bitcraze_crazyflie_2/scene.xml"


# make a class

def create_obstacles(obstacle_type):
    tree = ET.parse(obstacle_path)
    global original_obstacles
    original_obstacles = deepcopy(tree)  # save original obstacles.xml for reset
    root = tree.getroot()
    body = root.find('body')

    x_range = 1
    y_range = 1
    z_range = 1
    n_obstacles = 3

    random_positions = []  # make a set to avoid duplicates
    for i in range(n_obstacles):   # generate random positions for the obstacles - perhaps should avoid where the model is generated
        x = rand(1*x_range)[0]  # fix precision
        y = rand(1*y_range)[0]
        z = rand(1*z_range)[0]
        random_positions.append(f"{x} {y} {z}")

    # print(random_positions)
    for obs in range(n_obstacles):   # add obstacles to obstacles.xml
        obstacle_elem = ET.fromstring(obstacle_type)
        obstacle_elem.set("pos", random_positions[obs])
        body.append(obstacle_elem)

    tree.write(obstacle_path)

    return

def load_obstacles(scene, obstacles):    # include obstacles.xml in simulation file
    scene_elem = ET.parse(scene)
    global original_scene # change this?
    original_scene = deepcopy(scene_elem)  # save original scene for reset

    root = scene_elem.getroot()
    world_body = root.find('worldbody')
    include = ET.Element('include')
    include.set("file", obstacles)
    world_body.append(include)
    scene_elem.write(scene)
    return

def reset_xmls():
    global original_scene  # change this?
    global original_obstacles

    original_scene.write(scene_path)
    original_obstacles.write(obstacle_path)  # 'reset' obstacle file
    return

create_obstacles( basic_sphere)
load_obstacles(scene_path, "obstacles.xml")
# reset_xmls()

# falling_sphere =  """"<mujoco>
#     <option gravity =" 0 0 10" />
#     <worldbody>
#         <body name="sphere" pos="0 0 0">
#             <geom size="0.1" type="sphere" rgba = "0 0 1 1"/>
#         </body>
#     </worldbody>
# </mujoco>
# """


# model = mujoco.MjModel.from_xml_string(xml=falling_sphere)
# data = mujoco.MjData(model)
#
# with mujoco.Renderer(model) as renderer:
#     mujoco.mj_forward(model, data)
#     renderer.update_scene(data)
#
#     media.show_image(renderer.render())
#     plt.imshow(renderer.render())
#     # plt.show()


