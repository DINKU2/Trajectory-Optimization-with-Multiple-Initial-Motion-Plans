import os
import re
import numpy as np
from pathlib import Path

from tesseract_robotics.tesseract_common import (
    FilesystemPath, GeneralResourceLocator, ResourceLocator, SimpleLocatedResource,
    Isometry3d, Translation3d, ManipulatorInfo
)
from tesseract_robotics.tesseract_environment import Environment, AddLinkCommand
from tesseract_robotics.tesseract_scene_graph import Joint, Link, Visual, Collision, JointType_FIXED
from tesseract_robotics.tesseract_geometry import Sphere
from tesseract_robotics_viewer import TesseractViewer

# Set TESSERACT_RESOURCE_PATH to include project/assets/
assets_path = Path(__file__).parent / "assets"
current_path = os.environ.get("TESSERACT_RESOURCE_PATH", "")
if str(assets_path) not in current_path:
    if current_path:
        separator = ";" if os.name == 'nt' else ":"
        os.environ["TESSERACT_RESOURCE_PATH"] = f"{current_path}{separator}{assets_path}"
    else:
        os.environ["TESSERACT_RESOURCE_PATH"] = str(assets_path)

locator = GeneralResourceLocator()
env = Environment()
urdf_path_str = str(Path(__file__).parent / "assets" / "panda" / "panda.urdf")
srdf_path_str = str(Path(__file__).parent / "assets" / "panda" / "panda.srdf")
urdf_path = FilesystemPath(urdf_path_str)
srdf_path = FilesystemPath(srdf_path_str)
assert env.init(urdf_path, srdf_path, locator)

# Set initial robot state
joint_names = [f"panda_joint{i+1}" for i in range(7)]  # Panda has 7 joints
initial_joint_positions = np.zeros(7)  # All joints at zero position
env.setState(joint_names, initial_joint_positions)

# Parse obstacles from obstacles.txt and add them as spheres
obstacles_file = Path(__file__).parent / "assets" / "obstacles" / "obstacles.txt"
obstacle_radius = 0.2

# Parse obstacle positions from file
obstacle_positions = []
with open(obstacles_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('['):
            # Remove brackets and parse coordinates
            coords = [float(x.strip()) for x in line[1:-1].split(',')]
            obstacle_positions.append(coords)

# Add each obstacle as a sphere
for i, pos in enumerate(obstacle_positions):
    obstacle_link = Link(f"obstacle_{i}")
    
    # Visual and collision geometry
    obstacle_visual = Visual()
    obstacle_visual.geometry = Sphere(obstacle_radius)
    obstacle_link.visual.push_back(obstacle_visual)
    
    obstacle_collision = Collision()
    obstacle_collision.geometry = Sphere(obstacle_radius)
    obstacle_link.collision.push_back(obstacle_collision)
    
    # Fixed joint to attach obstacle
    obstacle_joint = Joint(f"obstacle_{i}_joint")
    obstacle_joint.parent_link_name = "panda_link0"
    obstacle_joint.child_link_name = obstacle_link.getName()
    obstacle_joint.type = JointType_FIXED
    obstacle_joint.parent_to_joint_origin_transform = Isometry3d.Identity() * Translation3d(*pos)
    
    env.applyCommand(AddLinkCommand(obstacle_link, obstacle_joint))

# Start Tesseract viewer
viewer = TesseractViewer()
viewer.update_environment(env, [0, 0, 0])  # Update environment with offset [0,0,0]
viewer.update_joint_positions(joint_names, initial_joint_positions)  # Set initial joint positions
viewer.start_serve_background()  # Start the web server in background

print("Tesseract viewer started!")
print("Open your browser and navigate to the URL shown above (typically http://localhost:8000)")
print("Press Enter to exit...")
input()





