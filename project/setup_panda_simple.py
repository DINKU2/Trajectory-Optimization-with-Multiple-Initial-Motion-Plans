"""
Simplified Panda environment setup for Project 5.
Keeps required functionality, removes unnecessary abstraction.
"""

import os
import re
import numpy as np
from pathlib import Path

from tesseract_robotics.tesseract_common import (
    FilesystemPath, ResourceLocator, SimpleLocatedResource,
    Isometry3d, Translation3d, ManipulatorInfo
)
from tesseract_robotics.tesseract_environment import Environment, AddLinkCommand
from tesseract_robotics.tesseract_scene_graph import Joint, Link, Visual, Collision, JointType_FIXED
from tesseract_robotics.tesseract_geometry import Sphere
from tesseract_robotics_viewer import TesseractViewer


class PandaResourceLocator(ResourceLocator):
    """Required: Handles mesh paths in URDF (meshes/collision/link0.obj)"""
    def __init__(self, panda_dir):
        super().__init__()
        self.panda_dir = Path(panda_dir)
    
    def locateResource(self, url):
        # Handle relative mesh paths (meshes/collision/link0.obj)
        if "/" in url and (url.startswith("meshes/") or not os.path.isabs(url)):
            full_path = self.panda_dir / url
            if full_path.exists():
                return SimpleLocatedResource(url, str(full_path), self)
        return None


def setup_panda_environment(assets_dir="assets"):
    """Setup Tesseract environment with Panda robot and obstacles."""
    # Get paths
    root_dir = Path(__file__).parent.parent.absolute()
    panda_dir = root_dir / assets_dir / "panda"
    urdf_path = panda_dir / "panda_spherized.urdf"
    srdf_path = panda_dir / "panda_minimal.srdf"
    obstacles_file = root_dir / assets_dir / "obstacles" / "obstacles.txt"
    
    # Use minimal SRDF (avoids hand group parsing error)
    if not srdf_path.exists():
        raise FileNotFoundError(f"Minimal SRDF not found: {srdf_path}. Run create_minimal_srdf.py first.")
    
    # Load robot
    locator = PandaResourceLocator(panda_dir)
    env = Environment()
    if not env.init(FilesystemPath(str(urdf_path.absolute())), FilesystemPath(str(srdf_path.absolute())), locator):
        raise RuntimeError("Failed to load Panda robot")
    
    # Configure manipulator
    manip_info = ManipulatorInfo()
    manip_info.tcp_frame = "panda_link8"
    manip_info.manipulator = "panda_arm"
    manip_info.working_frame = "panda_link0"
    joint_names = [f"panda_joint{i+1}" for i in range(7)]
    env.setState(joint_names, np.zeros(7))
    
    # Load obstacles
    obstacles = []
    with open(obstacles_file, 'r') as f:
        for line in f:
            if line.strip().startswith('[') and line.strip().endswith(']'):
                coords = [float(x.strip()) for x in line.strip()[1:-1].split(',')]
                obstacles.append(coords)
    
    # Add obstacles
    for i, pos in enumerate(obstacles):
        link = Link(f"obstacle_{i}")
        sphere = Sphere(0.1)
        link.visual.push_back(Visual())
        link.visual[0].geometry = sphere
        link.collision.push_back(Collision())
        link.collision[0].geometry = sphere
        
        joint = Joint(f"obstacle_{i}_joint")
        joint.parent_link_name = "panda_link0"
        joint.child_link_name = link.getName()
        joint.type = JointType_FIXED
        joint.parent_to_joint_origin_transform = Isometry3d.Identity() * Translation3d(*pos)
        
        env.applyCommand(AddLinkCommand(link, joint))
    
    print(f"[OK] Environment ready: Panda robot + {len(obstacles)} obstacles")
    return env, manip_info, joint_names


def visualize(env, manip_info, joint_names):
    """Start viewer."""
    viewer = TesseractViewer()
    viewer.update_environment(env, [0, 0, 0])
    viewer.update_joint_positions(joint_names, np.zeros(7))
    viewer.start_serve_background()
    print("[OK] Viewer: http://localhost:8000")
    return viewer


if __name__ == "__main__":
    env, manip_info, joint_names = setup_panda_environment()
    viewer = visualize(env, manip_info, joint_names)
    input("Press Enter to exit...")

