"""
Setup script for Project 5: Trajectory Optimization with Multiple Initial Motion Plans
This script loads the Panda robot and adds obstacles to the Tesseract environment.
"""

import os
import re
import numpy as np
from pathlib import Path

from tesseract_robotics.tesseract_common import FilesystemPath, GeneralResourceLocator, ResourceLocator, \
    SimpleLocatedResource, Isometry3d, Translation3d, ManipulatorInfo
from tesseract_robotics.tesseract_environment import Environment, AddLinkCommand
from tesseract_robotics.tesseract_scene_graph import Joint, Link, Visual, Collision, JointType_FIXED
from tesseract_robotics.tesseract_geometry import Sphere
from tesseract_robotics_viewer import TesseractViewer


class PandaResourceLocator(ResourceLocator):
    """
    Custom resource locator for Panda robot that handles relative mesh paths.
    It resolves meshes/... paths relative to the panda directory.
    """
    def __init__(self, panda_dir):
        super().__init__()
        self.panda_dir = Path(panda_dir) if isinstance(panda_dir, str) else panda_dir
    
    def locateResource(self, url):
        """
        Locate resource files. Handles:
        - Direct file paths (file:// or absolute paths)
        - Relative paths (meshes/...)
        - Package URLs (package://...)
        """
        # Try direct file path first (absolute or relative)
        if os.path.exists(url):
            return SimpleLocatedResource(url, url, self)
        
        # Handle package:// URLs
        package_match = re.match(r"^package://([^/]+)/(.*)$", url)
        if package_match:
            package_name = package_match.group(1)
            resource_path = package_match.group(2)
            
            # If it's a meshes package, resolve relative to panda_dir
            if package_name == "meshes":
                full_path = self.panda_dir / resource_path
                if full_path.exists():
                    return SimpleLocatedResource(url, str(full_path), self)
                # Try parent directory (in case meshes folder is there)
                parent_meshes = self.panda_dir.parent / "meshes" / resource_path
                if parent_meshes.exists():
                    return SimpleLocatedResource(url, str(parent_meshes), self)
        
        # Handle relative paths (meshes/collision/link0.obj, etc.)
        # Check if it looks like a relative mesh path
        if "/" in url and (url.startswith("meshes/") or not os.path.isabs(url)):
            # Try relative to panda_dir
            full_path = self.panda_dir / url
            if full_path.exists():
                return SimpleLocatedResource(url, str(full_path), self)
            
            # Try with meshes subdirectory
            meshes_path = self.panda_dir / "meshes" / url if not url.startswith("meshes/") else self.panda_dir / url
            if meshes_path.exists():
                return SimpleLocatedResource(url, str(meshes_path), self)
        
        # For missing visual meshes, return a dummy resource so parsing doesn't fail
        # The URDF parser will skip visual if mesh can't be loaded, but collision spheres should still work
        # Return None to let it fail gracefully - the environment might still initialize
        return None


def load_obstacles(obstacles_file):
    """
    Load obstacle positions from a text file.
    Each line should contain [x, y, z] coordinates.
    """
    obstacles = []
    with open(obstacles_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                # Parse [x, y, z] format
                coords = line[1:-1].split(',')
                x = float(coords[0].strip())
                y = float(coords[1].strip())
                z = float(coords[2].strip())
                obstacles.append([x, y, z])
    return obstacles


def add_obstacle_to_environment(env, obstacle_pos, obstacle_id, radius=0.05, parent_link="world"):
    """
    Add a spherical obstacle to the environment.
    
    Args:
        env: Tesseract Environment
        obstacle_pos: [x, y, z] position of the obstacle
        obstacle_id: Unique identifier for the obstacle
        radius: Radius of the sphere (default 0.05m = 5cm)
        parent_link: Parent link to attach obstacle to (default "world")
    """
    # Create sphere link
    obstacle_link = Link(f"obstacle_{obstacle_id}")
    
    # Create visual geometry
    obstacle_visual = Visual()
    obstacle_visual.geometry = Sphere(radius)
    obstacle_link.visual.push_back(obstacle_visual)
    
    # Create collision geometry
    obstacle_collision = Collision()
    obstacle_collision.geometry = Sphere(radius)
    obstacle_link.collision.push_back(obstacle_collision)
    
    # Create fixed joint to attach obstacle
    obstacle_joint = Joint(f"obstacle_{obstacle_id}_joint")
    obstacle_joint.parent_link_name = parent_link
    obstacle_joint.child_link_name = obstacle_link.getName()
    obstacle_joint.type = JointType_FIXED
    
    # Set transform (position of obstacle)
    obstacle_transform = Isometry3d.Identity() * Translation3d(obstacle_pos[0], obstacle_pos[1], obstacle_pos[2])
    obstacle_joint.parent_to_joint_origin_transform = obstacle_transform
    
    # Create and apply command
    add_obstacle_command = AddLinkCommand(obstacle_link, obstacle_joint)
    env.applyCommand(add_obstacle_command)
    
    print(f"Added obstacle {obstacle_id} at position [{obstacle_pos[0]:.2f}, {obstacle_pos[1]:.2f}, {obstacle_pos[2]:.2f}]")


def setup_panda_environment(assets_dir="assets"):
    """
    Set up the Tesseract environment with Panda robot and obstacles.
    
    Args:
        assets_dir: Directory containing panda/ and obstacles/ folders
    
    Returns:
        env: Initialized Tesseract Environment
        manip_info: ManipulatorInfo for the Panda robot
        joint_names: List of joint names
    """
    # Get absolute paths (go up from project/ to root)
    script_dir = Path(__file__).parent.absolute()
    # If script is in project/, go up one level to root
    if script_dir.name == "project":
        root_dir = script_dir.parent
    else:
        root_dir = script_dir
    assets_path = root_dir / assets_dir
    
    panda_dir = assets_path / "panda"
    obstacles_file = assets_path / "obstacles" / "obstacles.txt"
    
    # Use spherized URDF (works with mesh files via PandaResourceLocator)
    # Falls back to regular URDF if spherized doesn't exist
    urdf_path = panda_dir / "panda_spherized.urdf"
    if not urdf_path.exists():
        urdf_path = panda_dir / "panda.urdf"
    
    # Use minimal SRDF (no hand group) - create if it doesn't exist
    srdf_path = panda_dir / "panda_minimal.srdf"
    if not srdf_path.exists():
        # Create minimal SRDF on the fly
        minimal_srdf_content = """<?xml version="1.0" encoding="utf-8"?>
<robot name="panda">
  <group name="panda_arm">
    <chain base_link="panda_link0" tip_link="panda_link8"/>
  </group>
  <group_state group="panda_arm" name="ready">
    <joint name="panda_joint1" value="0"/>
    <joint name="panda_joint2" value="-0.785"/>
    <joint name="panda_joint3" value="0"/>
    <joint name="panda_joint4" value="-2.356"/>
    <joint name="panda_joint5" value="0"/>
    <joint name="panda_joint6" value="1.571"/>
    <joint name="panda_joint7" value="0.785"/>
  </group_state>
  <virtual_joint child_link="panda_link0" name="virtual_joint" parent_frame="world" type="floating"/>
  <disable_collisions link1="panda_link0" link2="panda_link1" reason="Adjacent"/>
  <disable_collisions link1="panda_link0" link2="panda_link2" reason="Never"/>
  <disable_collisions link1="panda_link0" link2="panda_link3" reason="Never"/>
  <disable_collisions link1="panda_link0" link2="panda_link4" reason="Never"/>
  <disable_collisions link1="panda_link1" link2="panda_link2" reason="Adjacent"/>
  <disable_collisions link1="panda_link1" link2="panda_link3" reason="Never"/>
  <disable_collisions link1="panda_link1" link2="panda_link4" reason="Never"/>
  <disable_collisions link1="panda_link2" link2="panda_link3" reason="Adjacent"/>
  <disable_collisions link1="panda_link2" link2="panda_link4" reason="Never"/>
  <disable_collisions link1="panda_link2" link2="panda_link6" reason="Never"/>
  <disable_collisions link1="panda_link3" link2="panda_link4" reason="Adjacent"/>
  <disable_collisions link1="panda_link3" link2="panda_link5" reason="Never"/>
  <disable_collisions link1="panda_link3" link2="panda_link6" reason="Never"/>
  <disable_collisions link1="panda_link3" link2="panda_link7" reason="Never"/>
  <disable_collisions link1="panda_link4" link2="panda_link5" reason="Adjacent"/>
  <disable_collisions link1="panda_link4" link2="panda_link6" reason="Never"/>
  <disable_collisions link1="panda_link4" link2="panda_link7" reason="Never"/>
  <disable_collisions link1="panda_link5" link2="panda_link6" reason="Adjacent"/>
  <disable_collisions link1="panda_link6" link2="panda_link7" reason="Adjacent"/>
</robot>
"""
        srdf_path.write_text(minimal_srdf_content, encoding='utf-8')
        print(f"[INFO] Created minimal SRDF: {srdf_path}")
    
    print("Setting up Panda robot environment...")
    print(f"URDF: {urdf_path}")
    print(f"SRDF: {srdf_path}")
    
    # Check if files exist
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    if not srdf_path.exists():
        raise FileNotFoundError(f"SRDF file not found: {srdf_path}")
    if not obstacles_file.exists():
        raise FileNotFoundError(f"Obstacles file not found: {obstacles_file}")
    
    # Initialize custom resource locator for Panda
    # This handles relative mesh paths in the URDF (meshes/...)
    locator = PandaResourceLocator(panda_dir)
    
    # Initialize environment with Panda robot
    env = Environment()
    
    # Convert to FilesystemPath (can use string directly)
    urdf_fs_path = FilesystemPath(str(urdf_path))
    srdf_fs_path = FilesystemPath(str(srdf_path))
    
    # Initialize environment
    if not env.init(urdf_fs_path, srdf_fs_path, locator):
        raise RuntimeError("Failed to initialize environment with Panda URDF/SRDF")
    
    print("[OK] Panda robot loaded successfully")
    
    # Configure manipulator info for Panda
    manip_info = ManipulatorInfo()
    manip_info.tcp_frame = "panda_link8"  # End effector frame
    manip_info.manipulator = "panda_arm"  # From SRDF group name
    manip_info.working_frame = "panda_link0"    # Base frame
    
    # Get joint names (Panda has 7 joints)
    joint_names = [f"panda_joint{i+1}" for i in range(7)]
    
    # Load obstacles
    print("\nLoading obstacles...")
    obstacle_positions = load_obstacles(obstacles_file)
    print(f"Found {len(obstacle_positions)} obstacles")
    
    # Add obstacles to environment
    # Use panda_link0 as parent (base link) since "world" doesn't exist in the scene graph
    for i, obs_pos in enumerate(obstacle_positions):
        # Use radius 0.1m (10cm) for obstacles - adjust as needed
        add_obstacle_to_environment(env, obs_pos, i, radius=0.1, parent_link="panda_link0")
    
    print(f"\n[OK] Successfully added {len(obstacle_positions)} obstacles to environment")
    
    # Set initial robot state (all joints at zero)
    initial_joint_positions = np.zeros(7)
    env.setState(joint_names, initial_joint_positions)
    
    print("\n[OK] Environment setup complete!")
    print(f"  Robot: Panda (7-DOF)")
    print(f"  Joints: {joint_names}")
    print(f"  Manipulator: {manip_info.manipulator}")
    print(f"  TCP Frame: {manip_info.tcp_frame}")
    print(f"  Working Frame: {manip_info.working_frame}")
    print(f"  Obstacles: {len(obstacle_positions)}")
    
    return env, manip_info, joint_names


def visualize_setup(env, manip_info, joint_names, initial_joint_positions=None):
    """
    Visualize the environment setup using TesseractViewer.
    
    Args:
        env: Tesseract Environment
        manip_info: ManipulatorInfo
        joint_names: List of joint names
        initial_joint_positions: Initial joint positions (default: zeros)
    """
    if initial_joint_positions is None:
        initial_joint_positions = np.zeros(len(joint_names))
    
    print("\nStarting viewer...")
    viewer = TesseractViewer()
    
    # Update environment in viewer
    viewer.update_environment(env, [0, 0, 0])
    
    # Set initial joint positions for visualization
    viewer.update_joint_positions(joint_names, initial_joint_positions)
    
    # Start viewer server
    viewer.start_serve_background()
    
    print("[OK] Viewer started!")
    print("  Open http://localhost:8000 in your browser to see the visualization")
    
    return viewer


if __name__ == "__main__":
    # Setup environment
    env, manip_info, joint_names = setup_panda_environment()
    
    # Visualize
    viewer = visualize_setup(env, manip_info, joint_names)
    
    # Keep script running to view
    print("\nPress Enter to exit...")
    input()
    
    print("Done!")

