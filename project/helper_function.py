import sys
import numpy as np

# Suppress SWIG memory leak warnings (noise only)
class SWIGWarningFilter:
    """Filter to suppress SWIG memory leak warnings"""
    def __init__(self):
        self.stderr = sys.stderr
        self.filtered = False
    
    def write(self, text):
        if 'memory leak' in text.lower() and 'swig' in text.lower():
            return  # Suppress SWIG memory leak warnings
        self.stderr.write(text)
    
    def flush(self):
        self.stderr.flush()


# Visual marker helper for debugging target positions
def add_target_marker(env, joint_group, joint_names, joint_config, marker_name, 
                      tcp_frame="panda_link8", marker_radius=0.1, parent_link="panda_link0"):
    """
    Add a visual marker (sphere) at the end-effector position for a given joint configuration.
    
    Args:
        env: Tesseract Environment
        joint_group: JointGroup object (e.g., from env.getJointGroup("panda_arm"))
        joint_names: List of joint names
        joint_config: Joint configuration (numpy array)
        marker_name: Name for the marker link (e.g., "start_marker" or "goal_marker")
        tcp_frame: Name of the TCP frame (default: "panda_link8")
        marker_radius: Radius of the marker sphere (default: 0.1m)
        parent_link: Parent link to attach marker to (default: "panda_link0")
    
    Returns:
        The end-effector position as a numpy array [x, y, z]
    """
    from tesseract_robotics.tesseract_scene_graph import Joint, Link, Visual, Collision, JointType_FIXED
    from tesseract_robotics.tesseract_geometry import Sphere
    from tesseract_robotics.tesseract_environment import AddLinkCommand
    from tesseract_robotics.tesseract_common import Isometry3d, Translation3d
    
    # Calculate forward kinematics to get end-effector position
    transforms = joint_group.calcFwdKin(joint_config)
    
    # Get the transform of the TCP frame
    if tcp_frame not in transforms:
        print(f"Warning: TCP frame '{tcp_frame}' not found in forward kinematics result")
        return None
    
    tcp_transform = transforms[tcp_frame]
    ee_position = tcp_transform.translation()
    # translation() returns a numpy array (Eigen::Vector3d), convert to 1D array
    if isinstance(ee_position, np.ndarray):
        ee_pos_array = ee_position.flatten()[:3]  # Ensure it's a 1D array with 3 elements
    else:
        # Fallback if it's not a numpy array (shouldn't happen, but just in case)
        ee_pos_array = np.array([ee_position[0], ee_position[1], ee_position[2]])
    
    # Create visual marker (sphere)
    marker_link = Link(marker_name)
    
    # Visual geometry
    marker_visual = Visual()
    marker_visual.geometry = Sphere(marker_radius)
    marker_link.visual.push_back(marker_visual)
    
    # Collision geometry (optional, but helps with visualization)
    marker_collision = Collision()
    marker_collision.geometry = Sphere(marker_radius)
    marker_link.collision.push_back(marker_collision)
    
    # Fixed joint to attach marker to parent link
    marker_joint = Joint(f"{marker_name}_joint")
    marker_joint.parent_link_name = parent_link
    marker_joint.child_link_name = marker_link.getName()
    marker_joint.type = JointType_FIXED
    marker_joint.parent_to_joint_origin_transform = Isometry3d.Identity() * Translation3d(ee_pos_array[0], ee_pos_array[1], ee_pos_array[2])
    
    # Add marker to environment
    env.applyCommand(AddLinkCommand(marker_link, marker_joint))
    
    print(f"Added {marker_name} at position: [{ee_pos_array[0]:.3f}, {ee_pos_array[1]:.3f}, {ee_pos_array[2]:.3f}]")
    
    return ee_pos_array

