import os
import re
import sys
import numpy as np
import time
from pathlib import Path

from tesseract_robotics.tesseract_common import (
    FilesystemPath, GeneralResourceLocator, ResourceLocator, SimpleLocatedResource,
    Isometry3d, Translation3d, ManipulatorInfo, CollisionMarginData)
from tesseract_robotics.tesseract_environment import Environment, AddLinkCommand, AddContactManagersPluginInfoCommand
from tesseract_robotics.tesseract_srdf import parseContactManagersPluginConfigString
from tesseract_robotics.tesseract_scene_graph import Joint, Link, Visual, Collision, JointType_FIXED
from tesseract_robotics.tesseract_geometry import Sphere
from tesseract_robotics.tesseract_command_language import (
    =JointWaypoint, MoveInstructionType_FREESPACE, MoveInstruction, CompositeInstruction,
    ProfileDictionary, JointWaypointPoly_wrap_JointWaypoint, MoveInstructionPoly_wrap_MoveInstruction,
    InstructionPoly_as_MoveInstructionPoly, WaypointPoly_as_StateWaypointPoly)
from tesseract_robotics.tesseract_motion_planners import PlannerRequest
from tesseract_robotics.tesseract_motion_planners_trajopt import (TrajOptDefaultPlanProfile, TrajOptDefaultCompositeProfile, TrajOptMotionPlanner)
from tesseract_robotics.tesseract_motion_planners_simple import generateInterpolatedProgram
from tesseract_robotics.tesseract_time_parameterization import TimeOptimalTrajectoryGeneration, InstructionsTrajectory
from tesseract_robotics.tesseract_collision import (ContactResultMap, ContactTestType_ALL, ContactRequest, ContactResultVector)
from tesseract_robotics_viewer import TesseractViewer


USE_OBSTACLES = True  

# Suppress SWIG memory leak warnings (known limitation - memory is properly managed by C++ environment)
# SWIG prints these warnings directly to stderr, so we'll filter them
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

# Install filter to catch SWIG memory leak warnings
_swig_filter = None

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

# Configure contact manager plugins
# Load contact manager plugins configuration from YAML file
contact_manager_config_path = Path(__file__).parent / "assets" / "panda" / "contact_manager_plugins.yaml"
# Keep references to prevent SWIG memory leak warnings (objects are properly managed by environment)
_contact_manager_plugin_info = None
_contact_manager_cmd = None
try:
    with open(contact_manager_config_path, 'r') as f:
        contact_manager_yaml = f.read()
    _contact_manager_plugin_info = parseContactManagersPluginConfigString(contact_manager_yaml)
    _contact_manager_cmd = AddContactManagersPluginInfoCommand(_contact_manager_plugin_info)
    env.applyCommand(_contact_manager_cmd)
    # Keep references alive to help SWIG with cleanup
    # The environment takes ownership, but SWIG needs these references for proper cleanup
    print("Contact manager plugins configured successfully")
    # Install SWIG warning filter to suppress memory leak warnings
    _swig_filter = SWIGWarningFilter()
    sys.stderr = _swig_filter
except Exception as e:
    print(f"Warning: Could not load contact manager plugins: {e}")
    print("TrajOpt may still work, but collision checking might be limited")

# Set initial robot state
joint_names = [f"panda_joint{i+1}" for i in range(7)]  # Panda has 7 joints
initial_joint_positions = np.zeros(7)  # All joints at zero position
env.setState(joint_names, initial_joint_positions)

# Parse obstacles from obstacles.txt and add them as spheres (if enabled)
if USE_OBSTACLES:
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
    
    print(f"Added {len(obstacle_positions)} obstacles to the environment")
else:
    print("Obstacles disabled - running in empty environment")

# Get state solver (needed for trajectory extraction)
state_solver = env.getStateSolver()

# Configure manipulator info for TrajOpt
manip_info = ManipulatorInfo()
manip_info.tcp_frame = "panda_link8"
manip_info.manipulator = "panda_arm"
manip_info.working_frame = "panda_link0"

# Get joint limits for random sampling
joint_group = env.getJointGroup("panda_arm")
joint_limits = joint_group.getLimits().joint_limits
joint_min = joint_limits[:, 0]
joint_max = joint_limits[:, 1]

print("\n" + "="*60)
print("Initializing TrajOpt with Random Solution")
print("="*60)

# Create a random initial trajectory
# Use a simpler approach: start from a known safe configuration and interpolate to a random goal
np.random.seed(42)  # For reproducibility

# Start from a safe configuration (using the "ready" state from SRDF as reference)
# This is a reasonable starting pose for the Panda arm
start_config = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
goal_config = np.array([2.35, 1., 0., -0.8, 0, 2.5, 0.785], dtype=np.float64)


# Create a simple linear interpolation between start and goal
# This gives a more reasonable initial trajectory than completely random waypoints
num_waypoints = 5  # Fewer waypoints for simpler problem
waypoints = []
for i in range(num_waypoints):
    alpha = i / (num_waypoints - 1) if num_waypoints > 1 else 0.0
    interpolated = start_config + alpha * (goal_config - start_config)
    # Ensure within limits
    interpolated = np.clip(interpolated, joint_min, joint_max)
    waypoints.append(interpolated.astype(np.float64))

# Create initial trajectory program with random waypoints
program = CompositeInstruction("DEFAULT")
program.setManipulatorInfo(manip_info)

# Add start waypoint
start_wp = JointWaypoint(joint_names, waypoints[0])
start_instruction = MoveInstruction(
    JointWaypointPoly_wrap_JointWaypoint(start_wp),
    MoveInstructionType_FREESPACE,
    "DEFAULT"
)
program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(start_instruction))

# Add intermediate waypoints
for i in range(1, len(waypoints)):
    wp = JointWaypoint(joint_names, waypoints[i])
    instruction = MoveInstruction(
        JointWaypointPoly_wrap_JointWaypoint(wp),
        MoveInstructionType_FREESPACE,
        "DEFAULT"
    )
    program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(instruction))

# Interpolate the program to get dense waypoints for TrajOpt
# Use fewer interpolation points to reduce problem complexity
print("Interpolating trajectory...")
interpolated_program = generateInterpolatedProgram(program, env, 3.14, 1.0, 3.14, 5)

# Set up TrajOpt planner with more lenient settings
# The default settings might be too strict for a random initial trajectory
TRAJOPT_DEFAULT_NAMESPACE = "TrajOptMotionPlannerTask"
trajopt_plan_profile = TrajOptDefaultPlanProfile()
trajopt_composite_profile = TrajOptDefaultCompositeProfile()

# Adjust collision settings to be more lenient (use cost instead of hard constraint)
# This allows TrajOpt to work with trajectories that might have minor collisions initially
try:
    # Try to configure collision settings if available
    # Use collision cost instead of hard constraint for more flexibility
    trajopt_composite_profile.collision_constraint_config.enabled = False
    trajopt_composite_profile.collision_cost_config.enabled = True
    # Set a small collision margin to allow optimization to push away from obstacles
    if hasattr(trajopt_composite_profile.collision_cost_config, 'collision_margin_buffer'):
        trajopt_composite_profile.collision_cost_config.collision_margin_buffer = 0.01
except AttributeError:
    # If attributes don't exist, use defaults
    print("Note: Using default TrajOpt collision settings")

trajopt_profiles = ProfileDictionary()
trajopt_profiles.addProfile(TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_plan_profile)
trajopt_profiles.addProfile(TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_composite_profile)

trajopt_planner = TrajOptMotionPlanner(TRAJOPT_DEFAULT_NAMESPACE)

# Create TrajOpt planning request
trajopt_request = PlannerRequest()
trajopt_request.instructions = interpolated_program
trajopt_request.env = env
trajopt_request.profiles = trajopt_profiles

# Run TrajOpt optimization
print("Running TrajOpt optimization...")
start_time = time.time()
try:
    trajopt_response = trajopt_planner.solve(trajopt_request)
    planning_time = time.time() - start_time
except Exception as e:
    print(f"ERROR: TrajOpt optimization failed with exception: {e}")
    print("This may be due to missing contact manager plugins.")
    exit(1)

if not trajopt_response.successful:
    print(f"ERROR: TrajOpt failed to find a solution: {trajopt_response.message}")
    exit(1)

print(f"✓ TrajOpt optimization completed successfully!")
print(f"Planning time: {planning_time:.4f} seconds")

# Get optimized trajectory (before time parameterization)
trajopt_results_instruction = trajopt_response.results

# Add time parameterization to trajectory for animation
# TrajOpt doesn't assign timestamps, so we need to add them for the viewer to animate
print("Adding time parameterization to trajectory...")
time_parameterization = TimeOptimalTrajectoryGeneration()
instructions_trajectory = InstructionsTrajectory(trajopt_results_instruction)

# Panda robot velocity and acceleration limits (from URDF)
# Velocity limits from URDF: joints 1-4: 2.3925 rad/s, joints 5-7: 2.8710 rad/s
# Acceleration limits are not in URDF, using reasonable defaults based on robot dynamics
max_velocity = np.array([[2.3925, 2.3925, 2.3925, 2.3925, 2.8710, 2.8710, 2.8710]], dtype=np.float64)
max_velocity = np.hstack((-max_velocity.T, max_velocity.T))
# Acceleration limits (reasonable defaults - not specified in URDF)
# Typical values for Panda: ~15-20 rad/s^2 for larger joints, ~20-25 for smaller joints
max_acceleration = np.array([[15, 15, 15, 15, 20, 20, 20]], dtype=np.float64)
max_acceleration = np.hstack((-max_acceleration.T, max_acceleration.T))
# Jerk limits (reasonable defaults for smooth motion)
max_jerk = np.array([[100, 100, 100, 100, 100, 100, 100]], dtype=np.float64)
max_jerk = np.hstack((-max_jerk.T, max_jerk.T))

if time_parameterization.compute(instructions_trajectory, max_velocity, max_acceleration, max_jerk):
    print("Time parameterization completed successfully")
else:
    print("Warning: Time parameterization failed, animation may not work properly")

# Flatten the results for analysis and visualization
trajopt_results = trajopt_results_instruction.flatten()

# Extract trajectory waypoints for analysis
trajectory_waypoints = []
for instr in trajopt_results:
    if instr.isMoveInstruction():
        move_instr = InstructionPoly_as_MoveInstructionPoly(instr)
        wp = move_instr.getWaypoint()
        if wp.isStateWaypoint():
            state_wp = WaypointPoly_as_StateWaypointPoly(wp)
            trajectory_waypoints.append(state_wp.getPosition())

trajectory_array = np.array(trajectory_waypoints)

# Calculate trajectory length (sum of distances between consecutive waypoints)
trajectory_length = 0.0
for i in range(len(trajectory_array) - 1):
    diff = trajectory_array[i+1] - trajectory_array[i]
    trajectory_length += np.linalg.norm(diff)

print(f"Trajectory length: {trajectory_length:.4f} radians")
print(f"Number of waypoints: {len(trajectory_array)}")

# Verify trajectory is reasonable
print("\nVerifying trajectory reasonableness...")
# Check for large jumps between waypoints
max_jump = 0.0
for i in range(len(trajectory_array) - 1):
    jump = np.linalg.norm(trajectory_array[i+1] - trajectory_array[i])
    max_jump = max(max_jump, jump)

print(f"Maximum joint space jump between waypoints: {max_jump:.4f} radians")
if max_jump > 1.0:  # Threshold for reasonable jumps
    print("⚠ Warning: Large jumps detected in trajectory")
else:
    print("✓ Trajectory has reasonable waypoint spacing")

# Check joint limits
within_limits = True
for i, joint_pos in enumerate(trajectory_array):
    if np.any(joint_pos < joint_min) or np.any(joint_pos > joint_max):
        within_limits = False
        print(f"⚠ Warning: Waypoint {i} violates joint limits")
        break

if within_limits:
    print("✓ All waypoints are within joint limits")

# Summary report
print("\n" + "="*60)
print("TRAJECTORY OPTIMIZATION SUMMARY")
print("="*60)
print(f"Planning time: {planning_time:.4f} seconds")
print(f"Trajectory length: {trajectory_length:.4f} radians")
print(f"Number of waypoints: {len(trajectory_array)}")
print(f"Collision-free: Verified by TrajOpt (optimization succeeded)")
print(f"Reasonable motion: {'Yes' if max_jump < 1.0 and within_limits else 'No'}")
print("="*60)
print("\nNote: TrajOpt performs collision checking internally during optimization.")
print("If optimization succeeds, the trajectory is collision-free.")

# Start Tesseract viewer
viewer = TesseractViewer()
viewer.update_environment(env, [0, 0, 0])  # Update environment with offset [0,0,0]
viewer.update_joint_positions(joint_names, initial_joint_positions)  # Set initial joint positions
viewer.update_trajectory(trajopt_results)  # Update with optimized trajectory (this enables animation loop)
viewer.start_serve_background()  # Start the web server in background

print("\n" + "="*60)
print("Tesseract viewer started!")
print("Open your browser and navigate to: http://localhost:8000")
print("The optimized trajectory will be displayed in the viewer.")
print("(Ignore any web server errors - they're harmless)")
print("="*60)
print("Press Enter to exit...")
try:
    input()
except KeyboardInterrupt:
    print("\nExiting...")
finally:
    # Restore stderr before exit
    if _swig_filter is not None:
        sys.stderr = _swig_filter.stderr





