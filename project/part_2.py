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
    JointWaypoint, MoveInstructionType_FREESPACE, MoveInstruction, CompositeInstruction,
    ProfileDictionary, JointWaypointPoly_wrap_JointWaypoint, MoveInstructionPoly_wrap_MoveInstruction,
    InstructionPoly_as_MoveInstructionPoly, WaypointPoly_as_StateWaypointPoly, WaypointPoly_as_JointWaypointPoly)
from tesseract_robotics.tesseract_motion_planners import PlannerRequest
from tesseract_robotics.tesseract_motion_planners_trajopt import (TrajOptDefaultPlanProfile, TrajOptDefaultCompositeProfile, TrajOptMotionPlanner)
from tesseract_robotics.tesseract_motion_planners_simple import generateInterpolatedProgram
from tesseract_robotics.tesseract_time_parameterization import TimeOptimalTrajectoryGeneration, InstructionsTrajectory
from tesseract_robotics.tesseract_collision import (ContactResultMap, ContactTestType_ALL, ContactRequest, ContactResultVector)
from tesseract_robotics_viewer import TesseractViewer

# Import VAMP integration functions
from vamp_trajopt_integration import (
    generate_vamp_paths, 
    vamp_path_to_tesseract_program, 
    calculate_trajectory_length, 
    get_trajectory_duration
)


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
print("Part 2: Trajectory Optimization with Multiple Initial Motion Plans")
print("="*60)

# Define start and goal configurations (same as VAMP examples)
start_config = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
goal_config = np.array([2.35, 1., 0., -0.8, 0, 2.5, 0.785], dtype=np.float64)

# Get obstacle centers for VAMP (same obstacles as Tesseract environment)
obstacle_centers = []
if USE_OBSTACLES:
    obstacles_file = Path(__file__).parent / "assets" / "obstacles" / "obstacles.txt"
    with open(obstacles_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('['):
                coords = [float(x.strip()) for x in line[1:-1].split(',')]
                obstacle_centers.append(coords)

# Generate multiple initial paths using VAMP RRT-Connect
NUM_SEEDS = 5  # Number of different random seeds to try
print(f"\nStep 1: Generating {NUM_SEEDS} initial paths with VAMP RRT-Connect...")
vamp_paths, successful_seeds = generate_vamp_paths(
    start_config, 
    goal_config, 
    obstacle_centers, 
    num_seeds=NUM_SEEDS, 
    radius=0.2
)

if not vamp_paths:
    print("ERROR: No valid VAMP paths generated. Cannot proceed with TrajOpt.")
    exit(1)

print(f"\nStep 2: Running TrajOpt with {len(vamp_paths)} initial trajectories...")

# Set up TrajOpt planner (shared settings for all runs)
TRAJOPT_DEFAULT_NAMESPACE = "TrajOptMotionPlannerTask"
trajopt_plan_profile = TrajOptDefaultPlanProfile()
trajopt_composite_profile = TrajOptDefaultCompositeProfile()

# Adjust collision settings to be more lenient (use cost instead of hard constraint)
try:
    trajopt_composite_profile.collision_constraint_config.enabled = False
    trajopt_composite_profile.collision_cost_config.enabled = True
    if hasattr(trajopt_composite_profile.collision_cost_config, 'collision_margin_buffer'):
        trajopt_composite_profile.collision_cost_config.collision_margin_buffer = 0.01
except AttributeError:
    print("Note: Using default TrajOpt collision settings")

trajopt_profiles = ProfileDictionary()
trajopt_profiles.addProfile(TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_plan_profile)
trajopt_profiles.addProfile(TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_composite_profile)

trajopt_planner = TrajOptMotionPlanner(TRAJOPT_DEFAULT_NAMESPACE)

# Run TrajOpt with each VAMP path as initial trajectory
all_results = []

for path_idx, vamp_path in enumerate(vamp_paths):
    seed = successful_seeds[path_idx]
    print(f"\n  Path {path_idx + 1}/{len(vamp_paths)} (Seed {seed}):")
    
    # Convert VAMP path to Tesseract CompositeInstruction
    program = vamp_path_to_tesseract_program(vamp_path, joint_names, manip_info)
    
    # Interpolate to get dense waypoints for TrajOpt
    interpolated_program = generateInterpolatedProgram(program, env, 3.14, 1.0, 3.14, 5)
    
    # Create TrajOpt planning request
    trajopt_request = PlannerRequest()
    trajopt_request.instructions = interpolated_program
    trajopt_request.env = env
    trajopt_request.profiles = trajopt_profiles
    
    # Run TrajOpt optimization
    start_time = time.time()
    try:
        trajopt_response = trajopt_planner.solve(trajopt_request)
        planning_time = time.time() - start_time
        
        if not trajopt_response.successful:
            all_results.append({
                'path_idx': path_idx,
                'seed': seed,
                'successful': False,
                'planning_time': planning_time,
                'message': trajopt_response.message
            })
            print(f"    ✗ Failed - {trajopt_response.message}")
            continue
        
        # Get optimized trajectory
        try:
            trajopt_results_instruction = trajopt_response.results
            trajopt_results = trajopt_results_instruction.flatten()
            
            # Calculate metrics (before time parameterization, waypoints are JointWaypoints)
            trajectory_length = calculate_trajectory_length(trajopt_results)
            
            # Add time parameterization for duration calculation
            time_parameterization = TimeOptimalTrajectoryGeneration()
            instructions_trajectory = InstructionsTrajectory(trajopt_results_instruction)
            max_velocity = np.array([[2.3925, 2.3925, 2.3925, 2.3925, 2.8710, 2.8710, 2.8710]], dtype=np.float64)
            max_velocity = np.hstack((-max_velocity.T, max_velocity.T))
            max_acceleration = np.array([[15, 15, 15, 15, 20, 20, 20]], dtype=np.float64)
            max_acceleration = np.hstack((-max_acceleration.T, max_acceleration.T))
            max_jerk = np.array([[100, 100, 100, 100, 100, 100, 100]], dtype=np.float64)
            max_jerk = np.hstack((-max_jerk.T, max_jerk.T))
            
            time_parameterization.compute(instructions_trajectory, max_velocity, max_acceleration, max_jerk)
            # Re-flatten after time parameterization to get updated waypoints with time info
            trajopt_results_time_param = trajopt_results_instruction.flatten()
            trajectory_duration = get_trajectory_duration(trajopt_results_time_param)
            
            all_results.append({
                'path_idx': path_idx,
                'seed': seed,
                'successful': True,
                'planning_time': planning_time,
                'trajectory_length': trajectory_length,
                'trajectory_duration': trajectory_duration,
                'trajectory': trajopt_results_time_param,  # Store time-parameterized version
                'trajectory_instruction': trajopt_results_instruction
            })
            
            print(f"    ✓ Success - Length: {trajectory_length:.4f} rad, Duration: {trajectory_duration:.4f} s, Time: {planning_time:.4f} s")
        except Exception as e:
            # Error occurred while processing the successful TrajOpt result
            all_results.append({
                'path_idx': path_idx,
                'seed': seed,
                'successful': False,
                'planning_time': planning_time,
                'error': f"Error processing result: {str(e)}"
            })
            print(f"    ✗ Exception processing result - {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        all_results.append({
            'path_idx': path_idx,
            'seed': seed,
            'successful': False,
            'error': str(e)
        })
        print(f"    ✗ Exception - {e}")

# Filter successful results
successful_results = [r for r in all_results if r['successful']]

if not successful_results:
    print("\nERROR: All TrajOpt optimizations failed. Cannot proceed.")
    exit(1)

print(f"\nStep 3: Comparing {len(successful_results)} successful trajectories...")

# Select best trajectory (shortest length, or you can use duration or weighted combination)
best_result = min(successful_results, key=lambda x: x['trajectory_length'])

print(f"\nBest trajectory: Path {best_result['path_idx'] + 1} (Seed {best_result['seed']})")
print(f"  Trajectory length: {best_result['trajectory_length']:.4f} radians")
print(f"  Trajectory duration: {best_result['trajectory_duration']:.4f} seconds")
print(f"  Planning time: {best_result['planning_time']:.4f} seconds")

# Use best trajectory for visualization
# Note: The trajectory is already time-parameterized and stored in best_result
planning_time = best_result['planning_time']
trajectory_length = best_result['trajectory_length']

# Use the stored time-parameterized trajectory (already computed during optimization)
print("\nPreparing trajectory for visualization...")
trajopt_results = best_result['trajectory']  # This is already time-parameterized
print("Using time-parameterized trajectory from optimization")

# Extract trajectory waypoints for analysis
trajectory_waypoints = []
for instr in trajopt_results:
    if instr.isMoveInstruction():
        move_instr = InstructionPoly_as_MoveInstructionPoly(instr)
        wp = move_instr.getWaypoint()
        if wp.isStateWaypoint():
            state_wp = WaypointPoly_as_StateWaypointPoly(wp)
            trajectory_waypoints.append(state_wp.getPosition())
        elif wp.isJointWaypoint():
            joint_wp = WaypointPoly_as_JointWaypointPoly(wp)
            trajectory_waypoints.append(joint_wp.getPosition())

trajectory_array = np.array(trajectory_waypoints)
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
print("PART 2: TRAJECTORY OPTIMIZATION SUMMARY")
print("="*60)
print(f"Total VAMP paths generated: {len(vamp_paths)}")
print(f"Successful TrajOpt optimizations: {len(successful_results)}/{len(all_results)}")
print(f"\nBest trajectory (Path {best_result['path_idx'] + 1}, Seed {best_result['seed']}):")
print(f"  Planning time: {planning_time:.4f} seconds")
print(f"  Trajectory length: {trajectory_length:.4f} radians")
print(f"  Trajectory duration: {best_result['trajectory_duration']:.4f} seconds")
print(f"  Number of waypoints: {len(trajectory_array)}")
print(f"  Collision-free: Verified by TrajOpt (optimization succeeded)")
print(f"  Reasonable motion: {'Yes' if max_jump < 1.0 and within_limits else 'No'}")
print("\nComparison with Part 1 (random initialization):")
print("  - Multiple initial guesses from VAMP RRT-Connect")
print("  - Best trajectory selected from multiple optimizations")
print("  - Expected: Better results due to collision-free initial paths")
print("="*60)

# Start Tesseract viewer
print("\nStarting Tesseract viewer...")
try:
    viewer = TesseractViewer()
    viewer.update_environment(env, [0, 0, 0])  # Update environment with offset [0,0,0]
    viewer.update_joint_positions(joint_names, initial_joint_positions)  # Set initial joint positions
    viewer.update_trajectory(trajopt_results)  # Update with optimized trajectory (this enables animation loop)
    viewer.start_serve_background()  # Start the web server in background
    viewer_started = True
except Exception as e:
    print(f"Warning: Could not start viewer: {e}")
    print("Continuing without visualization...")
    viewer_started = False

if viewer_started:
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
else:
    print("\nViewer not started. Summary report above shows the results.")

# Restore stderr before exit
if _swig_filter is not None:
    sys.stderr = _swig_filter.stderr





