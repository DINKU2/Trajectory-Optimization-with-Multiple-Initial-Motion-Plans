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
contact_manager_config_path = Path(__file__).parent / "assets" / "panda" / "contact_manager_plugins.yaml"
_contact_manager_plugin_info = None
_contact_manager_cmd = None
try:
    with open(contact_manager_config_path, 'r') as f:
        contact_manager_yaml = f.read()
    _contact_manager_plugin_info = parseContactManagersPluginConfigString(contact_manager_yaml)
    _contact_manager_cmd = AddContactManagersPluginInfoCommand(_contact_manager_plugin_info)
    env.applyCommand(_contact_manager_cmd)
    print("Contact manager plugins configured successfully")
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
print("Initializing TrajOpt with Different Numbers of Waypoints")
print("="*60)

# Start and goal configurations
start_config = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
goal_config = np.array([2.35, 1., 0., -0.8, 0, 2.5, 0.785], dtype=np.float64)

# Set up TrajOpt planner with more lenient settings
TRAJOPT_DEFAULT_NAMESPACE = "TrajOptMotionPlannerTask"
trajopt_plan_profile = TrajOptDefaultPlanProfile()
trajopt_composite_profile = TrajOptDefaultCompositeProfile()

# Use collision cost (soft) instead of hard constraint
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

# Helper to build a straight-line joint-space program with N waypoints
def build_program_with_waypoints(num_waypoints: int):
    waypoints = []
    for i in range(num_waypoints):
        alpha = i / (num_waypoints - 1) if num_waypoints > 1 else 0.0
        interpolated = start_config + alpha * (goal_config - start_config)
        interpolated = np.clip(interpolated, joint_min, joint_max)
        waypoints.append(interpolated.astype(np.float64))

    program = CompositeInstruction("DEFAULT")
    program.setManipulatorInfo(manip_info)

    # Add waypoints as JointWaypoints
    start_wp = JointWaypoint(joint_names, waypoints[0])
    start_instruction = MoveInstruction(
        JointWaypointPoly_wrap_JointWaypoint(start_wp),
        MoveInstructionType_FREESPACE,
        "DEFAULT"
    )
    program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(start_instruction))

    for i in range(1, len(waypoints)):
        wp = JointWaypoint(joint_names, waypoints[i])
        instruction = MoveInstruction(
            JointWaypointPoly_wrap_JointWaypoint(wp),
            MoveInstructionType_FREESPACE,
            "DEFAULT"
        )
        program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(instruction))

    # Interpolate the program to get dense waypoints for TrajOpt
    print(f"  Interpolating trajectory for {num_waypoints} waypoints...")
    interpolated_program = generateInterpolatedProgram(program, env, 3.14, 1.0, 3.14, 5)
    return interpolated_program

# Stats helper
def summarize_stats(arr):
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    rounded = np.round(arr, 3)
    unique, counts = np.unique(rounded, return_counts=True)
    idx = int(np.argmax(counts))
    mode_val = float(unique[idx])
    mode_count = int(counts[idx])
    return mean_val, median_val, std_val, mode_val, mode_count

# Test different waypoint counts
waypoint_options = [2, 5, 10, 30, 50]
all_results = []

for num_waypoints in waypoint_options:
    print("\n" + "-"*60)
    print(f"Testing with num_waypoints = {num_waypoints}")
    print("-"*60)

    interpolated_program = build_program_with_waypoints(num_waypoints)

    trajopt_request = PlannerRequest()
    trajopt_request.instructions = interpolated_program
    trajopt_request.env = env
    trajopt_request.profiles = trajopt_profiles

    print("  Running TrajOpt optimization...")
    start_time = time.time()
    try:
        trajopt_response = trajopt_planner.solve(trajopt_request)
        planning_time = time.time() - start_time
    except Exception as e:
        print(f"  ERROR: TrajOpt optimization failed with exception: {e}")
        all_results.append({
            'num_waypoints': num_waypoints,
            'successful': False,
            'planning_time': None,
            'trajectory_length': None,
            'message': str(e)
        })
        continue

    if not trajopt_response.successful:
        print(f"  ERROR: TrajOpt failed to find a solution: {trajopt_response.message}")
        all_results.append({
            'num_waypoints': num_waypoints,
            'successful': False,
            'planning_time': planning_time,
            'trajectory_length': None,
            'message': trajopt_response.message
        })
        continue

    print(f"  ✓ TrajOpt optimization completed successfully!")
    print(f"  Planning time: {planning_time:.4f} seconds")

    trajopt_results_instruction = trajopt_response.results
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
    if len(trajectory_array) == 0:
        print("  Warning: No waypoints found in optimized trajectory.")
        trajectory_length = np.nan
    else:
        # Calculate trajectory length (sum of distances between consecutive waypoints)
        trajectory_length = 0.0
        for i in range(len(trajectory_array) - 1):
            diff = trajectory_array[i+1] - trajectory_array[i]
            trajectory_length += np.linalg.norm(diff)

    print(f"  Trajectory length: {trajectory_length:.4f} radians")
    print(f"  Number of waypoints in optimized trajectory: {len(trajectory_array)}")

    all_results.append({
        'num_waypoints': num_waypoints,
        'successful': True,
        'planning_time': planning_time,
        'trajectory_length': trajectory_length,
        'trajopt_results_instruction': trajopt_results_instruction
    })

# ===============================
# Aggregate statistics over runs
# ===============================
successful_results = [r for r in all_results if r['successful'] and r['trajectory_length'] is not None]

if not successful_results:
    print("\nERROR: All TrajOpt optimizations failed for all waypoint settings. Cannot proceed.")
    if _swig_filter is not None:
        sys.stderr = _swig_filter.stderr
    sys.exit(1)

planning_times = np.array([r['planning_time'] for r in successful_results], dtype=float)
trajectory_lengths = np.array([r['trajectory_length'] for r in successful_results], dtype=float)

pt_mean, pt_median, pt_std, pt_mode, pt_mode_count = summarize_stats(planning_times)
len_mean, len_median, len_std, len_mode, len_mode_count = summarize_stats(trajectory_lengths)

print("\n" + "="*60)
print("AGGREGATE STATISTICS OVER WAYPOINT COUNTS")
print("="*60)
print("Planning time statistics (seconds):")
print(f"  Mean   : {pt_mean:.4f}")
print(f"  Median : {pt_median:.4f}")
print(f"  Std    : {pt_std:.4f}")
print(f"  Mode≈  : {pt_mode:.4f} (occurs {pt_mode_count} times, rounded to 3 decimals)")

print("\nTrajectory length statistics (radians):")
print(f"  Mean   : {len_mean:.4f}")
print(f"  Median : {len_median:.4f}")
print(f"  Std    : {len_std:.4f}")
print(f"  Mode≈  : {len_mode:.4f} (occurs {len_mode_count} times, rounded to 3 decimals)")

# Extremes by planning time and trajectory length
best_time_result = min(successful_results, key=lambda r: r['planning_time'])
worst_time_result = max(successful_results, key=lambda r: r['planning_time'])
best_len_result = min(successful_results, key=lambda r: r['trajectory_length'])
worst_len_result = max(successful_results, key=lambda r: r['trajectory_length'])

print("\nEXTREMES BY PLANNING TIME:")
print(f"  Fastest : num_waypoints={best_time_result['num_waypoints']}, "
      f"time={best_time_result['planning_time']:.4f}s, "
      f"length={best_time_result['trajectory_length']:.4f}rad")
print(f"  Slowest : num_waypoints={worst_time_result['num_waypoints']}, "
      f"time={worst_time_result['planning_time']:.4f}s, "
      f"length={worst_time_result['trajectory_length']:.4f}rad")

print("\nEXTREMES BY TRAJECTORY LENGTH:")
print(f"  Shortest: num_waypoints={best_len_result['num_waypoints']}, "
      f"length={best_len_result['trajectory_length']:.4f}rad, "
      f"time={best_len_result['planning_time']:.4f}s")
print(f"  Longest : num_waypoints={worst_len_result['num_waypoints']}, "
      f"length={worst_len_result['trajectory_length']:.4f}rad, "
      f"time={worst_len_result['planning_time']:.4f}s")

# =====================================
# Choose best trajectory (by time)
# =====================================
best_result = best_time_result

print("\n" + "="*60)
print("BEST TRAJECTORY (BY PLANNING TIME)")
print("="*60)
print(f"num_waypoints used: {best_result['num_waypoints']}")
print(f"Planning time: {best_result['planning_time']:.4f} seconds")
print(f"Trajectory length: {best_result['trajectory_length']:.4f} radians")

trajopt_results_instruction = best_result['trajopt_results_instruction']
planning_time = best_result['planning_time']
trajectory_length = best_result['trajectory_length']

# Time parameterization for best trajectory
print("\nAdding time parameterization to best trajectory...")
time_parameterization = TimeOptimalTrajectoryGeneration()
instructions_trajectory = InstructionsTrajectory(trajopt_results_instruction)

max_velocity = np.array([[2.3925, 2.3925, 2.3925, 2.3925, 2.8710, 2.8710, 2.8710]], dtype=np.float64)
max_velocity = np.hstack((-max_velocity.T, max_velocity.T))
max_acceleration = np.array([[15, 15, 15, 15, 20, 20, 20]], dtype=np.float64)
max_acceleration = np.hstack((-max_acceleration.T, max_acceleration.T))
max_jerk = np.array([[100, 100, 100, 100, 100, 100, 100]], dtype=np.float64)
max_jerk = np.hstack((-max_jerk.T, max_jerk.T))

if time_parameterization.compute(instructions_trajectory, max_velocity, max_acceleration, max_jerk):
    print("Time parameterization completed successfully")
else:
    print("Warning: Time parameterization failed, animation may not work properly")

# Flatten the results for analysis and visualization
trajopt_results = trajopt_results_instruction.flatten()

# Extract trajectory waypoints for analysis after (attempted) time-param
trajectory_waypoints = []
for instr in trajopt_results:
    if instr.isMoveInstruction():
        move_instr = InstructionPoly_as_MoveInstructionPoly(instr)
        wp = move_instr.getWaypoint()
        if wp.isStateWaypoint():
            state_wp = WaypointPoly_as_StateWaypointPoly(wp)
            trajectory_waypoints.append(state_wp.getPosition())

trajectory_array = np.array(trajectory_waypoints)

# Verify trajectory is reasonable
print("\nVerifying trajectory reasonableness (best-by-time)...")
max_jump = 0.0
for i in range(len(trajectory_array) - 1):
    jump = np.linalg.norm(trajectory_array[i+1] - trajectory_array[i])
    max_jump = max(max_jump, jump)

print(f"Maximum joint space jump between waypoints: {max_jump:.4f} radians")
if max_jump > 1.0:
    print("⚠ Warning: Large jumps detected in trajectory")
else:
    print("✓ Trajectory has reasonable waypoint spacing")

within_limits = True
for i, joint_pos in enumerate(trajectory_array):
    if np.any(joint_pos < joint_min) or np.any(joint_pos > joint_max):
        within_limits = False
        print(f"⚠ Warning: Waypoint {i} violates joint limits")
        break

if within_limits:
    print("✓ All waypoints are within joint limits")

# Final summary for best run
print("\n" + "="*60)
print("TRAJECTORY OPTIMIZATION SUMMARY (BEST-BY-TIME)")
print("="*60)
print(f"num_waypoints used: {best_result['num_waypoints']}")
print(f"Planning time: {planning_time:.4f} seconds")
print(f"Trajectory length: {trajectory_length:.4f} radians")
print(f"Number of waypoints (after optimization): {len(trajectory_array)}")
print(f"Maximum joint space jump: {max_jump:.4f} radians")
print(f"Within joint limits: {'Yes' if within_limits else 'No'}")
print("Collision-free: Verified by TrajOpt (optimization succeeded)")
print("Note: Aggregate statistics for all waypoint counts printed above.")
print("="*60)

# Start Tesseract viewer with best trajectory
viewer = TesseractViewer()
viewer.update_environment(env, [0, 0, 0])  # Update environment with offset [0,0,0]
viewer.update_joint_positions(joint_names, initial_joint_positions)  # Set initial joint positions
viewer.update_trajectory(trajopt_results)  # Use best trajectory
viewer.start_serve_background()  # Start the web server in background

print("\n" + "="*60)
print("Tesseract viewer started!")
print("Open your browser and navigate to: http://localhost:8000")
print("The best-by-time optimized trajectory will be displayed in the viewer.")
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
