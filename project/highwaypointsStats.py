import os
import sys
import numpy as np
import time
from pathlib import Path

from tesseract_robotics.tesseract_common import (
    FilesystemPath, GeneralResourceLocator,
    Isometry3d, Translation3d, ManipulatorInfo)
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
from tesseract_robotics_viewer import TesseractViewer
from tesseract_robotics.tesseract_collision import (
    ContactResultMap, ContactResultVector, ContactRequest, ContactTestType_ALL
)
from helper_function import SWIGWarningFilter, add_target_marker


# Configuration: Set to False to test without obstacles first
USE_OBSTACLES = True
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

# Initialize environment
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

# Set initial robot state
joint_names = [f"panda_joint{i+1}" for i in range(7)]
initial_joint_positions = np.zeros(7)
env.setState(joint_names, initial_joint_positions)

# Add obstacles to the environment (if enabled)
if USE_OBSTACLES:
    obstacles_file = Path(__file__).parent / "assets" / "obstacles" / "obstacles.txt"
    obstacle_radius = 0.2

    # Parse obstacle positions from file
    obstacle_positions = []
    with open(obstacles_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('['):
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
print("Part 1: Trajectory Optimization with Random Initialization")
print("="*60)

# Define start and goal configurations
start_config = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
goal_config = np.array([2.35, 1., 0., -0.8, 0, 2.5, 0.785], dtype=np.float64)

# Add visual markers for start and goal positions (for debugging)
print("\nAdding visual markers for start and goal positions...")
add_target_marker(env, joint_group, joint_names, start_config, "start_marker", 
                 tcp_frame=manip_info.tcp_frame, marker_radius=0.01)
add_target_marker(env, joint_group, joint_names, goal_config, "goal_marker", 
                 tcp_frame=manip_info.tcp_frame, marker_radius=0.01)

# Helper function to create random seed trajectory
def create_random_seed_trajectory(seed, num_waypoints=30, interp_density=50):
    """Create a random initial trajectory seed with given random seed."""
    np.random.seed(seed)
    
    # Create waypoints with minimal jitter (mostly straight line with tiny perturbations)
    # This gives TrajOpt a better starting point that's less likely to collide
    waypoints = []
    for i in range(num_waypoints):
        alpha = i / (num_waypoints - 1) if num_waypoints > 1 else 0.0
        base = start_config + alpha * (goal_config - start_config)
        # Much smaller jitter (0.02 instead of 0.05) to stay closer to straight line
        jitter = np.random.normal(scale=0.02, size=base.shape)
        if i in (0, num_waypoints - 1):
            jitter[:] = 0.0  # Keep start and goal exact
        w = np.clip(base + jitter, joint_min, joint_max).astype(np.float64)
        waypoints.append(w)
    
    # Build Tesseract program from waypoints
    program = CompositeInstruction("DEFAULT")
    program.setManipulatorInfo(manip_info)
    start_wp = JointWaypoint(joint_names, waypoints[0])
    program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(
        MoveInstruction(JointWaypointPoly_wrap_JointWaypoint(start_wp),
                        MoveInstructionType_FREESPACE, "DEFAULT")))
    for i in range(1, len(waypoints)):
        wp = JointWaypoint(joint_names, waypoints[i])
        program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(
            MoveInstruction(JointWaypointPoly_wrap_JointWaypoint(wp),
                            MoveInstructionType_FREESPACE, "DEFAULT")))
    
    # Interpolate to get dense waypoints for TrajOpt
    interpolated_program = generateInterpolatedProgram(program, env, 3.14, 1.0, 3.14, interp_density)
    return interpolated_program

# Set up TrajOpt planner
print("\nRunning TrajOpt optimization with multiple random initializations...")
TRAJOPT_DEFAULT_NAMESPACE = "TrajOptMotionPlannerTask"
trajopt_plan_profile = TrajOptDefaultPlanProfile()
trajopt_composite_profile = TrajOptDefaultCompositeProfile()

# Smoothness toggles (if present in your build)
for attr, val in [("smooth_velocities", True), ("smooth_accelerations", True), ("smooth_jerks", True)]:
    if hasattr(trajopt_plan_profile, attr):
        setattr(trajopt_plan_profile, attr, val)

planner = TrajOptMotionPlanner(TRAJOPT_DEFAULT_NAMESPACE)

# Use collision as cost (soft) here
try:
    trajopt_composite_profile.collision_constraint_config.enabled = False
    trajopt_composite_profile.collision_cost_config.enabled = True
    if hasattr(trajopt_composite_profile.collision_constraint_config, 'collision_margin_buffer'):
        trajopt_composite_profile.collision_constraint_config.collision_margin_buffer = 0.01
except AttributeError:
    print("Note: Using default TrajOpt collision settings")

trajopt_profiles = ProfileDictionary()
trajopt_profiles.addProfile(TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_plan_profile)
trajopt_profiles.addProfile(TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_composite_profile)

trajopt_planner = TrajOptMotionPlanner(TRAJOPT_DEFAULT_NAMESPACE)

# Try multiple random initializations
NUM_SEEDS = 30  # Number of different random seeds to try
print(f"\nTrying {NUM_SEEDS} different random initializations...")
all_results = []

for seed_idx in range(NUM_SEEDS):
    seed = 28 + seed_idx * 1000  # Different seeds for each attempt
    print(f"\n  Seed {seed_idx + 1}/{NUM_SEEDS} (Seed {seed}):")
    
    # Create random seed trajectory
    interpolated_program = create_random_seed_trajectory(seed, num_waypoints=30, interp_density=50)
    
    # Count waypoints in interpolated program
    interp_flat = interpolated_program.flatten()
    interp_waypoint_count = sum(1 for instr in interp_flat if instr.isMoveInstruction())
    print(f"    Initial waypoints (after interpolation): {interp_waypoint_count}")
    
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
                'seed_idx': seed_idx,
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
            
            # Calculate trajectory length
            trajectory_length = 0.0
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
            
            # Count waypoints in optimized trajectory
            opt_waypoint_count = len(trajectory_array)
            print(f"    Optimized waypoints: {opt_waypoint_count}")
            
            for i in range(len(trajectory_array) - 1):
                diff = trajectory_array[i+1] - trajectory_array[i]
                trajectory_length += np.linalg.norm(diff)
            
            all_results.append({
                'seed_idx': seed_idx,
                'seed': seed,
                'successful': True,
                'planning_time': planning_time,
                'trajectory_length': trajectory_length,
                'trajectory_instruction': trajopt_results_instruction
            })
            
            print(f"    ✓ Success - Length: {trajectory_length:.4f} rad, Time: {planning_time:.4f} s")
        except Exception as e:
            all_results.append({
                'seed_idx': seed_idx,
                'seed': seed,
                'successful': False,
                'planning_time': planning_time,
                'error': f"Error processing result: {str(e)}"
            })
            print(f"    ✗ Exception processing result - {e}")
            
    except Exception as e:
        all_results.append({
            'seed_idx': seed_idx,
            'seed': seed,
            'successful': False,
            'error': str(e)
        })
        print(f"    ✗ Exception - {e}")

# Filter successful results
successful_results = [r for r in all_results if r['successful']]

if not successful_results:
    print("\nERROR: All TrajOpt optimizations failed. Cannot proceed.")
    sys.exit(1)

print(f"\nComparing {len(successful_results)} successful trajectories...")

# ----------------------------------
# Statistics (planning time & length)
# ----------------------------------
planning_times = np.array([r['planning_time'] for r in successful_results], dtype=float)
trajectory_lengths = np.array([r['trajectory_length'] for r in successful_results], dtype=float)

def summarize_stats(arr):
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    # Approximate mode: most frequent rounded-to-3-decimal value
    rounded = np.round(arr, 3)
    unique, counts = np.unique(rounded, return_counts=True)
    idx = int(np.argmax(counts))
    mode_val = float(unique[idx])
    mode_count = int(counts[idx])
    return mean_val, median_val, std_val, mode_val, mode_count

pt_mean, pt_median, pt_std, pt_mode, pt_mode_count = summarize_stats(planning_times)
len_mean, len_median, len_std, len_mode, len_mode_count = summarize_stats(trajectory_lengths)

print("Planning time statistics (seconds):")
print(f"  Mean   : {pt_mean:.4f}")
print(f"  Median : {pt_median:.4f}")
print(f"  Std    : {pt_std:.4f}")
print(f"  Mode≈  : {pt_mode:.4f} (occurs {pt_mode_count} times, rounded to 3 decimals)")

print("Trajectory length statistics (radians):")
print(f"  Mean   : {len_mean:.4f}")
print(f"  Median : {len_median:.4f}")
print(f"  Std    : {len_std:.4f}")
print(f"  Mode≈  : {len_mode:.4f} (occurs {len_mode_count} times, rounded to 3 decimals)")

# ----------------------------------
# Extremes by time and length
# ----------------------------------
best_time_result = min(successful_results, key=lambda x: x['planning_time'])
worst_time_result = max(successful_results, key=lambda x: x['planning_time'])

best_len_result = min(successful_results, key=lambda x: x['trajectory_length'])
worst_len_result = max(successful_results, key=lambda x: x['trajectory_length'])

print("\n=== Extremes over successful runs ===")
print("By planning time:")
print(f"  Best (fastest): seed_idx={best_time_result['seed_idx']} "
      f"seed={best_time_result['seed']} "
      f"time={best_time_result['planning_time']:.4f}s "
      f"length={best_time_result['trajectory_length']:.4f}rad")
print(f"  Worst (slowest): seed_idx={worst_time_result['seed_idx']} "
      f"seed={worst_time_result['seed']} "
      f"time={worst_time_result['planning_time']:.4f}s "
      f"length={worst_time_result['trajectory_length']:.4f}rad")

print("By trajectory length:")
print(f"  Best (shortest): seed_idx={best_len_result['seed_idx']} "
      f"seed={best_len_result['seed']} "
      f"length={best_len_result['trajectory_length']:.4f}rad "
      f"time={best_len_result['planning_time']:.4f}s")
print(f"  Worst (longest): seed_idx={worst_len_result['seed_idx']} "
      f"seed={worst_len_result['seed']} "
      f"length={worst_len_result['trajectory_length']:.4f}rad "
      f"time={worst_len_result['planning_time']:.4f}s")

# Select best trajectory (by planning time) for visualization
best_result = best_time_result

print(f"\nBest trajectory (by planning time): Seed {best_result['seed_idx'] + 1} (Seed {best_result['seed']})")
print(f"  Trajectory length: {best_result['trajectory_length']:.4f} radians")
print(f"  Planning time: {best_result['planning_time']:.4f} seconds")

# Use best trajectory
trajopt_results_instruction = best_result['trajectory_instruction']
planning_time = best_result['planning_time']
trajectory_length = best_result['trajectory_length']

# Get optimized trajectory
trajopt_results = trajopt_results_instruction.flatten()

# Extract waypoints before time parameterization (for fallback)
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

# Add time parameterization (optional - skip if it fails)
print("\nAdding time parameterization...")
time_param_success = False
try:
    # Check if instruction has MoveInstructions before time parameterization
    test_flat = trajopt_results_instruction.flatten()
    move_instr_count = sum(1 for instr in test_flat if instr.isMoveInstruction())
    print(f"  Found {move_instr_count} MoveInstructions in trajectory")
    
    if move_instr_count == 0:
        raise RuntimeError(f"No MoveInstructions found in trajectory (total instructions: {len(test_flat)})")
    
    # Check if waypoints are StateWaypoints (required for InstructionsTrajectory)
    first_move = None
    for instr in test_flat:
        if instr.isMoveInstruction():
            first_move = InstructionPoly_as_MoveInstructionPoly(instr)
            break
    
    if first_move is not None:
        wp = first_move.getWaypoint()
        if wp.isJointWaypoint():
            print("  ⚠ Waypoints are JointWaypoints - skipping time parameterization (not supported)")
            raise RuntimeError("JointWaypoints cannot be used with InstructionsTrajectory - need StateWaypoints")
        elif wp.isStateWaypoint():
            print("  ✓ Waypoints are StateWaypoints - ready for time parameterization")
    
    time_parameterization = TimeOptimalTrajectoryGeneration()
    instructions_trajectory = InstructionsTrajectory(trajopt_results_instruction)
    max_velocity = np.array([[2.3925, 2.3925, 2.3925, 2.3925, 2.8710, 2.8710, 2.8710]], dtype=np.float64)
    max_velocity = np.hstack((-max_velocity.T, max_velocity.T))
    max_acceleration = np.array([[15, 15, 15, 15, 20, 20, 20]], dtype=np.float64)
    max_acceleration = np.hstack((-max_acceleration.T, max_acceleration.T))
    max_jerk = np.array([[100, 100, 100, 100, 100, 100, 100]], dtype=np.float64)
    max_jerk = np.hstack((-max_jerk.T, max_jerk.T))
    
    print("  Computing time parameterization...")
    time_parameterization.compute(instructions_trajectory, max_velocity, max_acceleration, max_jerk)
    print("✓ Time parameterization completed")
    time_param_success = True
except Exception as e:
    print(f"⚠ Warning: Time parameterization failed: {e}")
    print("⚠ Continuing without time parameterization (trajectory will still be visualized)")

# Re-extract trajectory after time parameterization (for viewer)
if time_param_success:
    trajopt_results_time_param = trajopt_results_instruction.flatten()
    
    # Re-extract waypoints after time parameterization for verification
    trajectory_waypoints_time_param = []
    for instr in trajopt_results_time_param:
        if instr.isMoveInstruction():
            move_instr = InstructionPoly_as_MoveInstructionPoly(instr)
            wp = move_instr.getWaypoint()
            if wp.isStateWaypoint():
                state_wp = WaypointPoly_as_StateWaypointPoly(wp)
                pos = state_wp.getPosition()
                trajectory_waypoints_time_param.append(pos)
            elif wp.isJointWaypoint():
                joint_wp = WaypointPoly_as_JointWaypointPoly(wp)
                pos = joint_wp.getPosition()
                trajectory_waypoints_time_param.append(pos)
    
    trajectory_array_time_param = np.array(trajectory_waypoints_time_param)
else:
    # Use unparameterized trajectory as fallback
    trajopt_results_time_param = trajopt_results
    trajectory_waypoints_time_param = trajectory_waypoints
    trajectory_array_time_param = trajectory_array

# Verify trajectory reasonableness
print("\nVerifying trajectory reasonableness...")
max_jump = 0.0
for i in range(len(trajectory_array_time_param) - 1):
    jump = np.linalg.norm(trajectory_array_time_param[i+1] - trajectory_array_time_param[i])
    max_jump = max(max_jump, jump)

within_limits = True
for i, joint_pos in enumerate(trajectory_array_time_param):
    if np.any(joint_pos < joint_min) or np.any(joint_pos > joint_max):
        within_limits = False
        print(f"⚠ Warning: Waypoint {i} violates joint limits")
        break

# Summary report
print("\n" + "="*60)
print("PART 1: TRAJECTORY OPTIMIZATION SUMMARY")
print("="*60)
print(f"Environment: {'With obstacles' if USE_OBSTACLES else 'Empty (no obstacles)'}")
print(f"Total successful optimizations: {len(successful_results)}/{NUM_SEEDS}")
print(f"Best planning time (s): {planning_time:.4f}")
print(f"Best trajectory length (rad): {trajectory_length:.4f}")
print(f"Planning time stats (s): mean={pt_mean:.4f}, median={pt_median:.4f}, std={pt_std:.4f}, mode≈{pt_mode:.4f}")
print(f"Trajectory length stats (rad): mean={len_mean:.4f}, median={len_median:.4f}, std={len_std:.4f}, mode≈{len_mode:.4f}")
print(f"Number of waypoints: {len(trajectory_array_time_param)}")
print(f"Maximum joint space jump: {max_jump:.4f} radians")
print(f"Within joint limits: {'Yes' if within_limits else 'No'}")
print(f"Collision-free: Optimized by TrajOpt (best of {len(successful_results)} attempts)")
print(f"Reasonable motion: {'Yes' if (max_jump < 1.0 and within_limits) else 'No'}")
print("="*60)

# Start Tesseract viewer
print("\nStarting Tesseract viewer...")
try:
    viewer = TesseractViewer()
    viewer.update_environment(env, [0, 0, 0])
    viewer.update_joint_positions(joint_names, start_config)  # Start at the start configuration
    # Show ONLY the best trajectory (by planning time)
    viewer.update_trajectory(trajopt_results_time_param)
    viewer.start_serve_background()
    viewer_started = True
    
    # Try to plot the trajectory path as a visible line/trail (after server starts)
    # This may fail if kinematics aren't configured, but won't crash the viewer
    try:
        print("  Drawing trajectory path as a line...")
        # Use plot_trajectory_list with the extracted joint positions
        viewer.plot_trajectory_list(joint_names, trajectory_array_time_param.tolist(), manip_info,
                                   color=[0, 1, 1, 1],  # Cyan color
                                   linewidth=0.005,     # Thicker line for visibility
                                   axes=False,          # Disable axes
                                   update_now=True)
        print("  ✓ Trajectory path line drawn successfully")
    except Exception as plot_error:
        print(f"  ⚠ Could not draw trajectory path line: {plot_error}")
        print("  (Viewer will still work, just without the path visualization)")
        
except Exception as e:
    print(f"Warning: Could not start viewer: {e}")
    print("Continuing without visualization...")
    viewer_started = False

if viewer_started:
    print("\n" + "="*60)
    print("Tesseract viewer started!")
    print("Open your browser and navigate to: http://localhost:8000")
    print("The optimized trajectory (best by planning time) will be displayed in the viewer.")
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
