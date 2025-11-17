import os
import sys
import numpy as np
import time
from pathlib import Path
from statistics import mean, median, multimode, stdev  # stats

from tesseract_robotics.tesseract_common import (
    FilesystemPath, GeneralResourceLocator,
    Isometry3d, Translation3d, ManipulatorInfo
)
from tesseract_robotics.tesseract_environment import (
    Environment, AddLinkCommand, AddContactManagersPluginInfoCommand
)
from tesseract_robotics.tesseract_srdf import parseContactManagersPluginConfigString
from tesseract_robotics.tesseract_scene_graph import Joint, Link, Visual, Collision, JointType_FIXED
from tesseract_robotics.tesseract_geometry import Sphere
from tesseract_robotics.tesseract_command_language import (
    JointWaypoint, MoveInstructionType_FREESPACE, MoveInstruction, CompositeInstruction,
    ProfileDictionary, JointWaypointPoly_wrap_JointWaypoint, MoveInstructionPoly_wrap_MoveInstruction,
    InstructionPoly_as_MoveInstructionPoly, WaypointPoly_as_StateWaypointPoly, WaypointPoly_as_JointWaypointPoly
)
from tesseract_robotics.tesseract_motion_planners import PlannerRequest
from tesseract_robotics.tesseract_motion_planners_trajopt import (
    TrajOptDefaultPlanProfile, TrajOptDefaultCompositeProfile, TrajOptMotionPlanner
)
from tesseract_robotics.tesseract_motion_planners_simple import generateInterpolatedProgram
from tesseract_robotics.tesseract_time_parameterization import TimeOptimalTrajectoryGeneration, InstructionsTrajectory
from tesseract_robotics_viewer import TesseractViewer

# For start/goal collision checks
from tesseract_robotics.tesseract_collision import (
    ContactResultMap, ContactResultVector, ContactRequest, ContactTestType_ALL
)

from helper_function import SWIGWarningFilter, add_target_marker


# ============================================================
# Configuration
# ============================================================
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

# ============================================================
# Initialize environment
# ============================================================
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

# ============================================================
# Add obstacles (spheres) if enabled
# ============================================================
if USE_OBSTACLES:
    obstacles_file = Path(__file__).parent / "assets" / "obstacles" / "obstacles.txt"
    obstacle_radius = 0.2

    obstacle_positions = []
    with open(obstacles_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('['):
                coords = [float(x.strip()) for x in line[1:-1].split(',')]
                obstacle_positions.append(coords)

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

# Get state solver
state_solver = env.getStateSolver()

# ============================================================
# Manipulator info and joint limits
# ============================================================
manip_info = ManipulatorInfo()
manip_info.tcp_frame = "panda_link8"
manip_info.manipulator = "panda_arm"
manip_info.working_frame = "panda_link0"

joint_group = env.getJointGroup("panda_arm")
joint_limits = joint_group.getLimits().joint_limits
joint_min = joint_limits[:, 0]
joint_max = joint_limits[:, 1]

print("\n" + "="*60)
print("Part 1: Trajectory Optimization with Random Initialization")
print("="*60)

# Define start and goal
start_config = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
goal_config = np.array([2.35, 1., 0., -0.8, 0, 2.5, 0.785], dtype=np.float64)

# Markers
print("\nAdding visual markers for start and goal positions...")
add_target_marker(
    env, joint_group, joint_names, start_config, "start_marker",
    tcp_frame=manip_info.tcp_frame, marker_radius=0.01
)
add_target_marker(
    env, joint_group, joint_names, goal_config, "goal_marker",
    tcp_frame=manip_info.tcp_frame, marker_radius=0.01
)

# ============================================================
# Helper: collision check for single configuration
# ============================================================
def check_single_config_for_collision(q, label):
    state_solver = env.getStateSolver()
    state_solver.setState(joint_names, q)
    scene_state = state_solver.getState()
    manager = env.getDiscreteContactManager()
    manager.setActiveCollisionObjects(env.getActiveLinkNames())
    manager.setCollisionObjectsTransform(scene_state.link_transforms)

    crm = ContactResultMap()
    manager.contactTest(crm, ContactRequest(ContactTestType_ALL))
    results = ContactResultVector()
    crm.flattenMoveResults(results)

    if len(results) > 0:
        print(f"[COLLISION] {label} is in collision with {len(results)} contact(s)")
    else:
        print(f"[OK] {label} is collision-free")

print("\nChecking start/goal for collisions...")
check_single_config_for_collision(start_config, "start_config")
check_single_config_for_collision(goal_config, "goal_config")

# ============================================================
# Helper: create random seed trajectory (reduced density)
# ============================================================
def create_random_seed_trajectory(seed, num_waypoints=15, interp_density=4):
    """
    Create a random initial trajectory seed with fewer waypoints and
    modest jitter to keep the optimization problem smaller and smoother.
    """
    np.random.seed(seed)

    waypoints = []
    for i in range(num_waypoints):
        alpha = i / (num_waypoints - 1) if num_waypoints > 1 else 0.0
        base = start_config + alpha * (goal_config - start_config)
        # Small jitter to help escape straight-line collisions
        jitter = np.random.normal(scale=0.05, size=base.shape)
        if i in (0, num_waypoints - 1):
            jitter[:] = 0.0  # exact start and goal
        w = np.clip(base + jitter, joint_min, joint_max).astype(np.float64)
        waypoints.append(w)

    program = CompositeInstruction("DEFAULT")
    program.setManipulatorInfo(manip_info)

    start_wp = JointWaypoint(joint_names, waypoints[0])
    program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(
        MoveInstruction(JointWaypointPoly_wrap_JointWaypoint(start_wp),
                        MoveInstructionType_FREESPACE, "DEFAULT")
    ))
    for i in range(1, len(waypoints)):
        wp = JointWaypoint(joint_names, waypoints[i])
        program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(
            MoveInstruction(JointWaypointPoly_wrap_JointWaypoint(wp),
                            MoveInstructionType_FREESPACE, "DEFAULT")
        ))

    # Lower interp_density -> fewer total timesteps
    interpolated_program = generateInterpolatedProgram(
        program, env,
        3.14, 1.0, 3.14,
        interp_density
    )
    return interpolated_program

# ============================================================
# TrajOpt setup with stronger smoothness + soft collision cost
# ============================================================
print("\nRunning TrajOpt optimization with multiple random initializations...")
TRAJOPT_DEFAULT_NAMESPACE = "TrajOptMotionPlannerTask"
trajopt_plan_profile = TrajOptDefaultPlanProfile()
trajopt_composite_profile = TrajOptDefaultCompositeProfile()

# Smoothness flags
for attr, val in [
    ("smooth_velocities", True),
    ("smooth_accelerations", True),
    ("smooth_jerks", True)
]:
    if hasattr(trajopt_plan_profile, attr):
        setattr(trajopt_plan_profile, attr, val)

# Stronger smoothness coefficients (if available)
for name, value in [
    ("velocity_coeff",     [10.0] * 7),
    ("acceleration_coeff", [5.0]  * 7),
    ("jerk_coeff",         [2.0]  * 7),
    ("velocity_coeffs",     [10.0] * 7),
    ("acceleration_coeffs", [5.0]  * 7),
    ("jerk_coeffs",         [2.0]  * 7),
]:
    if hasattr(trajopt_plan_profile, name):
        setattr(trajopt_plan_profile, name, value)

# Collision as SOFT COST for now (to get a nice solution)
try:
    trajopt_composite_profile.collision_cost_config.enabled = True
    trajopt_composite_profile.collision_constraint_config.enabled = False

    if hasattr(trajopt_composite_profile.collision_cost_config, "safety_margin"):
        trajopt_composite_profile.collision_cost_config.safety_margin = 0.01
    if hasattr(trajopt_composite_profile.collision_cost_config, "collision_margin_buffer"):
        trajopt_composite_profile.collision_cost_config.collision_margin_buffer = 0.005
    if hasattr(trajopt_composite_profile.collision_cost_config, "coeff"):
        trajopt_composite_profile.collision_cost_config.coeff = 10.0

except AttributeError:
    print("Note: Using default TrajOpt collision settings (could not access cost config attributes)")

trajopt_profiles = ProfileDictionary()
trajopt_profiles.addProfile(TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_plan_profile)
trajopt_profiles.addProfile(TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_composite_profile)

trajopt_planner = TrajOptMotionPlanner(TRAJOPT_DEFAULT_NAMESPACE)

# ============================================================
# Multi-seed loop (now 30 seeds + stats)
# ============================================================
NUM_SEEDS = 30  # Run 30 different random initializations
print(f"\nTrying {NUM_SEEDS} different random initializations...")
all_results = []

for seed_idx in range(NUM_SEEDS):
    seed = 28 + seed_idx * 1000
    print(f"\n  Seed {seed_idx + 1}/{NUM_SEEDS} (Seed {seed}):")

    seed_program = create_random_seed_trajectory(seed, num_waypoints=15, interp_density=4)
    seed_flat = seed_program.flatten()
    seed_waypoint_count = sum(1 for instr in seed_flat if instr.isMoveInstruction())
    print(f"    Seed waypoints (after interpolation): {seed_waypoint_count}")

    trajopt_request = PlannerRequest()
    # In this Python build, PlannerRequest has no .seed member,
    # so we pass the dense program directly as instructions.
    trajopt_request.instructions = seed_program
    trajopt_request.env = env
    trajopt_request.profiles = trajopt_profiles

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

        # Extract optimized trajectory
        try:
            trajopt_results_instruction_local = trajopt_response.results
            trajopt_results_local = trajopt_results_instruction_local.flatten()

            trajectory_length = 0.0
            trajectory_waypoints_local = []
            for instr in trajopt_results_local:
                if instr.isMoveInstruction():
                    move_instr = InstructionPoly_as_MoveInstructionPoly(instr)
                    wp = move_instr.getWaypoint()
                    if wp.isStateWaypoint():
                        state_wp = WaypointPoly_as_StateWaypointPoly(wp)
                        trajectory_waypoints_local.append(state_wp.getPosition())
                    elif wp.isJointWaypoint():
                        joint_wp = WaypointPoly_as_JointWaypointPoly(wp)
                        trajectory_waypoints_local.append(joint_wp.getPosition())

            trajectory_array_local = np.array(trajectory_waypoints_local)
            opt_waypoint_count = len(trajectory_array_local)
            print(f"    Optimized waypoints: {opt_waypoint_count}")

            for i in range(len(trajectory_array_local) - 1):
                diff = trajectory_array_local[i + 1] - trajectory_array_local[i]
                trajectory_length += np.linalg.norm(diff)

            all_results.append({
                'seed_idx': seed_idx,
                'seed': seed,
                'successful': True,
                'planning_time': planning_time,
                'trajectory_length': trajectory_length,
                'trajectory_instruction': trajopt_results_instruction_local
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

# ============================================================
# Choose best trajectory + statistics
# ============================================================
successful_results = [r for r in all_results if r['successful']]

if not successful_results:
    print("\nERROR: All TrajOpt optimizations failed. Cannot proceed.")
    if _swig_filter is not None:
        sys.stderr = _swig_filter.stderr
    sys.exit(1)

print(f"\nComparing {len(successful_results)} successful trajectories...")

# Compute statistics over planning_time and trajectory_length
planning_times = [r['planning_time'] for r in successful_results]
trajectory_lengths = [r['trajectory_length'] for r in successful_results]

def safe_mode(values):
    try:
        modes = multimode(values)
        return modes[0] if modes else None
    except Exception:
        return None

def safe_stdev(values):
    # stdev requires at least 2 data points
    return stdev(values) if len(values) > 1 else 0.0

pt_mean = mean(planning_times)
pt_median = median(planning_times)
pt_mode = safe_mode(planning_times)
pt_std = safe_stdev(planning_times)

len_mean = mean(trajectory_lengths)
len_median = median(trajectory_lengths)
len_mode = safe_mode(trajectory_lengths)
len_std = safe_stdev(trajectory_lengths)

print("\n=== Statistics over successful runs ===")
print(f"Planning time (s):")
print(f"  mean   = {pt_mean:.4f}")
print(f"  median = {pt_median:.4f}")
print(f"  std    = {pt_std:.4f}")
print(f"  mode   = {pt_mode:.4f}" if pt_mode is not None else "  mode   = N/A")

print(f"Trajectory length (rad):")
print(f"  mean   = {len_mean:.4f}")
print(f"  median = {len_median:.4f}")
print(f"  std    = {len_std:.4f}")
print(f"  mode   = {len_mode:.4f}" if len_mode is not None else "  mode   = N/A")

# Best/worst by planning time
best_time_result = min(successful_results, key=lambda x: x['planning_time'])
worst_time_result = max(successful_results, key=lambda x: x['planning_time'])

# Best/worst by trajectory length
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

# Select best trajectory by planning time for visualization
best_result = best_time_result

print(f"\nBest trajectory (by planning time): Seed {best_result['seed_idx'] + 1} (Seed {best_result['seed']})")
print(f"  Trajectory length: {best_result['trajectory_length']:.4f} radians")
print(f"  Planning time: {best_result['planning_time']:.4f} seconds")

trajopt_results_instruction = best_result['trajectory_instruction']
planning_time = best_result['planning_time']
trajectory_length = best_result['trajectory_length']

trajopt_results = trajopt_results_instruction.flatten()

# ============================================================
# Extract joint waypoints (pre-time-param)
# ============================================================
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

# ============================================================
# Time parameterization (if possible)
# ============================================================
print("\nAdding time parameterization...")
time_param_success = False
try:
    test_flat = trajopt_results_instruction.flatten()
    move_instr_count = sum(1 for instr in test_flat if instr.isMoveInstruction())
    print(f"  Found {move_instr_count} MoveInstructions in trajectory")

    if move_instr_count == 0:
        raise RuntimeError(f"No MoveInstructions found in trajectory (total instructions: {len(test_flat)})")

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

if time_param_success:
    trajopt_results_time_param = trajopt_results_instruction.flatten()

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
    trajopt_results_time_param = trajopt_results
    trajectory_waypoints_time_param = trajectory_waypoints
    trajectory_array_time_param = trajectory_array

# ============================================================
# Post-processing: shortcut + duplicate filter (trajectory_array_time_param)
# ============================================================
def is_segment_collision_free(q_start, q_end, num_checks=10):
    """
    Check straight-line in joint space between q_start and q_end for collisions.
    """
    ss = env.getStateSolver()
    manager = env.getDiscreteContactManager()
    manager.setActiveCollisionObjects(env.getActiveLinkNames())

    for k in range(num_checks + 1):
        alpha = k / float(num_checks)
        q = (1.0 - alpha) * q_start + alpha * q_end
        ss.setState(joint_names, q)
        scene_state = ss.getState()
        manager.setCollisionObjectsTransform(scene_state.link_transforms)

        crm = ContactResultMap()
        manager.contactTest(crm, ContactRequest(ContactTestType_ALL))
        results = ContactResultVector()
        crm.flattenMoveResults(results)
        if len(results) > 0:
            return False
    return True

def shortcut_trajectory(traj_array, num_iters=300, max_span=8):
    """
    Randomized shortcutting:
    try to replace sections of the trajectory with a straight joint-space segment
    if that segment is collision-free.
    """
    if len(traj_array) < 3:
        return traj_array.copy()

    traj = traj_array.copy()
    n = len(traj)

    for _ in range(num_iters):
        if n < 3:
            break

        i = np.random.randint(0, n - 2)
        j_min = i + 2
        j_max = min(i + max_span, n - 1)
        if j_min > j_max:
            continue
        j = np.random.randint(j_min, j_max + 1)

        q_i = traj[i]
        q_j = traj[j]

        if is_segment_collision_free(q_i, q_j, num_checks=10):
            # Replace middle segment with direct connection
            traj = np.vstack([traj[:i+1], traj[j:]])
            n = len(traj)

    return traj

def filter_duplicate_waypoints(traj_array, tol=5e-3):
    """
    Remove near-duplicate consecutive waypoints (tiny jitters).
    """
    if len(traj_array) == 0:
        return traj_array
    filtered = [traj_array[0]]
    for q in traj_array[1:]:
        if np.linalg.norm(q - filtered[-1]) > tol:
            filtered.append(q)
    return np.array(filtered)

# Apply shortcutting first, then deduplicate
trajectory_array_time_param = shortcut_trajectory(trajectory_array_time_param, num_iters=300, max_span=8)
trajectory_array_time_param = filter_duplicate_waypoints(trajectory_array_time_param, tol=5e-3)

# ============================================================
# Verify trajectory reasonableness
# ============================================================
print("\nVerifying trajectory reasonableness...")
max_jump = 0.0
for i in range(len(trajectory_array_time_param) - 1):
    jump = np.linalg.norm(trajectory_array_time_param[i + 1] - trajectory_array_time_param[i])
    max_jump = max(max_jump, jump)

within_limits = True
for i, joint_pos in enumerate(trajectory_array_time_param):
    if np.any(joint_pos < joint_min) or np.any(joint_pos > joint_max):
        within_limits = False
        print(f"⚠ Warning: Waypoint {i} violates joint limits")
        break

print("\n" + "="*60)
print("PART 1: TRAJECTORY OPTIMIZATION SUMMARY")
print("="*60)
print(f"Environment: {'With obstacles' if USE_OBSTACLES else 'Empty (no obstacles)'}")
print(f"Total successful optimizations: {len(successful_results)}/{NUM_SEEDS}")
print(f"Planning time (best): {planning_time:.4f} seconds")
print(f"Trajectory length (best): {trajectory_length:.4f} radians")
print(f"Number of waypoints (after filtering): {len(trajectory_array_time_param)}")
print(f"Maximum joint space jump: {max_jump:.4f} radians")
print(f"Within joint limits: {'Yes' if within_limits else 'No'}")
print(f"Collision-free: Encouraged by cost (soft) - verify with contact checks if needed")
print(f"Reasonable motion: {'Yes' if (max_jump < 1.0 and within_limits) else 'No'}")
print("="*60)

# ============================================================
# Viewer
# (unchanged, uses best-time solution)
# ============================================================
print("\nStarting Tesseract viewer...")
try:
    viewer = TesseractViewer()
    viewer.update_environment(env, [0, 0, 0])
    viewer.update_joint_positions(joint_names, start_config)
    viewer.update_trajectory(trajopt_results_time_param)
    viewer.start_serve_background()
    viewer_started = True

    try:
        print("  Drawing trajectory path as a line...")
        viewer.plot_trajectory_list(
            joint_names,
            trajectory_array_time_param.tolist(),
            manip_info,
            color=[0, 1, 1, 1],  # Cyan
            linewidth=0.005,
            axes=False,
            update_now=True
        )
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
