import os
import sys
import csv
import numpy as np
import time
from pathlib import Path

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

# For start/goal and shortcut collision checks
from tesseract_robotics.tesseract_collision import (
    ContactResultMap, ContactResultVector, ContactRequest, ContactTestType_ALL
)

from helper_function import SWIGWarningFilter, add_target_marker


# ============================================================
# Configuration
# ============================================================
USE_OBSTACLES = True
_swig_filter = None

# Hyperparameter ranges (isolated experiments)
JITTER_SCALES = [0.0, 0.01, 0.02, 0.05, 0.1]
SAFETY_MARGINS = [0.005, 0.01, 0.02, 0.1]
MARGIN_BUFFERS = [0.005, 0.01, 0.02, 0.1]
COEFFS = [1.0, 10.0, 20.0, 50.0]

# Baselines for "other" parameters when varying one at a time
BASELINE_JITTER = 0.05
BASELINE_SAFETY_MARGIN = 0.01
BASELINE_MARGIN_BUFFER = 0.005
BASELINE_COEFF = 10.0

NUM_SEEDS = 10
SEEDS = [28 + i * 1000 for i in range(NUM_SEEDS)]  # fixed seeds for reproducibility

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
print("Trajectory Optimization with Hyperparameter Experiments")
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
# Helper: create random seed trajectory (parametric jitter)
# ============================================================
def create_random_seed_trajectory(seed, jitter_scale, num_waypoints=15, interp_density=4):
    """
    Create a random initial trajectory seed with given jitter scale.
    """
    np.random.seed(seed)

    waypoints = []
    for i in range(num_waypoints):
        alpha = i / (num_waypoints - 1) if num_waypoints > 1 else 0.0
        base = start_config + alpha * (goal_config - start_config)
        jitter = np.random.normal(scale=jitter_scale, size=base.shape)
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

    interpolated_program = generateInterpolatedProgram(
        program, env,
        3.14, 1.0, 3.14,
        interp_density
    )
    return interpolated_program

# ============================================================
# Helper: build TrajOpt planner with given collision parameters
# ============================================================
def build_trajopt_planner(safety_margin, margin_buffer, coeff):
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

    # Collision as soft cost with provided hyperparameters
    try:
        trajopt_composite_profile.collision_cost_config.enabled = True
        trajopt_composite_profile.collision_constraint_config.enabled = False

        if hasattr(trajopt_composite_profile.collision_cost_config, "safety_margin"):
            trajopt_composite_profile.collision_cost_config.safety_margin = safety_margin
        if hasattr(trajopt_composite_profile.collision_cost_config, "collision_margin_buffer"):
            trajopt_composite_profile.collision_cost_config.collision_margin_buffer = margin_buffer
        if hasattr(trajopt_composite_profile.collision_cost_config, "coeff"):
            trajopt_composite_profile.collision_cost_config.coeff = coeff

    except AttributeError:
        print("Note: Using default TrajOpt collision settings (could not access cost config attributes)")

    trajopt_profiles = ProfileDictionary()
    trajopt_profiles.addProfile(TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_plan_profile)
    trajopt_profiles.addProfile(TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_composite_profile)

    trajopt_planner = TrajOptMotionPlanner(TRAJOPT_DEFAULT_NAMESPACE)
    return trajopt_planner, trajopt_profiles

# ============================================================
# Stats helper
# ============================================================
def summarize_stats(arr):
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr))
    # Approximate mode: most frequent rounded-to-3-decimal value
    rounded = np.round(arr, 3)
    unique, counts = np.unique(rounded, return_counts=True)
    idx = int(np.argmax(counts))
    mode = float(unique[idx])
    mode_count = int(counts[idx])
    return mean, median, std, mode, mode_count

# ============================================================
# Core experiment runner
# ============================================================
def run_experiment(label, jitter_scale, safety_margin, margin_buffer, coeff, seeds):
    print("\n" + "#" * 70)
    print(f"EXPERIMENT: {label}")
    print(f"  jitter_scale={jitter_scale}, safety_margin={safety_margin}, "
          f"margin_buffer={margin_buffer}, coeff={coeff}")
    print("#" * 70)

    trajopt_planner, trajopt_profiles = build_trajopt_planner(
        safety_margin, margin_buffer, coeff
    )

    all_results = []

    for idx, seed in enumerate(seeds):
        print(f"\n  Seed {idx + 1}/{len(seeds)} (Seed {seed}):")
        seed_program = create_random_seed_trajectory(
            seed,
            jitter_scale=jitter_scale,
            num_waypoints=15,
            interp_density=4
        )
        seed_flat = seed_program.flatten()
        seed_waypoint_count = sum(1 for instr in seed_flat if instr.isMoveInstruction())
        print(f"    Seed waypoints (after interpolation): {seed_waypoint_count}")

        trajopt_request = PlannerRequest()
        trajopt_request.instructions = seed_program
        trajopt_request.env = env
        trajopt_request.profiles = trajopt_profiles

        start_time = time.time()
        try:
            trajopt_response = trajopt_planner.solve(trajopt_request)
            planning_time = time.time() - start_time

            if not trajopt_response.successful:
                all_results.append({
                    'seed_idx': idx,
                    'seed': seed,
                    'successful': False,
                    'planning_time': planning_time,
                    'message': trajopt_response.message
                })
                print(f"    ✗ Failed - {trajopt_response.message}")
                continue

            try:
                trajopt_results_instruction = trajopt_response.results
                trajopt_results = trajopt_results_instruction.flatten()

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
                opt_waypoint_count = len(trajectory_array)
                print(f"    Optimized waypoints: {opt_waypoint_count}")

                for i in range(len(trajectory_array) - 1):
                    diff = trajectory_array[i + 1] - trajectory_array[i]
                    trajectory_length += np.linalg.norm(diff)

                all_results.append({
                    'seed_idx': idx,
                    'seed': seed,
                    'successful': True,
                    'planning_time': planning_time,
                    'trajectory_length': trajectory_length,
                    'trajectory_instruction': trajopt_results_instruction
                })

                print(f"    ✓ Success - Length: {trajectory_length:.4f} rad, Time: {planning_time:.4f} s")
            except Exception as e:
                all_results.append({
                    'seed_idx': idx,
                    'seed': seed,
                    'successful': False,
                    'planning_time': planning_time,
                    'error': f"Error processing result: {str(e)}"
                })
                print(f"    ✗ Exception processing result - {e}")

        except Exception as e:
            all_results.append({
                'seed_idx': idx,
                'seed': seed,
                'successful': False,
                'error': str(e)
            })
            print(f"    ✗ Exception - {e}")

    successful_results = [r for r in all_results if r['successful']]

    base_info = {
        'label': label,
        'params': {
            'jitter_scale': jitter_scale,
            'safety_margin': safety_margin,
            'margin_buffer': margin_buffer,
            'coeff': coeff,
        },
        'raw_results': all_results,
    }

    if not successful_results:
        print("\n  >>> No successful trajectories in this experiment.")
        base_info.update({
            'successful': False,
        })
        return base_info

    planning_times = np.array([r['planning_time'] for r in successful_results], dtype=float)
    trajectory_lengths = np.array([r['trajectory_length'] for r in successful_results], dtype=float)

    pt_mean, pt_median, pt_std, pt_mode, pt_mode_count = summarize_stats(planning_times)
    len_mean, len_median, len_std, len_mode, len_mode_count = summarize_stats(trajectory_lengths)

    print("\n  == Experiment statistics ==")
    print(f"  Successful runs: {len(successful_results)}/{len(seeds)}")
    print("  Planning time (s):")
    print(f"    Mean   : {pt_mean:.4f}")
    print(f"    Median : {pt_median:.4f}")
    print(f"    Std    : {pt_std:.4f}")
    print(f"    Mode≈  : {pt_mode:.4f} (occurs {pt_mode_count} times, rounded to 3 decimals)")
    print("  Trajectory length (rad):")
    print(f"    Mean   : {len_mean:.4f}")
    print(f"    Median : {len_median:.4f}")
    print(f"    Std    : {len_std:.4f}")
    print(f"    Mode≈  : {len_mode:.4f} (occurs {len_mode_count} times, rounded to 3 decimals)")

    best_time_result = min(successful_results, key=lambda x: x['planning_time'])
    worst_time_result = max(successful_results, key=lambda x: x['planning_time'])
    best_len_result = min(successful_results, key=lambda x: x['trajectory_length'])
    worst_len_result = max(successful_results, key=lambda x: x['trajectory_length'])

    print("\n  Best trajectory by planning time:")
    print(f"    Seed idx: {best_time_result['seed_idx'] + 1}, Seed: {best_time_result['seed']}")
    print(f"    Planning time: {best_time_result['planning_time']:.4f} s")
    print(f"    Trajectory length: {best_time_result['trajectory_length']:.4f} rad")

    print("\n  Worst trajectory by planning time:")
    print(f"    Seed idx: {worst_time_result['seed_idx'] + 1}, Seed: {worst_time_result['seed']}")
    print(f"    Planning time: {worst_time_result['planning_time']:.4f} s")
    print(f"    Trajectory length: {worst_time_result['trajectory_length']:.4f} rad")

    print("\n  Best trajectory by trajectory length:")
    print(f"    Seed idx: {best_len_result['seed_idx'] + 1}, Seed: {best_len_result['seed']}")
    print(f"    Trajectory length: {best_len_result['trajectory_length']:.4f} rad")
    print(f"    Planning time: {best_len_result['planning_time']:.4f} s")

    print("\n  Worst trajectory by trajectory length:")
    print(f"    Seed idx: {worst_len_result['seed_idx'] + 1}, Seed: {worst_len_result['seed']}")
    print(f"    Trajectory length: {worst_len_result['trajectory_length']:.4f} rad")
    print(f"    Planning time: {worst_len_result['planning_time']:.4f} s")

    stats = {
        'pt_mean': pt_mean,
        'pt_median': pt_median,
        'pt_std': pt_std,
        'pt_mode': pt_mode,
        'pt_mode_count': pt_mode_count,
        'len_mean': len_mean,
        'len_median': len_median,
        'len_std': len_std,
        'len_mode': len_mode,
        'len_mode_count': len_mode_count,
    }

    base_info.update({
        'successful': True,
        'planning_times': planning_times,
        'trajectory_lengths': trajectory_lengths,
        'best_by_time': best_time_result,
        'best_by_length': best_len_result,
        'stats': stats,
    })

    return base_info

# ============================================================
# Run isolated experiments for each hyperparameter
# ============================================================
all_experiment_results = []

# Jitter scale experiments
for js in JITTER_SCALES:
    label = f"JITTER_SCALE={js}"
    res = run_experiment(
        label,
        jitter_scale=js,
        safety_margin=BASELINE_SAFETY_MARGIN,
        margin_buffer=BASELINE_MARGIN_BUFFER,
        coeff=BASELINE_COEFF,
        seeds=SEEDS
    )
    all_experiment_results.append(res)

# Safety margin experiments
for sm in SAFETY_MARGINS:
    label = f"SAFETY_MARGIN={sm}"
    res = run_experiment(
        label,
        jitter_scale=BASELINE_JITTER,
        safety_margin=sm,
        margin_buffer=BASELINE_MARGIN_BUFFER,
        coeff=BASELINE_COEFF,
        seeds=SEEDS
    )
    all_experiment_results.append(res)

# Collision margin buffer experiments
for mb in MARGIN_BUFFERS:
    label = f"MARGIN_BUFFER={mb}"
    res = run_experiment(
        label,
        jitter_scale=BASELINE_JITTER,
        safety_margin=BASELINE_SAFETY_MARGIN,
        margin_buffer=mb,
        coeff=BASELINE_COEFF,
        seeds=SEEDS
    )
    all_experiment_results.append(res)

# Coeff experiments
for c in COEFFS:
    label = f"COEFF={c}"
    res = run_experiment(
        label,
        jitter_scale=BASELINE_JITTER,
        safety_margin=BASELINE_SAFETY_MARGIN,
        margin_buffer=BASELINE_MARGIN_BUFFER,
        coeff=c,
        seeds=SEEDS
    )
    all_experiment_results.append(res)

# ============================================================
# Write CSVs with experiment results
# ============================================================
results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)

summary_csv_path = results_dir / "trajopt_experiments_summary.csv"
raw_csv_path = results_dir / "trajopt_experiments_raw.csv"

# Summary CSV: one row per experiment
with open(summary_csv_path, "w", newline="") as f_sum:
    writer = csv.writer(f_sum)
    writer.writerow([
        "experiment_label",
        "jitter_scale",
        "safety_margin",
        "margin_buffer",
        "coeff",
        "num_seeds",
        "num_success",
        "planning_time_mean",
        "planning_time_median",
        "planning_time_std",
        "planning_time_mode",
        "planning_time_mode_count",
        "traj_length_mean",
        "traj_length_median",
        "traj_length_std",
        "traj_length_mode",
        "traj_length_mode_count",
        "best_time_seed_idx",
        "best_time_seed",
        "best_time_planning_time",
        "best_time_traj_length",
        "best_len_seed_idx",
        "best_len_seed",
        "best_len_planning_time",
        "best_len_traj_length",
    ])

    for exp_res in all_experiment_results:
        if not exp_res:
            continue
        label = exp_res.get("label", "")
        params = exp_res.get("params", {})
        js = params.get("jitter_scale", "")
        sm = params.get("safety_margin", "")
        mb = params.get("margin_buffer", "")
        coeff = params.get("coeff", "")
        raw_results = exp_res.get("raw_results", [])
        num_seeds = len(raw_results)
        num_success = sum(1 for r in raw_results if r.get("successful"))

        if not exp_res.get("successful", False):
            writer.writerow([
                label, js, sm, mb, coeff,
                num_seeds, num_success,
                "", "", "", "", "",
                "", "", "", "", "",
                "", "", "", "",
                "", "", "", "",
            ])
            continue

        stats = exp_res["stats"]
        best_time = exp_res["best_by_time"]
        best_len = exp_res["best_by_length"]

        writer.writerow([
            label,
            js,
            sm,
            mb,
            coeff,
            num_seeds,
            num_success,
            stats["pt_mean"],
            stats["pt_median"],
            stats["pt_std"],
            stats["pt_mode"],
            stats["pt_mode_count"],
            stats["len_mean"],
            stats["len_median"],
            stats["len_std"],
            stats["len_mode"],
            stats["len_mode_count"],
            best_time["seed_idx"],
            best_time["seed"],
            best_time["planning_time"],
            best_time["trajectory_length"],
            best_len["seed_idx"],
            best_len["seed"],
            best_len["planning_time"],
            best_len["trajectory_length"],
        ])

# Raw CSV: one row per seed run
with open(raw_csv_path, "w", newline="") as f_raw:
    writer = csv.writer(f_raw)
    writer.writerow([
        "experiment_label",
        "jitter_scale",
        "safety_margin",
        "margin_buffer",
        "coeff",
        "seed_idx",
        "seed",
        "successful",
        "planning_time",
        "trajectory_length",
        "message",
        "error",
    ])

    for exp_res in all_experiment_results:
        if not exp_res:
            continue
        label = exp_res.get("label", "")
        params = exp_res.get("params", {})
        js = params.get("jitter_scale", "")
        sm = params.get("safety_margin", "")
        mb = params.get("margin_buffer", "")
        coeff = params.get("coeff", "")
        for r in exp_res.get("raw_results", []):
            writer.writerow([
                label,
                js,
                sm,
                mb,
                coeff,
                r.get("seed_idx", ""),
                r.get("seed", ""),
                r.get("successful", False),
                r.get("planning_time", ""),
                r.get("trajectory_length", ""),
                r.get("message", ""),
                r.get("error", ""),
            ])

print(f"\nWrote summary CSV to: {summary_csv_path}")
print(f"Wrote raw CSV to    : {raw_csv_path}")

# ============================================================
# Pick overall best-by-planning-time trajectory across ALL experiments
# ============================================================
overall_best = None
overall_best_exp = None

for exp_res in all_experiment_results:
    if not exp_res or not exp_res.get('successful', False):
        continue
    candidate = exp_res['best_by_time']
    if (overall_best is None) or (candidate['planning_time'] < overall_best['planning_time']):
        overall_best = candidate
        overall_best_exp = exp_res

if overall_best is None:
    print("\nNo successful trajectories in any experiment. Exiting.")
    if _swig_filter is not None:
        sys.stderr = _swig_filter.stderr
    sys.exit(1)

print("\n" + "="*60)
print("OVERALL BEST TRAJECTORY (BY PLANNING TIME) ACROSS ALL EXPERIMENTS")
print("="*60)
print(f"Experiment label : {overall_best_exp['label']}")
print(f"Seed idx         : {overall_best['seed_idx'] + 1}")
print(f"Seed             : {overall_best['seed']}")
print(f"Planning time (s): {overall_best['planning_time']:.4f}")
print(f"Traj length (rad): {overall_best['trajectory_length']:.4f}")
print("="*60)

# ============================================================
# Use overall best trajectory for time parameterization + viewer
# ============================================================
trajopt_results_instruction = overall_best['trajectory_instruction']
planning_time = overall_best['planning_time']
trajectory_length = overall_best['trajectory_length']

trajopt_results = trajopt_results_instruction.flatten()

# Extract joint waypoints (pre-time-param)
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

# Time parameterization
print("\nAdding time parameterization for overall-best trajectory...")
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

# Post-processing: shortcut + duplicate filter
def is_segment_collision_free(q_start, q_end, num_checks=10):
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
            traj = np.vstack([traj[:i+1], traj[j:]])
            n = len(traj)

    return traj

def filter_duplicate_waypoints(traj_array, tol=5e-3):
    if len(traj_array) == 0:
        return traj_array
    filtered = [traj_array[0]]
    for q in traj_array[1:]:
        if np.linalg.norm(q - filtered[-1]) > tol:
            filtered.append(q)
    return np.array(filtered)

trajectory_array_time_param = shortcut_trajectory(trajectory_array_time_param, num_iters=300, max_span=8)
trajectory_array_time_param = filter_duplicate_waypoints(trajectory_array_time_param, tol=5e-3)

# Verify trajectory reasonableness
print("\nVerifying overall-best trajectory reasonableness...")
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
print("OVERALL-BEST TRAJECTORY SUMMARY")
print("="*60)
print(f"Planning time: {planning_time:.4f} seconds")
print(f"Trajectory length: {trajectory_length:.4f} radians")
print(f"Number of waypoints (after filtering): {len(trajectory_array_time_param)}")
print(f"Maximum joint space jump: {max_jump:.4f} radians")
print(f"Within joint limits: {'Yes' if within_limits else 'No'}")
print(f"Reasonable motion: {'Yes' if (max_jump < 1.0 and within_limits) else 'No'}")
print("="*60)

# Viewer
print("\nStarting Tesseract viewer for overall-best trajectory...")
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
    print("The optimized overall-best trajectory will be displayed in the viewer.")
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
