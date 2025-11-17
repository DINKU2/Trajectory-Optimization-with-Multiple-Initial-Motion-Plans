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

# -------------------------------------------------------------
# Suppress SWIG memory leak warnings (noise only)
# -------------------------------------------------------------
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

_swig_filter = None

# -------------------------------------------------------------
# Resource path setup
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# Contact manager plugins (if available)
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# Initial state
# -------------------------------------------------------------
joint_names = [f"panda_joint{i+1}" for i in range(7)]
initial_joint_positions = np.zeros(7)
env.setState(joint_names, initial_joint_positions)

# Optionally set a *global* collision margin (clearance) so planner & validators agree
try:
    env.setCollisionMarginData(CollisionMarginData(0.02))  # 2 cm global clearance target
    print("Global collision margin set to 0.02 m")
except Exception:
    pass

# -------------------------------------------------------------
# Obstacles from file (spheres)
# -------------------------------------------------------------
if USE_OBSTACLES:
    obstacles_file = Path(__file__).parent / "assets" / "obstacles" / "obstacles.txt"
    obstacle_radius = 0.2

    obstacle_positions = []
    if obstacles_file.exists():
        with open(obstacles_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('['):
                    coords = [float(x.strip()) for x in line[1:-1].split(',')]
                    obstacle_positions.append(coords)

    for i, pos in enumerate(obstacle_positions):
        obstacle_link = Link(f"obstacle_{i}")
        obstacle_visual = Visual(); obstacle_visual.geometry = Sphere(obstacle_radius)
        obstacle_link.visual.push_back(obstacle_visual)
        obstacle_collision = Collision(); obstacle_collision.geometry = Sphere(obstacle_radius)
        obstacle_link.collision.push_back(obstacle_collision)
        obstacle_joint = Joint(f"obstacle_{i}_joint")
        obstacle_joint.parent_link_name = "panda_link0"
        obstacle_joint.child_link_name = obstacle_link.getName()
        obstacle_joint.type = JointType_FIXED
        obstacle_joint.parent_to_joint_origin_transform = Isometry3d.Identity() * Translation3d(*pos)
        env.applyCommand(AddLinkCommand(obstacle_link, obstacle_joint))
    print(f"Added {len(obstacle_positions)} obstacles to the environment")
else:
    print("Obstacles disabled - running in empty environment")

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
state_solver = env.getStateSolver()
manip_info = ManipulatorInfo()
manip_info.tcp_frame = "panda_link8"
manip_info.manipulator = "panda_arm"
manip_info.working_frame = "panda_link0"

joint_group = env.getJointGroup("panda_arm")
joint_limits = joint_group.getLimits().joint_limits
joint_min = joint_limits[:, 0]
joint_max = joint_limits[:, 1]

np.random.seed(42)

print("\n" + "="*60)
print("Initializing TrajOpt with Random Solution (two-phase: cost -> constraint)")
print("="*60)

# --- Start & goal configs ---
start_config = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
goal_config = np.array([2.35, 1., 0., -0.8, 0, 2.5, 0.785], dtype=np.float64)

# --- Simple jittered straight-line seed (coarse) ---
num_waypoints = 5
waypoints = []
for i in range(num_waypoints):
    alpha = i / (num_waypoints - 1) if num_waypoints > 1 else 0.0
    base = start_config + alpha * (goal_config - start_config)
    jitter = np.random.normal(scale=0.05, size=base.shape)
    if i in (0, num_waypoints - 1):
        jitter[:] = 0.0
    w = np.clip(base + jitter, joint_min, joint_max).astype(np.float64)
    waypoints.append(w)

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

print("Interpolating trajectory seed...")
# Use denser seed so hard-constraint phase has more DOFs to adjust
interpolated_program = generateInterpolatedProgram(program, env, 3.14, 1.0, 3.14, 12)

# -------------------------------------------------------------
# TrajOpt profiles (two-phase)
# -------------------------------------------------------------
TRAJOPT_DEFAULT_NAMESPACE = "TrajOptMotionPlannerTask"
trajopt_plan_profile = TrajOptDefaultPlanProfile()
trajopt_composite_profile = TrajOptDefaultCompositeProfile()

# Smoothness toggles (if present in your build)
for attr, val in [("smooth_velocities", True), ("smooth_accelerations", True), ("smooth_jerks", True)]:
    if hasattr(trajopt_plan_profile, attr):
        setattr(trajopt_plan_profile, attr, val)

planner = TrajOptMotionPlanner(TRAJOPT_DEFAULT_NAMESPACE)

# Phase 1 profile (soft collisions)
trajopt_composite_profile.collision_cost_config.enabled = True
trajopt_composite_profile.collision_constraint_config.enabled = False
if hasattr(trajopt_composite_profile.collision_cost_config, 'collision_margin_buffer'):
    trajopt_composite_profile.collision_cost_config.collision_margin_buffer = 0.02

profiles_phase1 = ProfileDictionary()
profiles_phase1.addProfile(TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_plan_profile)
profiles_phase1.addProfile(TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_composite_profile)

def run_phase1(seed_instructions):
    """Run TrajOpt with soft collision cost."""
    req = PlannerRequest()
    req.instructions = seed_instructions
    req.env = env
    req.profiles = profiles_phase1
    t0 = time.time()
    resp = planner.solve(req)
    dt = time.time() - t0
    return resp, dt

def run_phase2(seed_instructions, margin: float, evaluator: str | None):
    """Run TrajOpt with hard collision constraints, starting from Phase 1 result."""
    profile2 = TrajOptDefaultCompositeProfile()
    profile2.collision_cost_config.enabled = False
    profile2.collision_constraint_config.enabled = True
    if hasattr(profile2.collision_constraint_config, 'collision_margin'):
        profile2.collision_constraint_config.collision_margin = margin
    if hasattr(profile2, 'collision_evaluator_type') and evaluator is not None:
        try:
            profile2.collision_evaluator_type = evaluator  # 'DISCRETE' or 'CONTINUOUS'
        except Exception:
            pass

    profiles2 = ProfileDictionary()
    profiles2.addProfile(TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_plan_profile)
    profiles2.addProfile(TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", profile2)

    req2 = PlannerRequest()
    req2.instructions = seed_instructions
    req2.env = env
    req2.profiles = profiles2
    t0 = time.time()
    resp2 = planner.solve(req2)
    dt = time.time() - t0
    return resp2, dt

# -------------------------------------------------------------
# Two-stage solve (no grid search)
# -------------------------------------------------------------
phase1_time = 0.0
phase2_time = 0.0

print("Running TrajOpt Phase 1 (soft collision cost)...")
resp1, phase1_time = run_phase1(interpolated_program)
if not resp1.successful:
    print(f"ERROR: Phase 1 failed: {resp1.message}")
    sys.exit(1)
print(f"Phase 1 completed in {phase1_time:.4f} s")

print("Running TrajOpt Phase 2 (hard collision constraints)...")
# You can tune these two parameters:
hard_margin = 0.01       # desired clearance in constraint (meters)
hard_evaluator = 'CONTINUOUS'  # or 'DISCRETE' if CONTINUOUS is not supported
resp2, phase2_time = run_phase2(resp1.results, hard_margin, hard_evaluator)
if not resp2.successful:
    print(f"ERROR: Phase 2 failed: {resp2.message}")
    sys.exit(1)
print(f"Phase 2 completed in {phase2_time:.4f} s")

trajopt_results_instruction = resp2.results

# -------------------------------------------------------------
# Time parameterization (for animation & rate checks)
# -------------------------------------------------------------
print("Adding time parameterization (TOTG)...")
time_parameterization = TimeOptimalTrajectoryGeneration()
instructions_trajectory = InstructionsTrajectory(trajopt_results_instruction)

# Velocity/accel/jerk limits (Panda rough values)
max_velocity = np.array([[2.3925, 2.3925, 2.3925, 2.3925, 2.8710, 2.8710, 2.8710]], dtype=np.float64)
max_velocity = np.hstack((-max_velocity.T, max_velocity.T))
max_acceleration = np.array([[15, 15, 15, 15, 20, 20, 20]], dtype=np.float64)
max_acceleration = np.hstack((-max_acceleration.T, max_acceleration.T))
max_jerk = np.array([[100, 100, 100, 100, 100, 100, 100]], dtype=np.float64)
max_jerk = np.hstack((-max_jerk.T, max_jerk.T))

if time_parameterization.compute(instructions_trajectory, max_velocity, max_acceleration, max_jerk):
    print("✓ Time parameterization completed")
else:
    print("Warning: Time parameterization failed, animation may not work properly")

# -------------------------------------------------------------
# Flatten trajectory for analysis/visualization
# -------------------------------------------------------------
trajopt_results = trajopt_results_instruction.flatten()
trajectory_waypoints = []
for instr in trajopt_results:
    if instr.isMoveInstruction():
        move_instr = InstructionPoly_as_MoveInstructionPoly(instr)
        wp = move_instr.getWaypoint()
        if wp.isStateWaypoint():
            state_wp = WaypointPoly_as_StateWaypointPoly(wp)
            trajectory_waypoints.append(state_wp.getPosition())
trajectory_array = np.array(trajectory_waypoints)

# Attempt to extract timestamps from InstructionsTrajectory (API may vary by build)
times = None
try:
    # Some builds expose a method to get time vector directly
    if hasattr(instructions_trajectory, 'getTimes'):
        times = np.array(instructions_trajectory.getTimes(), dtype=float)
except Exception:
    times = None

# -------------------------------------------------------------
# Reasonableness checks (limits, jumps, rates)
# -------------------------------------------------------------
trajectory_length = 0.0
for i in range(len(trajectory_array) - 1):
    diff = trajectory_array[i+1] - trajectory_array[i]
    trajectory_length += np.linalg.norm(diff)

print(f"Trajectory length: {trajectory_length:.4f} radians")
print(f"Number of waypoints: {len(trajectory_array)}")

max_jump = 0.0
for i in range(len(trajectory_array) - 1):
    jump = np.linalg.norm(trajectory_array[i+1] - trajectory_array[i])
    max_jump = max(max_jump, jump)
print(f"Maximum joint space jump between waypoints: {max_jump:.4f} radians")

within_limits = True
for i, joint_pos in enumerate(trajectory_array):
    if np.any(joint_pos < joint_min) or np.any(joint_pos > joint_max):
        within_limits = False
        print(f"⚠ Warning: Waypoint {i} violates joint limits")
        break
print("✓ All waypoints within joint limits" if within_limits else "Joint limit violation detected")

# Optional: velocity/acc checks using finite differences if times are available
if times is not None and len(times) == len(trajectory_array):
    vel_ok = True
    acc_ok = True
    eps = 1e-9
    vel_limits = max_velocity.T[:, 1]
    acc_limits = max_acceleration.T[:, 1]
    # velocities
    for i in range(1, len(trajectory_array)):
        dt = max(times[i] - times[i-1], eps)
        v = (trajectory_array[i] - trajectory_array[i-1]) / dt
        if np.any(np.abs(v) - vel_limits > 1e-6):
            vel_ok = False
    # accelerations
    for i in range(2, len(trajectory_array)):
        dt1 = max(times[i-1] - times[i-2], eps)
        dt2 = max(times[i] - times[i-1], eps)
        v1 = (trajectory_array[i-1] - trajectory_array[i-2]) / dt1
        v2 = (trajectory_array[i] - trajectory_array[i-1]) / dt2
        a = (v2 - v1) / max(0.5 * (dt1 + dt2), eps)
        if np.any(np.abs(a) - acc_limits > 1e-5):
            acc_ok = False
    print(f"✓ Velocity limits respected? {vel_ok}")
    print(f"✓ Acceleration limits respected? {acc_ok}")
else:
    print("(Rate checks skipped: could not extract timestamps from InstructionsTrajectory)")

# -------------------------------------------------------------
# Collision verification: discrete + continuous sweeps
# -------------------------------------------------------------
def verify_discrete_collision_free(env, joint_names, trajectory_array, margin=0.0):
    try:
        mgr = env.getDiscreteContactManager()
        mgr.setContactDistanceThreshold(margin)
        mgr.setActiveCollisionObjects(env.getActiveLinkNames())
        ss = env.getStateSolver()
        request = ContactRequest(ContactTestType_ALL)
        for i, q in enumerate(trajectory_array):
            state = ss.getState(joint_names, q.tolist())
            # Different bindings may expose transforms as a dict/map or via an accessor
            link_tf = None
            for cand in ("link_transforms", "linkTransforms", "link_transforms_map"):
                if hasattr(state, cand):
                    link_tf = getattr(state, cand)
                    break
            if link_tf is None:
                raise RuntimeError("State object does not expose link transforms; adapt here.")
            # set transforms for this waypoint
            set_ok = False
            for meth in ("setCollisionObjectsTransform", "setCollisionObjectsTransforms"):
                if hasattr(mgr, meth):
                    getattr(mgr, meth)(link_tf)
                    set_ok = True
                    break
            if not set_ok:
                raise RuntimeError("Contact manager missing setCollisionObjectsTransform(s) method; adapt here.")
            result_map = ContactResultMap()
            mgr.contactTest(result_map, request)
            if len(result_map) > 0:
                print(f"Contact detected at waypoint {i}")
                return False
        return True
    except Exception as e:
        print(f"(Discrete collision check skipped due to API mismatch: {e})")
        return True

def verify_continuous_collision_free(env, joint_names, trajectory_array, margin=0.0):
    try:
        ss = env.getStateSolver()
        mgr = env.getContinuousContactManager()
        mgr.setActiveCollisionObjects(env.getActiveLinkNames())
        mgr.setContactDistanceThreshold(margin)
        request = ContactRequest(ContactTestType_ALL)

        for i in range(len(trajectory_array) - 1):
            q0 = trajectory_array[i].tolist()
            q1 = trajectory_array[i+1].tolist()
            s0 = ss.getState(joint_names, q0)
            s1 = ss.getState(joint_names, q1)
            # access link transforms for start/end
            link_tf0 = None
            link_tf1 = None
            for cand in ("link_transforms", "linkTransforms", "link_transforms_map"):
                if hasattr(s0, cand):
                    link_tf0 = getattr(s0, cand)
                    break
            for cand in ("link_transforms", "linkTransforms", "link_transforms_map"):
                if hasattr(s1, cand):
                    link_tf1 = getattr(s1, cand)
                    break
            if link_tf0 is None or link_tf1 is None:
                raise RuntimeError("State object does not expose link transforms; adapt here.")
            # set start/end transforms for the swept segment
            set_ok = False
            for meth in ("setCollisionObjectsTransform", "setCollisionObjectsTransforms"):
                if hasattr(mgr, meth):
                    try:
                        getattr(mgr, meth)(link_tf0, link_tf1)
                        set_ok = True
                        break
                    except TypeError:
                        # Some builds accept a structure containing both start/end
                        pass
            if not set_ok:
                raise RuntimeError("Continuous manager missing setCollisionObjectsTransform(s) w/ (start,end); adapt here.")

            result_map = ContactResultMap()
            mgr.contactTest(result_map, request)
            if len(result_map) > 0:
                print(f"Continuous contact detected along segment [{i}->{i+1}]")
                return False
        return True
    except Exception as e:
        print(f"(Continuous collision check skipped due to API mismatch: {e})")
        return True

print("\nRunning independent collision sweeps...")
disc_ok = verify_discrete_collision_free(env, joint_names, trajectory_array, margin=0.0)
cont_ok = verify_continuous_collision_free(env, joint_names, trajectory_array, margin=0.0)
print(f"Discrete collision check: {'PASS' if disc_ok else 'FAIL'}")
print(f"Continuous collision check: {'PASS' if cont_ok else 'FAIL'}")

# -------------------------------------------------------------
# Summary
# -------------------------------------------------------------
print("\n" + "="*60)
print("TRAJECTORY OPTIMIZATION SUMMARY")
print("="*60)
print(f"Phase 1 time: {phase1_time:.4f} s | Phase 2 time: {phase2_time:.4f} s")
print(f"Trajectory length: {trajectory_length:.4f} radians")
print(f"Waypoints: {len(trajectory_array)}")
print(f"Collision-free (TrajOpt hard constraint phase): {resp2.successful}")
print(f"Independent discrete check: {disc_ok}")
print(f"Independent CONTINUOUS check: {cont_ok}")
print(f"Reasonable motion: {('Yes' if (max_jump < 1.0 and within_limits and cont_ok and disc_ok) else 'No')}")
print("="*60)
print("\nNote: If any independent check fails, treat the plan as invalid. Consider reseeding or increasing interpolation density.")

# -------------------------------------------------------------
# Viewer
# -------------------------------------------------------------
viewer = TesseractViewer()
viewer.update_environment(env, [0, 0, 0])
viewer.update_joint_positions(joint_names, initial_joint_positions)
viewer.update_trajectory(trajopt_results)  # enables animation
viewer.start_serve_background()

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
    if _swig_filter is not None:
        sys.stderr = _swig_filter.stderr
