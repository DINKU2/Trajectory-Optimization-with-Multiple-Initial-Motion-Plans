"""
Helper functions for integrating VAMP RRT-Connect with TrajOpt
This demonstrates how to use VAMP paths as initial trajectories for TrajOpt
"""

import numpy as np
import vamp
from tesseract_robotics.tesseract_command_language import (
    JointWaypoint, MoveInstructionType_FREESPACE, MoveInstruction, CompositeInstruction,
    JointWaypointPoly_wrap_JointWaypoint, MoveInstructionPoly_wrap_MoveInstruction
)
from tesseract_robotics.tesseract_motion_planners_simple import generateInterpolatedProgram


def generate_vamp_RRTC_paths(start_config, goal_config, obstacle_centers, num_seeds=5, radius=0.2):
    """
    Generate multiple initial paths using VAMP's RRT-Connect planner with different random seeds.
    
    Args:
        start_config: Starting joint configuration (numpy array, 7 joints)
        goal_config: Goal joint configuration (numpy array, 7 joints)
        obstacle_centers: List of [x, y, z] sphere centers
        num_seeds: Number of different random seeds to try
        radius: Radius of obstacle spheres
    
    Returns:
        List of paths, where each path is a list of joint configurations (numpy arrays)
    """
    # Configure VAMP for Panda robot with RRT-Connect planner
    (vamp_module, planner_func, plan_settings, simp_settings) = vamp.configure_robot_and_planner_with_kwargs("panda", "rrtc")
    
    # Create VAMP environment with obstacles
    env = vamp.Environment()
    for sphere_center in obstacle_centers:
        env.add_sphere(vamp.Sphere(sphere_center, radius))
    
    # Convert numpy arrays to lists (VAMP expects lists)
    start_list = start_config.tolist()
    goal_list = goal_config.tolist()
    
    initial_paths = []
    successful_seeds = []
    
    print(f"Generating {num_seeds} initial paths with VAMP RRT-Connect...")
    
    for seed in range(num_seeds):
        # Create sampler with different seed
        sampler = vamp_module.halton()
        sampler.skip(seed * 1000)  # Different seed = different random sequence
        
        # Validate start and goal are collision-free
        if not vamp_module.validate(start_list, env):
            print(f"  Seed {seed}: Start configuration is in collision, skipping...")
            continue
        if not vamp_module.validate(goal_list, env):
            print(f"  Seed {seed}: Goal configuration is in collision, skipping...")
            continue
        
        # Run RRT-Connect planner
        result = planner_func(start_list, goal_list, env, plan_settings, sampler)
        
        if result.solved:
            # Interpolate to make path denser (more waypoints)
            # This gives TrajOpt more points to work with
            result.path.interpolate_to_resolution(vamp_module.resolution())
            
            # Convert VAMP path to list of numpy arrays
            path_waypoints = []
            for i in range(len(result.path)):
                config = result.path[i]
                # VAMP path is a list of joint values
                joint_values = [config[j] for j in range(7)]
                path_waypoints.append(np.array(joint_values, dtype=np.float64))
            
            initial_paths.append(path_waypoints)
            successful_seeds.append(seed)
            print(f"  Seed {seed}: Generated path with {len(path_waypoints)} waypoints")
        else:
            print(f"  Seed {seed}: Failed to find path")
    
    print(f"Successfully generated {len(initial_paths)} paths out of {num_seeds} seeds")
    return initial_paths, successful_seeds


def vamp_path_to_tesseract_program(vamp_path, joint_names, manip_info):
    """
    Convert a VAMP path (list of joint configurations) to Tesseract CompositeInstruction.
    
    Args:
        vamp_path: List of numpy arrays, each representing a joint configuration
        joint_names: List of joint names (e.g., ['panda_joint1', ..., 'panda_joint7'])
        manip_info: Tesseract ManipulatorInfo object
    
    Returns:
        CompositeInstruction ready for TrajOpt
    """
    program = CompositeInstruction("DEFAULT")
    program.setManipulatorInfo(manip_info)
    
    # Add each waypoint from VAMP path
    for config in vamp_path:
        wp = JointWaypoint(joint_names, config)
        instruction = MoveInstruction(
            JointWaypointPoly_wrap_JointWaypoint(wp),
            MoveInstructionType_FREESPACE,
            "DEFAULT"
        )
        program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(instruction))
    
    return program


def calculate_trajectory_length(trajectory):
    """
    Calculate the total length of a trajectory in joint space.
    
    Args:
        trajectory: Tesseract trajectory (flattened CompositeInstruction)
    
    Returns:
        Total trajectory length (sum of distances between consecutive waypoints)
    """
    from tesseract_robotics.tesseract_command_language import (
        InstructionPoly_as_MoveInstructionPoly, WaypointPoly_as_JointWaypointPoly, WaypointPoly_as_StateWaypointPoly
    )
    
    waypoints = []
    for instr in trajectory:
        if not instr.isMoveInstruction():
            continue
        move_instr = InstructionPoly_as_MoveInstructionPoly(instr)
        wp = move_instr.getWaypoint()
        if wp.isJointWaypoint():
            joint_wp = WaypointPoly_as_JointWaypointPoly(wp)
            waypoints.append(joint_wp.getPosition().flatten())
        elif wp.isStateWaypoint():
            state_wp = WaypointPoly_as_StateWaypointPoly(wp)
            waypoints.append(state_wp.getPosition().flatten())
    
    if len(waypoints) < 2:
        return 0.0
    
    trajectory_array = np.array(waypoints)
    total_length = 0.0
    for i in range(len(trajectory_array) - 1):
        diff = trajectory_array[i+1] - trajectory_array[i]
        total_length += np.linalg.norm(diff)
    
    return total_length


def get_trajectory_duration(trajectory):
    """
    Get the total duration of a time-parameterized trajectory.
    
    Args:
        trajectory: Tesseract trajectory (flattened CompositeInstruction)
    
    Returns:
        Total trajectory duration in seconds
    """
    from tesseract_robotics.tesseract_command_language import (
        InstructionPoly_as_MoveInstructionPoly, WaypointPoly_as_StateWaypointPoly
    )
    
    max_time = 0.0
    for instr in trajectory:
        if not instr.isMoveInstruction():
            continue
        move_instr = InstructionPoly_as_MoveInstructionPoly(instr)
        wp = move_instr.getWaypoint()
        if wp.isStateWaypoint():
            state_wp = WaypointPoly_as_StateWaypointPoly(wp)
            max_time = max(max_time, state_wp.getTime())
    
    return max_time
