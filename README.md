# Trajectory Optimization with Multiple Initial Motion Plans


# This project was led by 

## Problem and Motivation

Trajectory optimization aims to produce smooth, collision-free, dynamically feasible motions for robotic manipulators, but methods like TrajOpt are sensitive to initialization and often get trapped in poor local minima. For a 7-DOF Panda robot, a single random initial trajectory may not provide enough structure for TrajOpt to converge to a high-quality solution—especially in cluttered environments with strict velocity and acceleration limits.

Modern robots require motions that are safe, smooth, and efficient, which geometric planners alone cannot provide. Sampling-based planners like RRT-Connect quickly find feasible but non-smooth paths; trajectory optimization can refine these paths, but only if given a good starting point. 


