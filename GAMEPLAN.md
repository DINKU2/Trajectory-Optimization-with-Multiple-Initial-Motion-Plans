# Project 5: Trajectory Optimization with Multiple Initial Motion Plans - Gameplan

## Overview
Create a trajectory optimization system that generates multiple initial motion plans and optimizes them to find the best trajectory for the Panda robot navigating around obstacles.

## Assets Available
- **Robot**: Panda 7-DOF arm (`assets/panda/`)
  - `panda.urdf` - Robot definition
  - `panda.srdf` - Semantic robot description (collision pairs, groups, etc.)
  - `panda_spherized.urdf` - Simplified collision model
- **Obstacles**: 14 obstacle positions in `assets/obstacles/obstacles.txt`

## Step-by-Step Implementation Plan

### Phase 1: Environment Setup
1. **Initialize Tesseract Environment**
   - Load Panda URDF/SRDF from `assets/panda/` (not using package:// URLs)
   - Set up resource locator pointing to assets folder
   - Configure manipulator info (panda_arm group, TCP frame, working frame)

2. **Add Obstacles to Environment**
   - Parse `assets/obstacles/obstacles.txt` (14 [x, y, z] positions)
   - Create collision objects (spheres or cylinders) at each position
   - Add them to the Tesseract environment

3. **Set Initial Robot State**
   - Define starting joint positions
   - Set environment state

### Phase 2: Define Planning Problem
1. **Define Start and Goal**
   - Start pose: Cartesian or joint space
   - Goal pose: Cartesian or joint space
   - Create waypoints for planning

2. **Configure Manipulator Info**
   - TCP frame: `panda_link8` (end effector)
   - Manipulator group: `panda_arm`
   - Working frame: `panda_link0` (base) or `world`

### Phase 3: Generate Multiple Initial Plans
**Key Concept**: Generate multiple different initial trajectories to explore different parts of the configuration space.

**Strategies for Multiple Initial Plans**:
1. **Multiple OMPL Runs** with different:
   - Random seeds
   - Different planners (RRTConnect, RRT, PRM, etc.)
   - Different time limits
   - Different simplification settings

2. **Different Start Configurations**: 
   - Slight variations in start state
   - Different intermediate waypoints

3. **Interpolation Variations**:
   - Different interpolation densities
   - Linear vs. spline interpolation

**Implementation**:
```python
# Run OMPL planner N times with different configurations
initial_plans = []
for i in range(num_plans):
    # Configure OMPL with different settings
    # Generate plan
    # Store initial plan
```

### Phase 4: Optimize Each Plan
For each initial plan:
1. **Interpolate** to get dense waypoints
2. **Create TrajOpt Request** with the initial plan as seed
3. **Run TrajOpt** to optimize the trajectory
4. **Apply Time Parameterization** for realistic timing
5. **Store optimized trajectory**

### Phase 5: Evaluate and Compare
Evaluate each optimized trajectory:
1. **Trajectory Metrics**:
   - Total time/duration
   - Path length (joint space or Cartesian)
   - Smoothness (jerk, acceleration)
   - Distance from obstacles
   - Number of waypoints

2. **Ranking Criteria**:
   - Shortest time
   - Shortest path
   - Smoothest motion
   - Combination (weighted cost function)

### Phase 6: Visualization and Output
1. **Display Best Trajectory**:
   - Use TesseractViewer
   - Show animation at http://localhost:8000
   - Plot trajectory with obstacles

2. **Report Results**:
   - Print metrics for all trajectories
   - Highlight best trajectory
   - Save trajectory data if needed

## Key Differences from Example
- **Use local file paths** instead of `package://` URLs
- **Add obstacles** to the environment
- **Multiple planning runs** instead of single run
- **Comparison and selection** of best trajectory

## Code Structure
```
your_project_script.py
├── Setup
│   ├── Load environment from assets
│   ├── Add obstacles
│   └── Configure manipulator
├── Generate Initial Plans
│   ├── Loop: Generate N initial plans
│   └── Store each plan
├── Optimize Plans
│   ├── Loop: Optimize each initial plan
│   └── Store optimized trajectories
├── Evaluate
│   ├── Compute metrics for each
│   └── Rank trajectories
└── Visualize
    ├── Display best trajectory
    └── Show metrics
```

## Technical Details

### Panda Robot Info
- **7 joints**: `panda_joint1` through `panda_joint7`
- **Manipulator group**: `panda_arm` (base: `panda_link0`, tip: `panda_link8`)
- **Joint limits**: Check URDF for limits

### Obstacle Format
- 14 obstacles at [x, y, z] positions
- Heights appear to be at z=0.25 and z=0.80
- Create collision objects (spheres recommended for simplicity)

### Planning Pipeline
1. OMPL → Generate initial collision-free path
2. Interpolate → Dense waypoints
3. TrajOpt → Optimize trajectory
4. Time Parameterization → Add realistic timing

## Next Steps
1. Start with basic script structure (like example)
2. Modify to use local Panda files
3. Add obstacle loading
4. Implement multiple plan generation loop
5. Add comparison logic
6. Test and refine


