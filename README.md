# Trajectory Optimization with Multiple Initial Motion Plans

**Project 5** - Created for [Dr. Kavraki's Robotics Class](https://www.kavrakilab.org/)  
📋 [Project Assignment (PDF)](project.pdf) | 📄 [Project Report (PDF)](Trajopt%20Project.pdf)

---

## Problem and Motivation

Trajectory optimization aims to produce smooth, collision-free, dynamically feasible motions for robotic manipulators, but methods like TrajOpt are sensitive to initialization and often get trapped in poor local minima. For a 7-DOF Panda robot, a single random initial trajectory may not provide enough structure for TrajOpt to converge to a high-quality solution—especially in cluttered environments with strict velocity and acceleration limits.

Modern robots require motions that are safe, smooth, and efficient, which geometric planners alone cannot provide. Sampling-based planners like RRT-Connect quickly find feasible but non-smooth paths; trajectory optimization can refine these paths, but only if given a good starting point.

---

## How to Run

### Part 1 (Windows - Local)

Run on Windows locally using the setup script:

```powershell
# Setup environment
.\setup_local_env.ps1

# Activate environment (May already be activated from the setup script)
.\traj_opt_env\Scripts\Activate.ps1

# Run Part 1
python project/part_1.py
```

### Part 2 (Ubuntu 22.04 / WSL)

Run on Ubuntu 22.04 or WSL:

```bash
# Setup environment
python3 -m venv traj_opt_env_wsl
source traj_opt_env_wsl/bin/activate

pip install vamp-planner
pip install tesseract-robotics tesseract-robotics-viewer setuptools
pip install numpy

# Activate environment
source traj_opt_env_wsl/bin/activate

# Run Part 2
python project/part_2.py
```

---

## Installation

### Prerequisites

Install `tesseract_robotics` and `tesseract_robotics_viewer` (handled by setup scripts).

**Optional:** Clone Tesseract repositories to retrieve example assets (not required for this project):

```bash
git clone --depth=1 https://github.com/tesseract-robotics/tesseract.git
git clone --depth=1 https://github.com/tesseract-robotics/tesseract_planning.git
git clone --depth=1 https://github.com/tesseract-robotics/tesseract_python.git
```

### Environment Variables (Optional)

Set `TESSERACT_RESOURCE_PATH` and `TESSERACT_TASK_COMPOSER_CONFIG_FILE` if using Tesseract examples:

**Linux/WSL:**
```bash
export TESSERACT_RESOURCE_PATH=`pwd`/tesseract
export TESSERACT_TASK_COMPOSER_CONFIG_FILE=`pwd`/tesseract_planning/tesseract_task_composer/config/task_composer_plugins_no_trajopt_ifopt.yaml
```

**Windows:**
```cmd
set TESSERACT_RESOURCE_PATH=%CD%/tesseract
set TESSERACT_TASK_COMPOSER_CONFIG_FILE=%CD%/tesseract_planning/tesseract_task_composer/config/task_composer_plugins_no_trajopt_ifopt.yaml
```


