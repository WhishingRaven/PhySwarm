# PhySwarm

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Webots 2023b](https://img.shields.io/badge/Webots-2023b-green.svg)](https://cyberbotics.com/)

This repository is the official implementation of the academic paper **"Physics-Informed Modeling and Control of Emergent Behaviors in Robot Swarms"**. 

The official project website is available [here](https://physwarm.github.io/).

`PhySwarm` is a physics-informed, micro-macro decentralized swarm control framework. It characterizes multi-stage emergent swarm behaviors as density field evolutions under physical constraints, which are integrated with executable robot physical movements. This repository is built on the open-source physics simulation platform Webots, using E-puck differential-wheeled robots as the experimental platform.

---

## Task Scenarios

The framework covers three core multi-agent collaborative task scenarios:
1. **Trail-Guided Swarm Foraging (Swarm Foraging)**
2. **Formation-Reconfigurable Swarm Navigation (Swarm Navigation)**
3. **Role-Adaptive Swarm Search and Rescue (Swarm Rescue)**

---

## 📂 Directory Structure

```text
PhySwarm/
├── controllers/
│   ├── epuck_controller/           # Low-level controller for E-puck robots in Webots
│   └── controllers/                # High-level decision controllers
│       ├── Swarm_Foraging/         
│       │   ├── supervisor_controller/  # Controllers and running scripts
│       │   ├── models/             # Trained model weight files
│       │   └── plot/               # Data plotting and video rendering scripts
│       ├── Swarm_Navigation/       
│       │   ├── supervisor_controller/
│       │   ├── models/
│       │   └── plot/
│       └── Swarm_Rescue/           
│           ├── supervisor_controller/
│           ├── models/
│           └── plot/
├── worlds/
│   ├── generate_wbt.py             # Simulation world generation script
│   └── generated_world.wbt         # Pre-generated Webots simulation environment file
├── requirements.txt                # Dependency list (pip)
└── environment.yml                 # Virtual environment configuration file (conda)
```

---

## Environment Setup

### 1. Install Webots Simulation Software
The simulation physical engine of this project is based on **Webots 2023b**.
- Please visit the [official Webots releases page](https://github.com/cyberbotics/webots/releases/tag/R2023b) to download and install the version compatible with your operating system.

### 2. Install Anaconda/Miniconda
It is recommended to use Anaconda or Miniconda to manage your Python virtual environments.
- Please visit the [official Anaconda website](https://www.anaconda.com/) or the [official Miniconda website](https://docs.conda.io/en/latest/miniconda.html) to complete the installation.

### 3. Configure Python Virtual Environment
This project is configured on Ubuntu **20.04** with Python **3.8.20**. You can install dependencies using either of the following methods:

**Method A: Use Conda configuration file (`environment.yml`)**
```bash
conda env create -f environment.yml
conda activate physwarm
```

**Method B: Use Pip dependency file (`requirements.txt`)**
```bash
# Create and activate environment
conda create -n physwarm python=3.8.20 -y
conda activate physwarm

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Clone this Repository
```bash
git clone https://github.com/SICC-Group/PhySwarm.git
cd PhySwarm
```

### 2. Generate Simulation World Files
The `worlds/` directory already contains the pre-generated `generated_world.wbt` scene file. If you need to customize the environment or regenerate the simulation world, navigate to that directory and run the script:
```bash
cd worlds
python generate_wbt.py
cd ..
```

### 3. Run Simulation Models
1. **Launch Simulation Environment**: Open Webots, select and load the `worlds/generated_world.wbt` world file.
2. **Run Controllers**: This project supports three task scenarios. Each scenario is pre-configured with a pre-trained model loading path (`model_dir`), allowing you to run the corresponding shell script directly for evaluation or continued training.
   
   For example, to run the **Trail-Guided Swarm Foraging (Swarm Foraging)** scenario:
   ```bash
   cd controllers/controllers/Swarm_Foraging/supervisor_controller
   bash train_mappo.sh
   ```
   For **Swarm_Navigation** or **Swarm_Rescue**, the procedure is similar. Simply navigate to their respective `supervisor_controller` directories and execute `bash train_mappo.sh`.

> **Note**: The low-level kinematics driver for the e-puck robots in Webots is managed by `controllers/epuck_controller`. Please ensure the controller settings for the Webots robot nodes are correctly bound.

---

## Data Visualization
The scripts used to generate the line plots, potential field figures, and multimedia videos (MP4) presented in the main text and supplementary materials are preserved in the `plot/` folders of each task directory.
To regenerate the figures or videos, navigate to the corresponding directory and execute the Python scripts:
```bash
cd controllers/controllers/Swarm_Foraging/plot
python draw_*.py
```

---

## License
This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. For more details on rights and limitations, please refer to the [LICENSE](LICENSE) file in the root directory.