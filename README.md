# PhySwarm

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Webots R2025a](https://img.shields.io/badge/Webots-R2025a-green.svg)](https://cyberbotics.com/)

This repository is fork of **“Physics-Informed Modeling and Control of Emergent Behaviors in Robot Swarms”** for MacOS. It aims for rough implementation of method suggested in the paper, not the exact reproduction of the experiment. Libraries for this project is updated to the latest version as possible.

PhySwarm is a physics-informed, micro-macro decentralized swarm-control framework. It models multi-stage emergent behavior as density-field evolution under physical constraints and connects that model to executable motion for E-puck robots in Webots. The project website is [physwarm.github.io](https://physwarm.github.io/).

## Scenarios

- Trail-guided swarm foraging
- Formation-reconfigurable swarm navigation
- Role-adaptive swarm search and rescue

## Requirements

- An Apple Silicon Mac
- Miniconda, Anaconda, or another compatible `conda` installation
- The native macOS build of [Webots R2025a](https://github.com/cyberbotics/webots/releases/tag/R2025a), installed as `/Applications/Webots.app`

The supported runtime is Python 3.13 with Gymnasium and PyTorch. PyTorch automatically uses the Apple Metal (`mps`) backend when it is available and otherwise falls back to the CPU. No CUDA, NVIDIA, Triton, Rosetta, or Linux toolchain is required.

## Setup

Create the reproducible conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate physwarm
```

If Webots is installed somewhere else, point `WEBOTS_HOME` to its application bundle. For an interactive shell, configure the controller runtime with:

```bash
export WEBOTS_HOME="/Applications/Webots.app"
export PYTHONPATH="$WEBOTS_HOME/Contents/lib/controller/python${PYTHONPATH:+:$PYTHONPATH}"
export DYLD_LIBRARY_PATH="$WEBOTS_HOME/Contents/lib/controller${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Verify the installation:

```bash
python -c "import controller, gymnasium, torch; print(torch.__version__); print('mps' if torch.backends.mps.is_available() else 'cpu')"
python -m pip check
```

`requirements.txt` contains the same Python dependencies for tools that need a pip requirements file. The conda environment remains the supported installation path.

## Run a scenario

The checked-in world is ready to use. To regenerate it with Webots R2025a references:

```bash
python worlds/generate_wbt.py
```

The scenario launchers can manage Webots for you. Their first argument selects the simulation mode:

- `fast`: headless, no rendering, maximum simulation speed.
- `slow`: visible Webots GUI in realtime mode.
- `existing` (default): use the Webots world you already opened yourself.

For a fast background evaluation:

```bash
conda activate physwarm
sh controllers/Swarm_Foraging/supervisor_controller/train_mappo.sh fast
```

For visible realtime evaluation, use `slow` instead. The Webots process started by either mode is stopped when the evaluation exits. The launcher accepts additional `train_prey.py` options after the mode; for a short check, use:

```bash
sh controllers/Swarm_Foraging/supervisor_controller/train_mappo.sh fast --episode_length 20 --num_eval_episodes 1
```

The navigation and rescue launchers are at:

```text
controllers/Swarm_Navigation/supervisor_controller/train_mappo.sh
controllers/Swarm_Rescue/supervisor_controller/train_mappo.sh
```

Each launcher configures the native Webots libraries, defaults `WEBOTS_CONTROLLER_URL` to `ipc://1234/supervisor`, and passes `--device auto`. Override `WEBOTS_CONTROLLER_URL` when Webots uses another port. To force a backend, invoke `train_prey.py` from the relevant `supervisor_controller` directory with `--device mps` or `--device cpu`. An unavailable requested MPS backend emits a warning and safely uses the CPU.

Local TensorBoardX logging is the default. Pass `--wandb` only when you intentionally want to enable Weights & Biases network logging.

## Gymnasium API

The vectorized supervisors use Gymnasium's reset and step contracts:

```python
observation, info = env.reset(seed=seed)
observation, rewards, terminated, truncated, info = env.step(actions)
done = terminated | truncated
```

The structured observation is `((agent_observations, neighbor_counts), shared_state)`. Per-environment diagnostics are stored in `info["per_env"]`. See [MIGRATION.md](MIGRATION.md) for the compatibility boundary and rollback procedure.

## Tests

Run the automated checks in the supported environment:

```bash
conda run -n physwarm python -m pip check
conda run -n physwarm pytest -q
```

The plotting and video scripts for each task are under the corresponding `controllers/Swarm_*/plot` directory.

## License

PhySwarm is licensed under the [GNU General Public License v3.0](LICENSE).
