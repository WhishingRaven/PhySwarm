# PhySwarm

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Webots R2025a](https://img.shields.io/badge/Webots-R2025a-green.svg)](https://cyberbotics.com/)

This repository is a macOS reproduction workbench for **“Physics-Informed Modeling and Control of Emergent Behaviors in Robot Swarms.”** It provides one shared rMAPPO pipeline, task-specific physics strategies, isolated Webots/Gymnasium adapters, and simulator-independent implementations of the paper's density, ADR, Micro-EDM, loss, metric, and analysis primitives. The checked-in models and short smoke tests are not a complete numerical reproduction of the paper; see the [implementation audit](docs/paper-implementation-audit.md) for the evidence and remaining gaps.

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

If Webots is installed somewhere else, point `WEBOTS_HOME` to its application bundle:

```bash
export WEBOTS_HOME="/Applications/Webots.app"
```

Verify the installation:

```bash
python -c "from adapters.webots.runtime import configure_webots_runtime; configure_webots_runtime(); import controller, gymnasium, torch; print(torch.__version__); print('mps' if torch.backends.mps.is_available() else 'cpu')"
python -m pip check
```

`requirements.txt` contains the same Python dependencies for tools that need a pip requirements file. The conda environment remains the supported installation path.

## Run a scenario

The checked-in world is ready to use. To regenerate it with Webots R2025a references:

```bash
python worlds/generate_wbt.py
```

The Python entrypoints manage Webots and the external controller lifecycle. `--webots` selects the simulation mode:

- `fast`: headless, no rendering, maximum simulation speed.
- `realtime`: visible Webots GUI in realtime mode.
- `existing` (default): use the Webots world you already opened yourself.

For a new training run in fast background mode:

```bash
python train.py foraging --webots fast
```

Any additional options are validated by the shared rMAPPO configuration. Evaluation and inference are explicit commands:

```bash
python evaluate.py foraging --webots fast \
  --model-dir controllers/Swarm_Foraging/models \
  --episodes 10

python infer.py foraging --webots realtime \
  --model-dir controllers/Swarm_Foraging/models
```

Use `navigation` or `rescue` in place of `foraging`. The Webots process is stopped when the Python command exits. Resume with `python train.py SCENARIO --webots fast --resume --model-dir PATH`. The adapter defaults to `ipc://1234/supervisor`; override it with `--controller-url`. Pass `--device mps` or `--device cpu` after the common options to force a backend. An unavailable requested MPS backend warns and falls back to CPU. The complete contract is in [docs/reproduction.md](docs/reproduction.md).

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

Generate a deterministic report from a trajectory or training CSV with:

```bash
conda run -n physwarm python -m analysis \
  --input /path/to/data.csv \
  --output /path/to/report \
  --scenario auto
```

The command writes a machine-readable summary and multiple task-aware plots without changing the input. See [docs/experiments.md](docs/experiments.md). The older figure-specific plotting and video scripts remain under each `controllers/Swarm_*/plot` directory.

New runs can emit that trajectory input directly with
`--save_trajectory --trajectory_output /absolute/path/to/run.csv`. Logging is
single-environment and non-overwriting by design; see
[docs/reproduction.md](docs/reproduction.md#7-실행-산출물) for the episode naming
contract.

## Documentation

- [Architecture and responsibility boundaries](docs/architecture.md)
- [Paper-to-code implementation audit](docs/paper-implementation-audit.md)
- [Training, evaluation, and reproduction procedure](docs/reproduction.md)
- [Metrics and experiment reports](docs/experiments.md)
- [Extending the methodology without changing the tasks](docs/extending-methodology.md)

## License

PhySwarm is licensed under the [GNU General Public License v3.0](LICENSE).
