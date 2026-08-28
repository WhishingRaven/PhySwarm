# macOS and Gymnasium migration

This migration moves PhySwarm from the former Ubuntu, Python 3.8, Gym, and CUDA baseline to macOS on Apple Silicon, Python 3.13, Gymnasium, PyTorch MPS with CPU fallback, and Webots R2025a.

The pre-migration source is commit `52393c4f8eb0b3df93bfb92ab4649884c21d1c71`. Simulation data and model checkpoints are not rewritten by this migration.

## Forward path

1. Keep the existing `webots` conda environment until the new runtime has passed verification. Do not update it in place.
2. Install the Apple Silicon build of Webots R2025a in `/Applications/Webots.app`.
3. Create the new environment from the repository root:

   ```bash
   conda env create -f environment.yml
   conda activate physwarm
   ```

4. Configure the Webots controller libraries for the active shell:

   ```bash
   export WEBOTS_HOME="/Applications/Webots.app"
   export WEBOTS_CONTROLLER_URL="ipc://1234/supervisor"
   export PYTHONPATH="$WEBOTS_HOME/Contents/lib/controller/python${PYTHONPATH:+:$PYTHONPATH}"
   export DYLD_LIBRARY_PATH="$WEBOTS_HOME/Contents/lib/controller${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
   ```

5. Update every caller to the Gymnasium contracts described below. The repository runners already use these contracts after this migration.
6. Run the automated and Webots smoke checks. Retire the old environment only after checkpoints load and all three scenarios complete a rollout.

## Compatibility boundary

The Gym to Gymnasium transition is an intentional API break. Each multi-agent environment now returns one structured observation:

```text
observation = ((observations, neighbor_counts), shared_state)
```

Its components retain their vectorized shapes:

- `observations`: `(n_envs, n_agents, n_observations)`
- `neighbor_counts`: `(n_envs, n_agents, n_neighbors)`
- `shared_state`: the existing task-specific vectorized state shape
- `rewards`, `terminated`, and `truncated`: `(n_envs, n_agents, 1)`

Reset now follows Gymnasium:

```python
observation, info = env.reset(seed=seed, options=options)
```

The full signature is `reset(*, seed=None, options=None)`. Step now follows Gymnasium:

```python
observation, rewards, terminated, truncated, info = env.step(actions)
done = terminated | truncated
```

`terminated` represents terminal task state, including agent death, all-agent completion, or an early-stop condition. `truncated` represents the episode-length limit. Callers that previously consumed one `done` array must combine both flags explicitly. `info` is a Gymnasium-compatible dictionary; per-environment reward and diagnostic metadata is stored in `info["per_env"]`.

Other compatibility changes:

- `gym` and the custom legacy `gym.Space` dependency are replaced by `gymnasium` spaces.
- Local TensorBoardX logging is now the safe default; the former inverted `--use_wandb` flag is replaced by explicit `--wandb`/`--no-wandb` options.
- CUDA device names, CUDA seeding, NVIDIA packages, Triton, and CUDA-only extensions are removed.
- Device selection prefers `mps` when `torch.backends.mps.is_available()` is true and otherwise uses `cpu`.
- Webots is no longer discovered through `/usr/local/webots`; macOS uses `WEBOTS_HOME` and the application bundle libraries.
- Worlds and PROTO references target Webots R2025a.

## Verification

Run dependency and unit checks from the repository root:

```bash
conda env create -f environment.yml
conda run -n physwarm python -m pip check
conda run -n physwarm python -c "import gymnasium, numpy, torch; print(torch.__version__)"
conda run -n physwarm pytest -q
```

Confirm the selected compute backend:

```bash
conda run -n physwarm python -c "import torch; print('mps' if torch.backends.mps.is_available() else 'cpu')"
```

For the Webots smoke check, export the variables from the forward path, open `worlds/generated_world.wbt` in Webots R2025a, start the simulation, and launch each scenario's external supervisor from its `supervisor_controller` directory. Verify that:

- the `controller` module imports from the R2025a application bundle;
- the external supervisor connects to the `<extern>` robot;
- one reset and step return the Gymnasium tuple shapes above;
- one short rollout completes on MPS, or on CPU when MPS is unavailable;
- existing model checkpoints load without device errors.

## Rollback

Keep rollback additive and recoverable. To inspect or run the old source without overwriting the migrated worktree:

```bash
git worktree add ../PhySwarm-pre-migration 52393c4f8eb0b3df93bfb92ab4649884c21d1c71
```

The old `environment.yml` creates the former `webots` environment and contains Linux/CUDA packages. Recreate it only on its original Linux x86-64 platform:

```bash
conda env create -f ../PhySwarm-pre-migration/environment.yml
conda activate webots
```

On macOS, keep using the migrated `physwarm` environment; the legacy environment is not Apple Silicon compatible. After the migration is committed, use `git revert <migration-commit>` if the branch itself must be rolled back. Do not delete either conda environment until the chosen version has passed its smoke test.
