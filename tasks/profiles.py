"""Runnable task profiles shared by training, evaluation, and inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskProfile:
    """Default MAPPO arguments for one paper task."""

    key: str
    controller_directory: str
    seed: int
    episode_length: int
    evaluation_episodes: int
    targets: int
    log_interval: int
    extra: tuple[str, ...] = ()

    def training_arguments(self) -> list[str]:
        args = [
            "--device", "auto",
            "--user_name", "physwarm",
            "--env_name", "Dynamical_system",
            "--algorithm_name", "mappo",
            "--experiment_name", "debug",
            "--seed", str(self.seed),
            "--use_feature_normalization",
            "--episode_length", str(self.episode_length),
            "--use_soft_update",
            "--hard_update_interval_episode", "2000",
            "--num_env_steps", "2000000",
            "--n_training_threads", "8",
            "--msg_iterations", "4",
            "--adj_output_dim", "32",
            "--eval_interval", "100000",
            "--num_eval_episodes", str(self.evaluation_episodes),
            "--buffer_size", "32",
            "--num_mini_batch", "4",
            "--log_interval", str(self.log_interval),
            "--save_interval", "50000",
            "--highest_orders", "6",
            "--lr", "3e-4",
            "--critic_lr", "5e-4",
            "--train_interval_episode", "32",
            "--gamma", "0.99",
            "--use_valuenorm",
            "--use_linear_lr_decay",
            "--entropy_coef", "0.0",
            "--num_rank", "1",
            "--sparsity", "0.3",
            "--gain", "0.01",
            "--gae_lambda", "0.95",
            "--use_vfunction",
            "--n_rollout_threads", "1",
            "--num_agents", "8",
            "--num_target", str(self.targets),
        ]
        return [*args, *self.extra]


TASK_PROFILES = {
    "foraging": TaskProfile(
        key="foraging",
        controller_directory="Swarm_Foraging",
        seed=42,
        episode_length=1000,
        evaluation_episodes=60,
        targets=2,
        log_interval=2400,
    ),
    "navigation": TaskProfile(
        key="navigation",
        controller_directory="Swarm_Navigation",
        seed=18,
        episode_length=600,
        evaluation_episodes=50,
        targets=0,
        log_interval=600,
        extra=("--data_chunk_length", "50", "--num_obs_targets", "0"),
    ),
    "rescue": TaskProfile(
        key="rescue",
        controller_directory="Swarm_Rescue",
        seed=42,
        episode_length=600,
        evaluation_episodes=50,
        targets=1,
        log_interval=2400,
    ),
}


def get_task_profile(name: str) -> TaskProfile:
    try:
        return TASK_PROFILES[name.lower()]
    except KeyError as error:
        choices = ", ".join(sorted(TASK_PROFILES))
        raise KeyError(f"unknown task {name!r}; choose {choices}") from error
