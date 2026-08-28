#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir"

: "${WEBOTS_HOME:=/Applications/Webots.app}"
: "${WEBOTS_CONTROLLER_URL:=ipc://1234/supervisor}"
export WEBOTS_HOME
export WEBOTS_CONTROLLER_URL
export PYTHONPATH="$WEBOTS_HOME/Contents/lib/controller/python${PYTHONPATH:+:$PYTHONPATH}"
export DYLD_LIBRARY_PATH="$WEBOTS_HOME/Contents/lib/controller${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

env="Dynamical_system"
algo="mappo"
exp="debug"
name="jzx"
seed_max=18
seed_min=18

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

seed=${seed_min}
while [ "$seed" -le "$seed_max" ]; do
    echo "seed is ${seed}:"

    python train_prey.py --device auto --user_name "${name}" --env_name "${env}" --algorithm_name "${algo}" --experiment_name "${exp}" \
    --seed "${seed}" --use_feature_normalization \
    --episode_length 600 --use_soft_update  --hard_update_interval_episode 2000 --num_env_steps 2000000 --n_training_threads 8 \
    --msg_iterations 4  --adj_output_dim 32  --eval_interval 100000 --num_eval_episodes 50 --buffer_size 32 --num_mini_batch 4 --data_chunk_length 50 --log_interval 600 --save_interval 50000 \
    --highest_orders 6 --lr 3e-4 --critic_lr 5e-4 --train_interval_episode 32 --gamma 0.99 --use_valuenorm --use_linear_lr_decay \
    --entropy_coef 0.0 --capture_freezes --num_rank 1 --sparsity 0.3  --gain 0.01 --gae_lambda 0.95  --use_vfunction \
    --n_rollout_threads 1 --num_agents 8  --num_obs_targets 0 --num_target 0 --model_dir ../models/
    seed=$((seed + 1))
done
