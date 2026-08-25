#!/bin/sh
env="Dynamical_system"
algo="mappo"
exp="debug"
name="jzx"
seed_max=42
seed_min=42

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"

    CUDA_VISIBLE_DEVICES=0 python train_prey.py --user_name ${name} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --seed ${seed} --use_feature_normalization \
    --episode_length 600 --use_soft_update  --hard_update_interval_episode 2000 --num_env_steps 2000000 --n_training_threads 8 \
    --msg_iterations 4  --adj_output_dim 32  --eval_interval 100000 --num_eval_episodes 50 --buffer_size 32 --num_mini_batch 4 --log_interval 2400 --save_interval 50000 \
    --highest_orders 6 --lr 3e-4 --critic_lr 5e-4 --train_interval_episode 32 --gamma 0.99 --use_wandb --use_valuenorm --use_linear_lr_decay \
    --entropy_coef 0.0 --capture_freezes --num_rank 1 --sparsity 0.3  --gain 0.01 --gae_lambda 0.95  --use_vfunction \
    --n_rollout_threads 1 --num_agents 8 --num_target 1 --model_dir ../models/
done
