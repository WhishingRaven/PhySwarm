import os
import random
import sys
from pathlib import Path

CONTROLLERS_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (CONTROLLERS_ROOT, SCENARIO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common.webots_runtime import configure_webots_runtime

configure_webots_runtime()

import numpy as np
import pandas as pd
import wandb
import socket
import setproctitle
import torch
from common.device import configure_torch, select_torch_device
from config import get_config
from utils.util import get_cent_act_dim, get_dim_from_space
from supervisor_rlcontroller import Epuck2Supervisor
from prey_runner import PREYRunner as Runner


def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    device = select_torch_device(all_args.device)
    configure_torch(all_args.seed, all_args.n_training_threads, all_args.deterministic)
    print(f"PyTorch device: {device}")

    # setup file to output tensorboard, hyperparameters, and saved models
    base_run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name

    run_dir = None 

    # 模式 A: 评估模式
    if all_args.model_dir and not all_args.resume_training:
        run_dir = Path(all_args.model_dir).parent 
        print(f"--- Mode: Evaluation. Using model from: {all_args.model_dir} ---")

    # 模式 B: 断点续训模式
    elif all_args.resume_training and all_args.model_dir:
        run_dir = Path(all_args.model_dir).parent
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {run_dir}")
        print(f"--- Mode: Resume Training. Continuing in: {run_dir} ---")

    # 模式 C: 全新训练模式
    else:
        print(f"--- Mode: New Training ---")
        if not base_run_dir.exists():
            os.makedirs(str(base_run_dir))

        if not all_args.use_wandb:
            exist_run_nums = [int(str(folder.name).split('run')[1]) for folder in base_run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exist_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exist_run_nums) + 1)
            run_dir = base_run_dir / curr_run
            if not run_dir.exists():
                os.makedirs(str(run_dir))
            print(f"Creating new directory for this run: {run_dir}")
        else:
            run_dir = base_run_dir

    if all_args.use_wandb:
        # init wandb
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         dir=str(run_dir),
                         job_type="training",
                         reinit=False,
                         resume="allow",
                         id=run_dir.name if all_args.resume_training else None)

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(
        all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    np.random.seed(all_args.seed)
    random.seed(all_args.seed)

    num_agents = all_args.num_agents
    env = Epuck2Supervisor(all_args)

    if all_args.share_policy:
        print(env.agent_state_spaces[0])
        policy_info = {
            'policy_0': {"cent_obs_dim": get_dim_from_space(env.agent_state_spaces[0]),
                         "cent_act_dim": get_cent_act_dim(env.agent_action_spaces),
                         "obs_space": env.agent_observation_spaces[0],
                         "share_obs_space": env.agent_state_spaces[0],
                         "act_space": env.agent_action_spaces[0]}
        }

        def policy_mapping_fn(id): return 'policy_0'
    else:
        policy_info = {
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(env.agent_state_spaces[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(env.agent_action_spaces),
                                        "obs_space": env.agent_observation_spaces[agent_id],
                                        "share_obs_space": env.agent_state_spaces[agent_id],
                                        "act_space": env.agent_action_spaces[agent_id]}
            for agent_id in range(num_agents)
        }

        def policy_mapping_fn(agent_id): return 'policy_' + str(agent_id)
    
    adj = torch.zeros((all_args.num_agents,all_args.num_factor),dtype=torch.int64)            
    config = {"args": all_args,
              "policy_info": policy_info,
              "policy_mapping_fn": policy_mapping_fn,
              "env": env,
              "num_agents": num_agents,
              "device": device,
              "run_dir": run_dir,
              "use_same_share_obs": all_args.use_same_share_obs,
              "use_available_actions": all_args.use_available_actions,
              "adj": adj}

    total_num_steps = 0
    runner = Runner(config=config)
    
    progress_filename = os.path.join(run_dir,'config.csv')
    df = pd.DataFrame(list(all_args.__dict__.items()),columns=['Name', 'Value'])
    df.to_csv(progress_filename,index=False)

    try:
        # 评估模式
        if all_args.model_dir and not all_args.resume_training:
            runner.restore() 
            runner.eval()
        # 续训模式
        elif all_args.resume_training and all_args.model_dir:
            if not os.path.exists(runner.checkpoint_path):
                 raise FileNotFoundError(f"Checkpoint file not found for resuming: {runner.checkpoint_path}")
        
            runner.restore()
            runner.load_checkpoint()
            
            total_num_steps = runner.total_env_steps
            while total_num_steps < all_args.num_env_steps:
                total_num_steps = runner.run()
        # 全新训练模式
        else:
            total_num_steps = 0
            while total_num_steps < all_args.num_env_steps:
                total_num_steps = runner.run()
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final checkpoint...")
        runner.save(is_checkpoint=True)
        print("Final checkpoint saved. Exiting.")

    env.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
