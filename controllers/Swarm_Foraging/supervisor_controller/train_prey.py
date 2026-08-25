import math
import sys
import os
sys.path.append("/usr/local/webots/lib/controller/python")
sys.path.append("/usr/local/webots/lib/controller/python/controller")
sys.path.append('../')
import numpy as np
import pandas as pd
from pathlib import Path
import wandb
import socket
import setproctitle
import torch
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

    # cuda and # threads
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

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

    # set seeds
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    num_agents = all_args.num_agents
    env = Epuck2Supervisor(all_args)

    if all_args.share_policy:
        print(env.state_space[0])
        policy_info = {
            'policy_0': {"cent_obs_dim": get_dim_from_space(env.state_space[0]),
                         "cent_act_dim": get_cent_act_dim(env.action_space),
                         "obs_space": env.observation_space[0],
                         "share_obs_space": env.state_space[0],
                         "act_space": env.action_space[0]}
        }

        def policy_mapping_fn(id): return 'policy_0'
    else:
        policy_info = {
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(env.state_space[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(env.action_space),
                                        "obs_space": env.observation_space[agent_id],
                                        "share_obs_space": env.state_space[agent_id],
                                        "act_space": env.action_space[agent_id]}
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