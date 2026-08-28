import os
import numpy as np
import wandb
import torch
from tensorboardX import SummaryWriter
import pandas as pd
import time
from supervisor_controller.utils.util import get_cent_act_dim, get_dim_from_space
from supervisor_controller.utils.util import DecayThenFlatSchedule
from supervisor_controller.utils.ppo_buffer import PPOBuffer
class RecRunner(object):
    """Base class for training recurrent policies."""

    def __init__(self, config):
        """
        Base class for training recurrent policies.
        :param config: (dict) Config dictionary containing parameters for training.
        """
        self.args = config["args"]
        self.device = config["device"]
        self.adj = config["adj"]
        self.q_learning = ["qplex","qtran","wqmix","qmix","vdn","rddfg_cent_rw","rddfg_continuous","sopcg","casec"]

        self.share_policy = self.args.share_policy
        self.algorithm_name = self.args.algorithm_name
        self.env_name = self.args.env_name
        self.num_env_steps = self.args.num_env_steps
        self.use_wandb = self.args.use_wandb
        self.use_reward_normalization = self.args.use_reward_normalization
        self.use_popart = self.args.use_popart
        self.use_per = self.args.use_per
        self.per_alpha = self.args.per_alpha
        self.per_beta_start = self.args.per_beta_start
        self.buffer_size = self.args.buffer_size
        self.batch_size = self.args.batch_size
        self.adj_buffer_size = self.args.adj_buffer_size
        self.hidden_size = self.args.hidden_size
        self.highest_orders = self.args.highest_orders
        self.use_soft_update = self.args.use_soft_update
        self.hard_update_interval_episode = self.args.hard_update_interval_episode
        self.popart_update_interval_step = self.args.popart_update_interval_step
        self.actor_train_interval_step = self.args.actor_train_interval_step
        self.train_interval_episode = self.args.train_interval_episode
        self.train_adj_episode = self.args.train_adj_episode
        self.drop_temperature_episode = self.args.drop_temperature_episode
        self.train_interval = self.args.train_interval
        self.use_dyn_graph = self.args.use_dyn_graph
        self.equal_vdn = self.args.equal_vdn
        self.use_eval = self.args.use_eval
        self.eval_interval = self.args.eval_interval
        self.save_interval = self.args.save_interval
        self.log_interval = self.args.log_interval
        self.gae_lambda = self.args.gae_lambda
        self.gamma = self.args.gamma
        self.use_linear_lr_decay = self.args.use_linear_lr_decay
        self.independent_p_q = self.args.independent_p_q
        self.pair_rnn_hidden_dim = self.args.pair_rnn_hidden_dim
        self.epsilon_anneal_time = self.args.epsilon_anneal_time
        self.total_env_steps = 0  # total environment interactions collected during training
        self.num_episodes_collected = 0  # total episodes collected during training
        self.total_train_steps = 0  # number of gradient updates performed
        self.last_train_episode = 0  # last episode after which a gradient update was performed
        self.last_train_adj_episode = 0
        self.last_drop_t_episode = 0
        self.last_eval_T = 0  # last episode after which a eval run was conducted
        self.last_save_T = 0  # last epsiode after which the models were saved
        self.last_log_T = 0 # last timestep after which information was logged
        self.last_hard_update_episode = 0 # last episode after which target policy was updated to equal live policy
        self.use_vfunction = self.args.use_vfunction
        self.use_save = self.args.use_save
        self.pretrain_adj = self.args.pretrain_adj
        self.num_mini_batch = self.args.num_mini_batch
        self.adj_begin_step = self.args.adj_begin_step
        self.use_adj_init = self.args.use_adj_init
        self.num_obs_targets = self.args.num_obs_targets
        self.num_obs_agents = self.args.num_obs_agents
        
        
        if config.__contains__("take_turn"):
            self.take_turn = config["take_turn"]
        else:
            self.take_turn = False

        if config.__contains__("use_same_share_obs"):
            self.use_same_share_obs = config["use_same_share_obs"]
        else:
            self.use_same_share_obs = False

        if config.__contains__("use_available_actions"):
            self.use_avail_acts = config["use_available_actions"]
        else:
            self.use_avail_acts = False

        if config.__contains__("buffer_length"):
            self.episode_length = config["buffer_length"]
            if self.args.use_naive_recurrent_policy:
                self.data_chunk_length = config["buffer_length"]
            else:
                self.data_chunk_length = self.args.data_chunk_length
        else:
            self.episode_length = self.args.episode_length
            if self.args.use_naive_recurrent_policy:
                self.data_chunk_length = self.args.episode_length
            else:
                self.data_chunk_length = self.args.data_chunk_length
        #import pdb;pdb.set_trace()
        self.policy_info = config["policy_info"]
        self.policy_ids = sorted(list(self.policy_info.keys()))
        self.policy_mapping_fn = config["policy_mapping_fn"]
        if not self.share_policy:
            raise NotImplementedError(
                "PREYRunner data collection currently stores rollout data for "
                "policy_0 only. Keep --share_policy enabled or extend "
                "prey_runner.py for multiple policies."
            )

        self.num_agents = config["num_agents"]
        # self.num_factor = self.args.num_factor
        self.agent_ids = [i for i in range(self.num_agents)]

        self.env = config["env"]
        # no parallel envs
        self.num_envs = self.args.n_rollout_threads
        self.action_repr_updating = True
        # dir
        #import pdb;pdb.set_trace()
        self.model_dir = self.args.model_dir
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            #import pdb;pdb.set_trace()
            self.run_dir = config["run_dir"]
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.checkpoint_path = self.save_dir + '/' +'checkpoint.pt'

        # initialize all the policies and organize the agents corresponding to each policy
        if self.algorithm_name == "mappo":
            from supervisor_controller.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            from supervisor_controller.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        else:
            raise NotImplementedError
        
        self.collecter = self.collect_rollout
        self.saver = self.save
        self.restorer = self.restore
        self.train = self.batch_train
        self.policies = {p_id: Policy(config, self.policy_info[p_id]) for p_id in self.policy_ids}

        # initialize trainer class for updating policies
        if self.algorithm_name == "mappo":
            self.obs_dim = get_dim_from_space(self.policy_info[self.policy_ids[0]]["obs_space"])
            self.state_dim = get_dim_from_space(self.policy_info[self.policy_ids[0]]["share_obs_space"])
            self.trainer = TrainAlgo(self.args, self.num_agents, self.policies, self.policy_mapping_fn,device=self.device, episode_length=self.episode_length)
        
        # if self.model_dir is not None:
        #     if self.pretrain_adj:
        #         self.load_adj()
        #     else:
        #         self.restorer()
            
        # map policy id to agent ids controlled by that policy
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in self.agent_ids if self.policy_mapping_fn(agent_id) == policy_id]) for policy_id in
            self.policies.keys()}

        self.policy_obs_dim = {
            policy_id: self.policies[policy_id].obs_dim for policy_id in self.policy_ids}
        self.policy_act_dim = {
            policy_id: self.policies[policy_id].act_dim for policy_id in self.policy_ids}
        self.policy_central_obs_dim = {
            policy_id: self.policies[policy_id].central_obs_dim for policy_id in self.policy_ids}

        num_train_episodes = (self.num_env_steps / self.episode_length) / (self.train_interval_episode)
        self.beta_anneal = DecayThenFlatSchedule(
            self.per_beta_start, 1.0, num_train_episodes, decay="linear")

        self.buffer = PPOBuffer(self.policy_info,
                                      self.policy_agents,
                                      self.num_agents,
                                      self.buffer_size,
                                      self.episode_length,
                                      self.use_same_share_obs,
                                      self.use_avail_acts,
                                      self.use_reward_normalization,
                                      self.num_obs_agents + self.num_obs_targets,
                                      self.gamma,
                                      self.gae_lambda,
                                      self.hidden_size,
                                      is_centralized_critic_rnn=False)

    def run(self):
        """
        Collects a rollout for training, performs training updates, and handles logging, saving, and periodic evaluation.
        This function is intended to be called within a training loop.
        """
        # 1. Prepare for data collection and decay learning rate
        self.trainer.prep_rollout()
        if self.use_linear_lr_decay:
            self.trainer.lr_decay(self.total_env_steps, self.num_env_steps)

        # The run() function is always for training, so we always explore.
        env_info = self.collecter(explore=True, training_episode=True, warmup=False)

        # 2. Store collected environment info for logging
        for k, v in env_info.items():
            if k not in self.env_infos:
                self.env_infos[k] = []
            self.env_infos[k].append(v)

        # 3. Train the policies if enough new episodes have been collected
        if ((self.num_episodes_collected - self.last_train_episode) / self.train_interval_episode) >= 1:
            self.train()
            self.buffer.after_update()
            self.total_train_steps += 1
            self.last_train_episode = self.num_episodes_collected           
            
        # 4. Log training and environment info periodically
        if ((self.total_env_steps - self.last_log_T) / self.log_interval) >= 1:
            self.last_log_T = self.total_env_steps
            self.log()
            
        # 5. Perform periodic evaluation during training
        if self.use_eval and ((self.total_env_steps - self.last_eval_T) / self.eval_interval) >= 1:
            self.last_eval_T = self.total_env_steps
            self.eval()
            
        # 6. Save the models periodically
        if self.use_save and ((self.total_env_steps - self.last_save_T) / self.save_interval) >= 1:
            self.last_save_T = self.total_env_steps
            self.saver(is_checkpoint=True)

        return self.total_env_steps

    def batch_train(self):
        """Do a q-learning update to policy (used for QMix and VDN)."""
        self.trainer.prep_training()
        # gradient updates
        self.train_infos = {}
        data_chunk_length = self.data_chunk_length

        for epoch in range(15):
            # print(f"\n--- Starting PPO Epoch {epoch + 1}/10 ---")
            for p_id in self.policy_info.keys():
                data_generator = self.buffer.policy_buffers[p_id].sample_inds(data_chunk_length, self.num_mini_batch, self.trainer.value_normalizer[p_id])
                for i, sample in enumerate(data_generator):
                    train_info, _, _ = self.trainer.train_policy_on_batch(sample, epoch=epoch, mini_batch_idx=i)
                    for k, v in train_info.items():
                        if k not in self.train_infos:
                            self.train_infos[k] = []
                        self.train_infos[k].append(v)


    def save(self, is_checkpoint=False):
        """
        Saves model weights and training state.
        :param is_checkpoint: (bool) If True, saves a full training checkpoint. Otherwise, saves only model weights.
        """
        # 1. 保存模型权重
        model_save_path = self.save_dir
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        for pid in self.policy_ids:
            p_save_path = os.path.join(model_save_path, str(pid))
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)
            
            torch.save(self.policies[pid].rnn_network.state_dict(), os.path.join(p_save_path, 'rnn_network.pt'))
            torch.save(self.policies[pid].act.state_dict(), os.path.join(p_save_path, 'act.pt'))
            if self.use_vfunction:
                torch.save(self.policies[pid].rnn_critic_network.state_dict(), os.path.join(p_save_path, 'rnn_critic.pt'))
                torch.save(self.policies[pid].v_network.state_dict(), os.path.join(p_save_path, 'v_network.pt'))

        # 2. 如果是保存检查点，则额外保存训练状态
        if is_checkpoint:
            checkpoint = {
                'total_env_steps': self.total_env_steps,
                'num_episodes_collected': self.num_episodes_collected,
                'last_train_episode': self.last_train_episode,
                'last_save_T': self.last_save_T,
                'last_log_T': self.last_log_T,
                'last_eval_T': self.last_eval_T,
                'actor_optimizer_state_dict': self.trainer.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.trainer.critic_optimizer.state_dict(),
            }
            # 保存 Value Normalizer 的状态
            if self.trainer._use_valuenorm:
                value_normalizer_state = {p_id: self.trainer.value_normalizer[p_id].state_dict() for p_id in self.policy_ids}
                checkpoint['value_normalizer_state'] = value_normalizer_state

            torch.save(checkpoint, self.checkpoint_path)
            print(f"Checkpoint saved to {self.checkpoint_path} at step {self.total_env_steps}")

    def load_checkpoint(self):
        """Loads training state from a checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            print(f"Warning: Checkpoint file not found at {self.checkpoint_path}. Starting from scratch.")
            return

        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # 恢复训练进度
        self.total_env_steps = checkpoint['total_env_steps']
        self.num_episodes_collected = checkpoint['num_episodes_collected']
        self.last_train_episode = checkpoint['last_train_episode']
        self.last_save_T = checkpoint['last_save_T']
        self.last_log_T = checkpoint['last_log_T']
        self.last_eval_T = checkpoint['last_eval_T']

        # 恢复优化器状态
        self.trainer.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.trainer.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # 恢复 Value Normalizer 状态
        if self.trainer._use_valuenorm and 'value_normalizer_state' in checkpoint:
            for p_id, state_dict in checkpoint['value_normalizer_state'].items():
                self.trainer.value_normalizer[p_id].load_state_dict(state_dict)

        print(f"Resumed training from step {self.total_env_steps}")

    def restore(self):
        """Load policies from pretrained models."""
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid) 
            print(f"Loading pretrained model for policy {pid} from {path}")
            
            map_location = self.device
            
            try:
                rnn_state_dict = torch.load(path + '/rnn_network.pt', map_location=map_location, weights_only=True)
                self.policies[pid].rnn_network.load_state_dict(rnn_state_dict)
                
                act_state_dict = torch.load(path + '/act.pt', map_location=map_location, weights_only=True)
                self.policies[pid].act.load_state_dict(act_state_dict)
                
                if self.use_vfunction:
                    try:
                        v_network_state_dict = torch.load(path + '/v_network.pt', map_location=map_location, weights_only=True)
                        self.policies[pid].v_network.load_state_dict(v_network_state_dict)
                        
                        rnn_critic_state_dict = torch.load(path + '/rnn_critic.pt', map_location=map_location, weights_only=True)
                        self.policies[pid].rnn_critic_network.load_state_dict(rnn_critic_state_dict)
                    except RuntimeError as e:
                        if "size mismatch" in str(e):
                            print(f"Warning: Skipping Critic loading due to dimension mismatch.")
                        else:
                            raise e
            except FileNotFoundError:
                print(f"Warning: Could not find model files in {path}. Skipping loading for policy {pid}.")
            except Exception as e:
                print(f"An error occurred while loading models for policy {pid}: {e}")

    def log(self):
        """Log relevent training and rollout colleciton information.."""
        raise NotImplementedError

    def log_clear(self):
        """Clear logging variables so they do not contain stale information."""
        raise NotImplementedError

    def _append_csv_row(self, filename, log_data):
        new_row = pd.DataFrame([log_data])
        if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
            new_row.to_csv(filename, index=False)
            return

        try:
            old_data = pd.read_csv(filename)
        except pd.errors.EmptyDataError:
            new_row.to_csv(filename, index=False)
            return

        all_columns = list(old_data.columns)
        for column in new_row.columns:
            if column not in all_columns:
                all_columns.append(column)

        old_data = old_data.reindex(columns=all_columns)
        new_row = new_row.reindex(columns=all_columns)
        pd.concat([old_data, new_row], ignore_index=True).to_csv(filename, index=False)

    def log_env(self, env_info, suffix=None):
        """
        Log information related to the environment.
        Automatically handles CSV file creation and header writing.
        """
        if suffix == "eval_":
            progress_filename = os.path.join(self.run_dir, 'progress_eval.csv')
        else:
            progress_filename = os.path.join(self.run_dir, 'progress.csv')

        log_data = {'step': self.total_env_steps}

        for k, v in env_info.items():
            if len(v) > 0:
                value = np.mean(v)
                suffix_k = k if suffix is None else suffix + k 
                # print(suffix_k + " is " + str(value))
                if self.use_wandb:
                    wandb.log({suffix_k: value}, step=self.total_env_steps)
                else:
                    self.writter.add_scalar(suffix_k, value, self.total_env_steps)
                log_data[k] = value

        self._append_csv_row(progress_filename, log_data)
        
    def log_train_continuous(self, train_info):
        progress_filename = os.path.join(self.run_dir, 'progress_train.csv')   
        log_data = {'step': self.total_env_steps}
        
        for k, v in train_info.items():
            if len(v) > 0:
                value = np.mean(v)
                log_data[k] = value
                # print(k + " is " + str(value))
                if self.use_wandb:
                    wandb.log({k: value}, step=self.total_env_steps)
                else:
                    self.writter.add_scalar(k, value, self.total_env_steps)

        if len(log_data) == 1:
            return

        self._append_csv_row(progress_filename, log_data)
        
    def collect_rollout(self):
        """Collect a rollout and store it in the buffer."""
        raise NotImplementedError
