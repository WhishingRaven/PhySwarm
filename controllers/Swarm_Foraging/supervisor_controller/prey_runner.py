import numpy as np
import torch
import time
from supervisor_controller.base_runner import RecRunner
from supervisor_controller.utils.util import make_onehot

class PREYRunner(RecRunner):
    def __init__(self, config):
        """Runner class for the StarcraftII environment (SMAC). See parent class for more information."""
        super(PREYRunner, self).__init__(config)
        # fill replay buffer with random actions
        self.start = time.time()
        
        self.train_infos = {}
        self.env_infos = {}
        
        self.log_clear()
    
    @torch.no_grad()
    def collect_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        Collect a rollout and store it in the buffer. All agents share a single policy.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}

        reward_components_info_episode = {}  # 动态收集所有指标键

        p_id = "policy_0"
        policy = self.policies[p_id]
        self.act_dim = self.policies['policy_0'].act_dim
        env = self.env

        env.reset()
        obs = env.get_observations()
        share_obs = env.get_state()
        obs_neighbor_num = env.obs_neighbor_num_buf.copy()

        rnn_states = np.zeros((self.num_envs, self.num_agents, self.hidden_size), dtype=np.float32)
        rnn_critic_states = np.zeros((self.num_envs, self.num_agents, self.hidden_size), dtype=np.float32)
        last_acts = np.zeros((self.num_envs, self.num_agents, self.act_dim), dtype=np.float32)

        
        episode_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.obs_dim), dtype=np.float32)}
        episode_obs_neighbor_num = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, self.num_obs_targets + self.num_obs_agents), dtype=np.int32)}
        episode_share_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.central_obs_dim), dtype=np.float32)}
        episode_acts = {p_id: np.zeros((self.episode_length, self.num_envs, self.num_agents, self.act_dim), dtype=np.float32)}
        episode_prob_acts = {p_id: np.zeros((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32)}
        episode_rewards = {p_id: np.zeros((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32)}
        episode_dones = {p_id: np.ones((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32)}
        episode_dones_env = {p_id: np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32)}
        episode_values = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, 1), dtype=np.float32)}
        episode_rnn_states = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, self.hidden_size), dtype=np.float32)}
        episode_rnn_critic = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, self.hidden_size), dtype=np.float32)}


        episode_obs[p_id][0] = obs
        episode_obs_neighbor_num[p_id][0] = obs_neighbor_num
        episode_share_obs[p_id][0] = share_obs
        episode_rnn_states[p_id][0] = rnn_states
        episode_rnn_critic[p_id][0] = rnn_critic_states


        for t in range(self.episode_length):
            obs_flat = obs.reshape(-1, policy.obs_dim)
            obs_neighbor_num_flat = obs_neighbor_num.reshape(-1, obs_neighbor_num.shape[-1])
            rnn_states_flat = rnn_states.reshape(-1, self.hidden_size)
            last_acts_flat = last_acts.reshape(-1, self.act_dim)
            actor_features, new_rnn_states_flat, _ = policy.get_hidden_states(obs_flat, last_acts_flat, rnn_states_flat, obs_neighbor_num_flat)
            rnn_critic_states_flat = rnn_critic_states.reshape(-1, self.hidden_size)
            critic_features, new_rnn_critic_states_flat, _ = policy.get_hidden_critic(share_obs, rnn_critic_states_flat)
            
            actor_input = actor_features.squeeze(0).reshape(self.num_envs, self.num_agents, -1)
            critic_input = critic_features.cpu().numpy().reshape(self.num_envs, self.num_agents, -1)
            
            acts_torch, log_probs_torch, values_torch = policy.get_actions(
                actor_input,
                torch.as_tensor(critic_input, device=self.device), 
                torch.as_tensor(obs, dtype=torch.float32, device=self.device),
                explore=explore
            )
            
            acts = acts_torch.cpu().numpy()
            log_probs = log_probs_torch.cpu().numpy()
            values = values_torch.cpu().numpy()
            new_rnn_states = new_rnn_states_flat.cpu().numpy().reshape(self.num_envs, self.num_agents, -1)

            next_obs_corr, next_share_obs, rewards, dones, info = env.step(acts)
            next_obs = next_obs_corr[0]
            next_obs_neighbor_num = next_obs_corr[1]

            for single_env_info in info:
                for key in single_env_info.keys():
                    if key not in reward_components_info_episode:
                        reward_components_info_episode[key] = []

            for key in reward_components_info_episode.keys():
                mean_val = np.mean([single_env_info.get(key, 0) for single_env_info in info])
                reward_components_info_episode[key].append(mean_val)

            if training_episode or warmup:
                self.total_env_steps += self.num_envs

            episode_acts[p_id][t] = acts
            episode_prob_acts[p_id][t] = log_probs
            episode_values[p_id][t] = values
            episode_rewards[p_id][t] = rewards
            episode_dones[p_id][t] = dones
            episode_dones_env[p_id][t] = np.all(dones, axis=1)

            obs = next_obs
            share_obs = next_share_obs
            obs_neighbor_num = next_obs_neighbor_num

            last_acts = acts
            
            rnn_states = new_rnn_states
            new_rnn_critic_states = new_rnn_critic_states_flat.cpu().numpy().reshape(self.num_envs, self.num_agents, -1)
            rnn_critic_states = new_rnn_critic_states

            done_mask = (1 - dones).astype(np.float32)
            rnn_states = rnn_states * done_mask

            env_done_mask = (1 - np.all(dones, axis=1)).astype(np.float32)
            env_done_mask = env_done_mask.reshape(self.num_envs, 1, 1) 
            rnn_critic_states = rnn_critic_states * env_done_mask
            
            episode_obs[p_id][t + 1] = next_obs
            episode_obs_neighbor_num[p_id][t + 1] = next_obs_neighbor_num
            episode_share_obs[p_id][t + 1] = next_share_obs
            episode_rnn_states[p_id][t + 1] = rnn_states
            episode_rnn_critic[p_id][t + 1] = rnn_critic_states

        obs_flat = obs.reshape(-1, policy.obs_dim)
        obs_neighbor_num_flat = obs_neighbor_num.reshape(-1, obs_neighbor_num.shape[-1])
        rnn_states_flat = rnn_states.reshape(-1, self.hidden_size)
        last_acts_flat = last_acts.reshape(-1, self.act_dim)
        
        _, _, _ = policy.get_hidden_states(obs_flat, last_acts_flat, rnn_states_flat, obs_neighbor_num_flat)
        rnn_critic_states_flat = rnn_critic_states.reshape(-1, self.hidden_size)
        final_critic_features, _, _ = policy.get_hidden_critic(share_obs, rnn_critic_states_flat)

        final_values_torch = policy.get_v_value(final_critic_features.shape[0], final_critic_features)
        final_values_reshaped = final_values_torch.reshape(self.num_envs, self.num_agents, -1)
        episode_values[p_id][self.episode_length] = final_values_reshaped.cpu().numpy()

        if explore:
            self.num_episodes_collected += self.num_envs
            idx = self.buffer.insert(self.num_envs, episode_obs, episode_share_obs, episode_acts, episode_prob_acts,
                                  episode_rewards, episode_dones, episode_dones_env, episode_values,
                                  episode_rnn_states, episode_rnn_critic, None, episode_obs_neighbor_num)
            self.buffer.compute_returns(idx, self.trainer.value_normalizer)

        all_agent_rewards = episode_rewards[p_id][:, :, :, 0]
        total_reward_per_environment = all_agent_rewards.sum(axis=(0, 2))
        average_team_reward = total_reward_per_environment.mean()
        env_info['average_episode_rewards'] = average_team_reward / self.num_agents
        env_info['num_collision'] = self.env.num_collision.sum() / (self.num_envs * self.num_agents)

        for key, values in reward_components_info_episode.items():
            env_info[key] = np.mean(values)

        return env_info

    def log(self):
        """See parent class."""
        self.log_train_continuous(self.train_infos)
        '''if self.use_dyn_graph:
            self.log_train_adj(p_id, self.train_adj_infos[0])'''

        self.log_env(self.env_infos)
        self.log_clear()

    def eval(self):
        """Collect episodes to evaluate the policy."""
        print("Eval start!")
        self.trainer.prep_rollout()

        eval_infos = {}  

        for _ in range(self.args.num_eval_episodes):
            env_info = self.collecter(explore=False, training_episode=False, warmup=False)
            
            for k, v in env_info.items():
                if k not in eval_infos:
                    eval_infos[k] = []
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")

    def log_clear(self):
        """See parent class."""

        for k in list(self.env_infos.keys()):
            self.env_infos[k] = []

        for k in self.train_infos.keys():
            self.train_infos[k] = []
