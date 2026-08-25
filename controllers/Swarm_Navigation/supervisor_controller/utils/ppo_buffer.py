import numpy as np
from supervisor_controller.utils.util import get_dim_from_space
import torch

def _cast(x):
    return x.transpose(2, 0, 1, 3)


class PPOBuffer(object):
    def __init__(self, policy_info, policy_agents, num_factor, buffer_size, episode_length, use_same_share_obs, use_avail_acts,use_reward_normalization=False,max_neighbor=1,gamma=0.97,gae_lambda=0.95,hidden_size=64, is_centralized_critic_rnn=False):
        """
        Replay buffer class for training RNN policies. Stores entire episodes rather than single transitions.

        :param policy_info: (dict) maps policy id to a dict containing information about corresponding policy.
        :param policy_agents: (dict) maps policy id to list of agents controled by corresponding policy.
        :param buffer_size: (int) max number of transitions to store in the buffer.
        :param use_same_share_obs: (bool) whether all agents share the same centralized observation.
        :param use_avail_acts: (bool) whether to store what actions are available.
        :param use_reward_normalization: (bool) whether to use reward normalization.
        """
        self.policy_info = policy_info
        self.policy_buffers = {p_id: AdjPolicyBuffer(buffer_size,
                                                     episode_length,
                                                     len(policy_agents[p_id]),
                                                     self.policy_info[p_id]['obs_space'],
                                                     self.policy_info[p_id]['share_obs_space'],
                                                     self.policy_info[p_id]['act_space'],
                                                     use_same_share_obs,
                                                     use_avail_acts,
                                                     use_reward_normalization,
                                                     max_neighbor,
                                                     gamma,
                                                     gae_lambda,
                                                     hidden_size,
                                                     is_centralized_critic_rnn)
                               for p_id in self.policy_info.keys()}

    def __len__(self):
        return self.policy_buffers['policy_0'].filled_i
      
    def insert(self, num_insert_episodes, obs, share_obs, acts, prob_acts, rewards, dones, dones_env,values=None,rnn_states=None,rnn_critic=None,avail_acts=None,obs_neighbor_num=None):
        """
        Insert a set of episodes into buffer. If the buffer size overflows, old episodes are dropped.

        :param num_insert_episodes: (int) number of episodes to be added to buffer
        :param obs: (dict) maps policy id to numpy array of observations of agents corresponding to that policy
        :param share_obs: (dict) maps policy id to numpy array of centralized observation corresponding to that policy
        :param acts: (dict) maps policy id to numpy array of actions of agents corresponding to that policy
        :param rewards: (dict) maps policy id to numpy array of rewards of agents corresponding to that policy
        :param dones: (dict) maps policy id to numpy array of terminal status of agents corresponding to that policy
        :param dones_env: (dict) maps policy id to numpy array of terminal status of env
        :param valid_transition: (dict) maps policy id to numpy array of whether the corresponding transition is valid of agents corresponding to that policy
        :param avail_acts: (dict) maps policy id to numpy array of available actions of agents corresponding to that policy

        :return: (np.ndarray) indexes in which the new transitions were placed.
        """
        for p_id in self.policy_info.keys():
            p_avail_acts = np.array(avail_acts[p_id]) if avail_acts is not None else None
        
            idx_range = self.policy_buffers[p_id].insert(num_insert_episodes, np.array(obs[p_id]),
                                                        np.array(share_obs[p_id]), np.array(acts[p_id]),
                                                        np.array(prob_acts[p_id]), np.array(rewards[p_id]),
                                                        np.array(dones[p_id]), np.array(dones_env[p_id]),
                                                        np.array(values[p_id]), np.array(rnn_states[p_id]),
                                                        np.array(rnn_critic[p_id]), 
                                                        p_avail_acts, 
                                                        np.array(obs_neighbor_num[p_id]))
        return idx_range

    # def sample(self, batch_size,data_chunk_length):
    #     """
    #     Sample a set of episodes from buffer, uniformly at random.
    #     :param batch_size: (int) number of episodes to sample from buffer.

    #     :return: obs: (dict) maps policy id to sampled observations corresponding to that policy
    #     :return: share_obs: (dict) maps policy id to sampled observations corresponding to that policy
    #     :return: acts: (dict) maps policy id to sampled actions corresponding to that policy
    #     :return: rewards: (dict) maps policy id to sampled rewards corresponding to that policy
    #     :return: dones: (dict) maps policy id to sampled terminal status of agents corresponding to that policy
    #     :return: dones_env: (dict) maps policy id to sampled environment terminal status corresponding to that policy
    #     :return: valid_transition: (dict) maps policy_id to whether each sampled transition is valid or not (invalid if corresponding agent is dead)
    #     :return: avail_acts: (dict) maps policy_id to available actions corresponding to that policy
    #     """
    #     inds = np.random.choice(self.__len__(), batch_size)
    #     #import pdb;pdb.set_trace()
    #     obs_batch, share_obs_batch, dones_batch, dones_env_batch, adj_batch, acts_batch, prob_acts_batch, advantages_batch, rnn_obs_batch, rnn_share_obs_batch ,returns_batch, value_preds_batch, avail_acts = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    #     for p_id in self.policy_info.keys():
    #         obs_batch[p_id], share_obs_batch[p_id], dones_batch[p_id], dones_env_batch[p_id], adj_batch[p_id], acts_batch[p_id], prob_acts_batch[p_id], advantages_batch[p_id], rnn_obs_batch[p_id], rnn_share_obs_batch[p_id], returns_batch[p_id], value_preds_batch[p_id], avail_acts[p_id] = self.policy_buffers[p_id].sample_inds(inds,data_chunk_length)

    #     return obs_batch, share_obs_batch, dones_batch, dones_env_batch, adj_batch, acts_batch, prob_acts_batch, advantages_batch, rnn_obs_batch, rnn_share_obs_batch, returns_batch, value_preds_batch, avail_acts
        
    def compute_returns(self, idx, value_normalizer=None):

        for p_id in self.policy_info.keys():
            p_value_normalizer = None if value_normalizer is None else value_normalizer[p_id]
            self.policy_buffers[p_id].compute_returns(idx, p_value_normalizer)
        return idx

    def after_update(self):
        for p_id in self.policy_info.keys():
            self.policy_buffers[p_id].after_update()

class AdjPolicyBuffer(object):
    def __init__(self, buffer_size, episode_length, num_agents, obs_space, share_obs_space, act_space,
                 use_same_share_obs, use_avail_acts, use_reward_normalization=False,max_neighbor=1,gamma=0.97,gae_lambda=0.95,hidden_size=64, is_centralized_critic_rnn=False):
        """
        Buffer class containing buffer data corresponding to a single policy.

        :param buffer_size: (int) max number of episodes to store in buffer.
        :param episode_length: (int) max length of an episode.
        :param num_agents: (int) number of agents controlled by the policy.
        :param obs_space: (gym.Space) observation space of the environment.
        :param share_obs_space: (gym.Space) centralized observation space of the environment.
        :param act_space: (gym.Space) action space of the environment.
        :use_same_share_obs: (bool) whether all agents share the same centralized observation.
        :use_avail_acts: (bool) whether to store what actions are available.
        :param use_reward_normalization: (bool) whether to use reward normalization.
        """
        self.buffer_size = buffer_size
        self.episode_length = episode_length
        self.num_agents = num_agents
        self.use_same_share_obs = use_same_share_obs
        self.use_avail_acts = use_avail_acts
        self.use_reward_normalization = use_reward_normalization
        self.filled_i = 0
        self.current_i = 0
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.hidden_size = hidden_size
        self.is_centralized_critic_rnn = is_centralized_critic_rnn
        # obs
        if obs_space.__class__.__name__ == 'Box':
            obs_shape = obs_space.shape
            share_obs_shape = share_obs_space.shape
        elif obs_space.__class__.__name__ == 'list':
            obs_shape = obs_space
            share_obs_shape = share_obs_space
        else:
            raise NotImplementedError

        self.obs = np.zeros((self.episode_length + 1, self.buffer_size,
                             self.num_agents, obs_shape[0]), dtype=np.float32)

        self.obs_neighbor_num = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, max_neighbor), dtype=np.int32)
        self.advantages = np.zeros((self.episode_length, self.buffer_size, self.num_agents, 1), dtype=np.float32)

        # if self.use_same_share_obs:
        #     self.share_obs = np.zeros((self.episode_length + 1, self.buffer_size, share_obs_shape[0]), dtype=np.float32)
        # else:
        #     self.share_obs = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, share_obs_shape[0]),
        #                               dtype=np.float32)

        self.share_obs = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, share_obs_shape[0]), dtype=np.float32)

        # action
        act_dim = np.sum(get_dim_from_space(act_space))
        self.acts = np.zeros((self.episode_length, self.buffer_size, self.num_agents, act_dim), dtype=np.float32)
        self.prob_acts = np.zeros((self.episode_length, self.buffer_size, self.num_agents, 1), dtype=np.float32)
        if self.use_avail_acts:
            self.avail_acts = np.ones((self.episode_length, self.buffer_size, self.num_agents, act_dim), dtype=np.float32)

        # rewards
        self.rewards = np.zeros((self.episode_length, self.buffer_size, self.num_agents, 1), dtype=np.float32)

        # default to done being True
        self.dones = np.ones_like(self.rewards, dtype=np.float32)
        self.dones_env = np.ones((self.episode_length, self.buffer_size, 1), dtype=np.float32)
        self.value_preds = np.zeros((self.episode_length+1,self.buffer_size, self.num_agents, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length+1,self.buffer_size, self.num_agents, 1), dtype=np.float32)
        # self.advantage = np.zeros((self.episode_length, self.buffer_size, self.num_agents, 1), dtype=np.float32)
        self.rnn_obs = np.zeros((self.episode_length + 1, self.buffer_size,self.num_agents, self.hidden_size), dtype=np.float32)

        if self.is_centralized_critic_rnn:
            rnn_share_obs_shape = (self.episode_length + 1, self.buffer_size, self.hidden_size)
        else:
            rnn_share_obs_shape = (self.episode_length + 1, self.buffer_size, self.num_agents, self.hidden_size)
        
        self.rnn_share_obs = np.zeros(rnn_share_obs_shape, dtype=np.float32)


    def __len__(self):
        return self.filled_i      

    def compute_returns(self, idx, value_normalizer=None):
        if value_normalizer is not None:
            values = value_normalizer.denormalize(self.value_preds[:, idx])
            if torch.is_tensor(values):
                values = values.detach().cpu().numpy()
        else:
            values = self.value_preds[:, idx]

        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            delta = self.rewards[step, idx] + self.gamma * values[step + 1] * (1 - self.dones[step, idx]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - self.dones[step, idx])
            self.advantages[step, idx] = gae
            self.returns[step, idx] = gae + values[step]
        
        return idx
    
    
    def insert(self, num_insert_episodes, obs, share_obs, acts, prob_acts, rewards, dones, dones_env,values=None,rnn_obs=None,rnn_share_obs=None,avail_acts=None,obs_neighbor_num=None):
        # obs: [step, episode, agent, dim]

        episode_length = acts.shape[0]
        assert episode_length == self.episode_length, ("different dimension!")
        expected_episode_shape = (self.episode_length, num_insert_episodes, self.num_agents)
        expected_time_shape = (self.episode_length + 1, num_insert_episodes, self.num_agents)

        assert obs.shape[:3] == expected_time_shape, (
            f"obs shape {obs.shape} does not match expected leading shape "
            f"{expected_time_shape}"
        )
        assert share_obs.shape[:3] == expected_time_shape, (
            f"share_obs shape {share_obs.shape} does not match expected leading shape "
            f"{expected_time_shape}"
        )
        assert obs_neighbor_num.shape[:3] == expected_time_shape, (
            f"obs_neighbor_num shape {obs_neighbor_num.shape} does not match expected "
            f"leading shape {expected_time_shape}"
        )
        assert acts.shape[:3] == expected_episode_shape, (
            f"acts shape {acts.shape} does not match expected leading shape "
            f"{expected_episode_shape}"
        )
        assert rewards.shape[:3] == expected_episode_shape, (
            f"rewards shape {rewards.shape} does not match expected leading shape "
            f"{expected_episode_shape}"
        )
        assert dones.shape[:3] == expected_episode_shape, (
            f"dones shape {dones.shape} does not match expected leading shape "
            f"{expected_episode_shape}"
        )

        if self.current_i + num_insert_episodes <= self.buffer_size:
            idx_range = np.arange(self.current_i, self.current_i + num_insert_episodes)
        else:
            num_left_episodes = self.current_i + num_insert_episodes - self.buffer_size
            idx_range = np.concatenate((np.arange(self.current_i, self.buffer_size), np.arange(num_left_episodes)))

        self.share_obs[:, idx_range] = share_obs.copy()
        
        self.obs[:, idx_range] = obs.copy()
        self.obs_neighbor_num[:, idx_range]  = obs_neighbor_num.copy()
        # self.share_obs[:, idx_range] = share_obs.copy()
        self.acts[:, idx_range] = acts.copy()
        self.rewards[:, idx_range] = rewards.copy()
        self.dones[:, idx_range] = dones.copy()
        self.dones_env[:, idx_range] = dones_env.copy()
        self.prob_acts[:, idx_range] = prob_acts.copy()
        self.value_preds[:,idx_range] = values.copy()
        self.rnn_obs[:,idx_range] =rnn_obs.copy()

        self.rnn_share_obs[:, idx_range] = rnn_share_obs.copy()
        if self.use_avail_acts and avail_acts is not None:
            self.avail_acts[:, idx_range] = avail_acts.copy()

        self.current_i = idx_range[-1] + 1
        self.filled_i = min(self.filled_i + len(idx_range), self.buffer_size)

        return idx_range

    def sample_inds(self, data_chunk_length, num_mini_batch, value_normalizer):
        if self.filled_i <= 0:
            return
        if data_chunk_length <= 0:
            raise ValueError(f"data_chunk_length must be > 0, got {data_chunk_length}")
        if self.episode_length % data_chunk_length != 0:
            raise ValueError(
                f"data_chunk_length={data_chunk_length} must divide "
                f"episode_length={self.episode_length}; otherwise RNN chunks can "
                "cross episode boundaries."
            )

        batch_size = self.episode_length * self.filled_i
        
        obs = self.obs[:-1, :self.filled_i].transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        obs_neighbor_num = self.obs_neighbor_num[:-1, :self.filled_i].transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        share_obs = self.share_obs[:-1, :self.filled_i].transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        acts = self.acts[:, :self.filled_i].transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        prob_acts = self.prob_acts[:, :self.filled_i].transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        value_preds = self.value_preds[:-1, :self.filled_i].transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        returns = self.returns[:-1, :self.filled_i].transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)

        dones = np.concatenate((np.zeros((1, self.filled_i, self.num_agents, 1)), self.dones[:, :self.filled_i]), axis=0)
        dones = dones.transpose(1, 0, 2, 3).reshape((self.filled_i, self.episode_length + 1, self.num_agents, 1))
        dones = dones[:, :-1].reshape(batch_size, self.num_agents, -1)

        dones_env = np.concatenate((np.zeros((1, self.filled_i, 1)), self.dones_env[:, :self.filled_i]), axis=0)
        dones_env = dones_env.transpose(1, 0, 2).reshape((self.filled_i, self.episode_length + 1, 1))
        dones_env = dones_env[:, :-1].reshape(batch_size, -1)

        advantages = self.advantages[:, :self.filled_i]
        active_masks = 1 - np.concatenate(
            (np.zeros((1, self.filled_i, self.num_agents, 1)), self.dones[:-1, :self.filled_i]),
            axis=0,
        )

        valid_advantages = advantages[active_masks == 1]

        if len(valid_advantages) > 0: 
            mean_adv = np.mean(valid_advantages)
            std_adv = np.std(valid_advantages)

            if std_adv > 1e-8:
                normalized_advantages = np.zeros_like(advantages)
                normalized_advantages[active_masks == 1] = (valid_advantages - mean_adv) / (std_adv + 1e-8)
                advantages = normalized_advantages
            else:
                advantages = np.zeros_like(advantages)
        else:
            print("!!! WARNING: No valid advantages in the batch. Setting all to zero. !!!")
            advantages = np.zeros_like(advantages)

        advantages = advantages.transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)


        data_chunks = batch_size // data_chunk_length
        if data_chunks < num_mini_batch:
            raise ValueError(
                f"Not enough data chunks ({data_chunks}) for num_mini_batch="
                f"{num_mini_batch}. Increase collected episodes or reduce "
                "num_mini_batch/data_chunk_length."
            )
        rand = torch.randperm(data_chunks).numpy()
        sampler = [inds for inds in np.array_split(rand, num_mini_batch) if len(inds) > 0]

        for indices in sampler:
            obs_batch = []
            share_obs_batch = []
            acts_batch = []
            prob_acts_batch = []
            returns_batch = []
            dones_batch = []
            advantages_batch = []
            value_preds_batch = []
            obs_neighbor_num_batch = []
            dones_env_batch = []

            rnn_obs_batch = []
            rnn_share_obs_batch = []

            for i in indices:
                ind = i * data_chunk_length
                
                obs_batch.append(obs[ind:ind + data_chunk_length])
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                acts_batch.append(acts[ind:ind + data_chunk_length])
                prob_acts_batch.append(prob_acts[ind:ind + data_chunk_length])
                returns_batch.append(returns[ind:ind + data_chunk_length])
                dones_batch.append(dones[ind:ind + data_chunk_length])
                advantages_batch.append(advantages[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                obs_neighbor_num_batch.append(obs_neighbor_num[ind:ind + data_chunk_length])
                dones_env_batch.append(dones_env[ind:ind + data_chunk_length])

                original_timestep = ind % self.episode_length
                original_episode_idx = ind // self.episode_length
                
                rnn_obs_batch.append(self.rnn_obs[original_timestep, original_episode_idx, :, :])

                if self.is_centralized_critic_rnn:
                    rnn_share_obs_batch.append(self.rnn_share_obs[original_timestep, original_episode_idx, :])
                else:
                    rnn_share_obs_batch.append(self.rnn_share_obs[original_timestep, original_episode_idx, :, :])


            obs_batch = np.stack(obs_batch, axis=0)
            share_obs_batch = np.stack(share_obs_batch, axis=0)
            acts_batch = np.stack(acts_batch, axis=0)
            prob_acts_batch = np.stack(prob_acts_batch, axis=0)
            returns_batch = np.stack(returns_batch, axis=0)
            dones_batch = np.stack(dones_batch, axis=0)
            advantages_batch = np.stack(advantages_batch, axis=0)
            value_preds_batch = np.stack(value_preds_batch, axis=0)
            obs_neighbor_num_batch = np.stack(obs_neighbor_num_batch, axis=0)
            dones_env_batch = np.stack(dones_env_batch, axis=0)

            rnn_obs_batch = np.stack(rnn_obs_batch, axis=0)
            rnn_share_obs_batch = np.stack(rnn_share_obs_batch, axis=0)


            yield obs_batch, obs_neighbor_num_batch, share_obs_batch, dones_batch, dones_env_batch, acts_batch, prob_acts_batch, advantages_batch, rnn_obs_batch, rnn_share_obs_batch, returns_batch, value_preds_batch, None # avail_acts


    def after_update(self):
        """
        在一次完整的训练更新后，重置所有存储数组和指针。
        """
        # --- 将所有数据存储数组的内容清零 ---
        self.obs[:] = 0
        self.share_obs[:] = 0
        self.acts[:] = 0
        self.prob_acts[:] = 0
        self.rewards[:] = 0
        self.dones[:] = 0
        self.dones_env[:] = 0
        self.value_preds[:] = 0
        self.returns[:] = 0
        self.advantages[:] = 0
        self.rnn_obs[:] = 0
        self.rnn_share_obs[:] = 0
        self.obs_neighbor_num[:] = 0
        
        if self.use_avail_acts:
            self.avail_acts[:] = 1 

        # --- 将指针重置为 0 ---
        self.current_i = 0
        self.filled_i = 0
