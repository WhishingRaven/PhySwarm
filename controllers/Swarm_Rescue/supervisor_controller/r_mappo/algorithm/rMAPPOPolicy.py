import numpy as np
import torch
import time
import torch.nn.functional as F
from supervisor_controller.r_mappo.algorithm.act_layer import DiagGaussian, Categorical
from supervisor_controller.r_mappo.algorithm.v_out import VFunction
from supervisor_controller.utils.util import get_dim_from_space, is_discrete, is_multidiscrete, make_onehot, DecayThenFlatSchedule, avail_choose, to_torch, to_numpy
from supervisor_controller.r_mappo.base.mlp_policy import MLPPolicy
from supervisor_controller.r_mappo.algorithm.rnn import RNNBase
from supervisor_controller.r_mappo.algorithm.critic_rnn import critic_RNNBase
from torch.distributions import Normal, MultivariateNormal


class R_MAPPOPolicy(MLPPolicy):
    """
    QMIX/VDN Policy Class to compute Q-values and actions (MLP). See parent class for details.
    :param config: (dict) contains information about hyperparameters and algorithm configuration
    :param policy_config: (dict) contains information specific to the policy (obs dim, act dim, etc)
    :param train: (bool) whether the policy will be trained.
    """
    def __init__(self, config, policy_config, train=True):
        self.args = config["args"]
        self.device = config['device']
        self.obs_space = policy_config["obs_space"]
        self.n_agents = config["num_agents"]
        # self.num_factor = self.args.num_factor
        self.num_target = self.args.num_target
        self.obs_dim = get_dim_from_space(self.obs_space)
        self.act_space = policy_config["act_space"]
        self.act_dim = get_dim_from_space(self.act_space)
        self.output_dim = sum(self.act_dim) if isinstance(self.act_dim, np.ndarray) else self.act_dim
        self.central_obs_dim = policy_config["cent_obs_dim"]
        self.discrete_action = is_discrete(self.act_space)
        self._debug_print_once = True
        self.num_agents = self.args.num_agents

        #import pdb;pdb.set_trace()
        self.hidden_size = self.args.hidden_size
        self.lamda = self.args.lamda
        self.num_rank = self.args.num_rank
        if self.args.prev_act_inp:
            # this is only local information so the agent can act decentralized
            self.rnn_network_input_dim = self.obs_dim + self.act_dim
        else:
            self.rnn_network_input_dim = self.obs_dim
        self.rnn_out_dim = self.args.hidden_size
        self.rnn_hidden_size = self.args.hidden_size
        self.act_input_dim = self.rnn_out_dim
        self.highest_orders = self.args.highest_orders
        
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.use_vfunction = self.args.use_vfunction
        # Local recurrent q network for the agent
        self.rnn_network = RNNBase(self.args, self.rnn_network_input_dim, self.rnn_hidden_size, self.rnn_out_dim, self.device)
        if self.discrete_action:
            self.act = Categorical(self.args, self.act_input_dim, self.act_dim, self.device)
        else:
            # self.act = DiagGaussian(self.args, self.act_dim, self.act_input_dim, self.act_dim, self.device)
            self.act = DiagGaussian(self.args, self.act_dim, self.act_input_dim, self.obs_dim, self.device)
        if self.use_vfunction:
            
            self.rnn_critic_network = critic_RNNBase(self.args, 
                                                     input_shape=self.obs_dim * self.num_agents, 
                                                     hidden_size=self.rnn_hidden_size,
                                                     out_shape=self.rnn_out_dim, 
                                                     device=self.device)
            self.v_network = VFunction(self.args, self.rnn_out_dim, self.rnn_hidden_size, 1, self.device)
        
    def get_hidden_states(self, obs, prev_actions, rnn_states, obs_neighbor_num):
        if self.args.prev_act_inp:
            prev_action_batch = to_torch(prev_actions)
            input_batch = torch.cat((obs, prev_action_batch), dim=-1)
        else:
            input_batch = to_torch(obs)
        
        q_batch, new_rnn_states, no_sequence = self.rnn_network(input_batch.to(**self.tpdv), to_torch(rnn_states).to(**self.tpdv), to_torch(obs_neighbor_num).to(self.device))
        #import pdb;pdb.set_trace()
        return q_batch,new_rnn_states,no_sequence

    def get_hidden_critic(self, obs, rnn_states):
        """
        处理结构化的全局状态 obs，并传递给 critic_RNNBase。
        
        Args:
            obs (torch.Tensor): 结构化的全局状态, 形状应为 
                                [batch_size, num_agents, agent_feature_dim] 或
                                [seq_len * batch_size, num_agents, agent_feature_dim]
            rnn_states (torch.Tensor): critic 的隐状态。
        """
        input_batch = to_torch(obs).to(**self.tpdv)
        rnn_states_torch = to_torch(rnn_states).to(**self.tpdv)
        q_batch, new_rnn_states, no_sequence = self.rnn_critic_network(input_batch, rnn_states_torch)
        
        return q_batch, new_rnn_states, no_sequence

    def get_v_value(self, batch_size, obs_batch):
        values = self.v_network(obs_batch)
        return values

    def get_actions(self, actor_hidden_states, critic_hidden_states, obs, explore=True):
        action_distribution = self.act(actor_hidden_states, obs)

        if explore:
            actions = action_distribution.sample()
        else:
            actions = action_distribution.mean

        actions_log_probs = action_distribution.log_prob(actions).sum(axis=-1, keepdim=True)

        values = self.v_network(critic_hidden_states)
        return actions, actions_log_probs, values
    
    def evaluate_actions(self, actor_hidden_states, action_batch, obs, old_log_probs_for_debug=None):
        action_distribution = self.act(actor_hidden_states, obs)

        action_log_probs = action_distribution.log_prob(action_batch).sum(axis=-1, keepdim=True)
        
        dist_entropy = action_distribution.entropy().sum(axis=-1).mean()

        return action_log_probs, dist_entropy, action_distribution

    def get_random_actions(self, obs, available_actions):
        return None

    def init_hidden(self, num_agents, batch_size):
        if num_agents == -1:
            return torch.zeros(batch_size, self.hidden_size)
        else:
            return torch.zeros(batch_size*num_agents, self.hidden_size)

    def actor_parameters(self):
        parameters_sum = []
        parameters_sum += self.rnn_network.parameters()
        parameters_sum += self.act.parameters()

        return parameters_sum

    def critic_parameters(self):
        parameters_sum = []

        if self.use_vfunction:
            parameters_sum += self.rnn_critic_network.parameters()
            parameters_sum += self.v_network.parameters()
        return parameters_sum

    def load_state(self, source_policy):
        self.rnn_network.load_state_dict(source_policy.rnn_network.state_dict())
        self.act.load_state_dict(source_policy.act.state_dict())
        if self.use_vfunction:
            self.rnn_critic_network.load_state_dict(source_policy.rnn_critic_network.state_dict())
            self.v_network.load_state_dict(source_policy.v_network.state_dict())
