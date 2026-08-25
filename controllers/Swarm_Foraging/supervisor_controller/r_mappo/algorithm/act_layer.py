import torch
import torch.nn as nn
from supervisor_controller.utils.util import init, adj_init
from supervisor_controller.utils.util import to_torch, init
from torch.distributions import Normal
import torch.nn.functional as F

from torch.distributions import Normal
import numpy as np


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_mlp_layer(input_dim, output_dim, gain=1.0):
    init_method = nn.init.orthogonal_
    def init_(m):
        return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
    return init_(nn.Linear(input_dim, output_dim))

class DiagGaussian(nn.Module):
    def __init__(self, args, act_dim, rnn_input_dim, obs_dim, device):
        super(DiagGaussian, self).__init__()
        
        expected_act_dim = act_dim
        assert act_dim == expected_act_dim, f"DiagGaussian expecting act_dim={expected_act_dim}, but got {act_dim}"

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        input_dim = rnn_input_dim + obs_dim
        
        branch_hidden_size = 64
        activation_func = nn.Tanh

        # --- 独立多头 (Multi-Heads) ---
        
        # Head A: w_food (Advection Task 1)
        self.branch_food = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.food_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.food_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)

        # Head B: w_nest (Advection Task 2)
        self.branch_nest = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.nest_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.nest_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)

        # Head C: w_rand (Diffusion)
        self.branch_rand = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.rand_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.rand_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)

        # Head : w_info (ShareInfo)
        self.branch_info = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.info_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.info_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)

        # Head D: k_diff (Interaction/Repulsion)
        self.branch_diff = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.diff_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.diff_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)

        # --- Head E: lambda_pick (Reaction 1) ---
        self.branch_pick = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.pick_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.pick_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)

        # --- Head F: lambda_drop (Reaction 2) ---
        self.branch_drop = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.drop_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.drop_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)

        self.to(device) 

    def forward(self, rnn_states, obs):
        rnn_states = rnn_states.to(**self.tpdv)
        obs = obs.to(**self.tpdv)
        x = torch.cat([rnn_states, obs], dim=-1) 
        
        # 1. w_food
        h_food = self.branch_food(x)
        mu_food = self.food_mu(h_food)
        logstd_food = self.food_logstd(h_food)

        # 2. w_nest
        h_nest = self.branch_nest(x)
        mu_nest = self.nest_mu(h_nest)
        logstd_nest = self.nest_logstd(h_nest)

        # 3. w_rand
        h_rand = self.branch_rand(x)
        mu_rand = self.rand_mu(h_rand)
        logstd_rand = self.rand_logstd(h_rand)

        # w_info
        h_info = self.branch_info(x)
        mu_info = self.info_mu(h_info)
        logstd_info = self.info_logstd(h_info)

        # 4. k_diff
        h_diff = self.branch_diff(x)
        mu_diff = self.diff_mu(h_diff)
        logstd_diff = self.diff_logstd(h_diff)

        # 5. lambda_pick 
        h_pick = self.branch_pick(x)
        mu_pick = self.pick_mu(h_pick)
        logstd_pick = self.pick_logstd(h_pick)

        # 6. lambda_drop 
        h_drop = self.branch_drop(x)
        mu_drop = self.drop_mu(h_drop)
        logstd_drop = self.drop_logstd(h_drop)

        all_mu = torch.cat([
            mu_food, mu_nest, mu_rand, mu_diff, mu_info,
            mu_pick, mu_drop
        ], dim=-1)
        
        all_logstd = torch.cat([
            logstd_food, logstd_nest, logstd_rand, logstd_diff, logstd_info,
            logstd_pick, logstd_drop
        ], dim=-1)

        all_std = torch.exp(all_logstd)

        action_dist = Normal(all_mu, all_std)
        
        return action_dist



class Categorical(nn.Module):
    def __init__(self, args, input_dim, output_dim, device):
        super(Categorical, self).__init__()
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_ReLU = args.use_ReLU
        self.use_orthogonal = args.use_orthogonal
        self.output_dim = output_dim
        active_func = [nn.Tanh(), nn.ReLU()][self.use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal]
        gain = args.gain
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0),gain=gain)

        self.layer = nn.Sequential(init_(nn.Linear(input_dim, self.output_dim)),nn.Sigmoid())

        self.to(device)

    def forward(self, x):
        x = to_torch(x).to(**self.tpdv)
        # bs = x.shape[0]

        out = self.layer(x)

        # print(F.softmax(out,dim=-1).reshape(8,6,-1))


        return out

