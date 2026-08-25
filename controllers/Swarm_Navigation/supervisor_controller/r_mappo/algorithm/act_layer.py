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

        # --- 超参数配置 ---
        input_dim = rnn_input_dim + obs_dim
        
        branch_hidden_size = 64
        activation_func = nn.Tanh
        self.logstd_min = np.log(0.08)
        self.logstd_max = np.log(0.35)
        self.logstd_bias = np.log(0.25)

        # ==========================================================
        # 独立多头 (Multi-Heads) 
        # ==========================================================
        
        # --- Head 1: w_flow (全局平流推进权重) ---
        self.branch_flow = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.flow_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.flow_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        nn.init.constant_(self.flow_logstd.bias, self.logstd_bias)

        # --- Head 2: w_shape (形态保持/向心收缩权重) ---
        self.branch_shape = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.shape_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.shape_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        nn.init.constant_(self.shape_logstd.bias, self.logstd_bias)

        # --- Head 3: k_diff (防撞排斥反应系数) ---
        self.branch_diff = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.diff_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.diff_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        nn.init.constant_(self.diff_logstd.bias, self.logstd_bias)

        # --- Head 4: beta (势场 Y 轴形变系数) ---
        self.branch_beta = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.beta_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.beta_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        nn.init.constant_(self.beta_logstd.bias, self.logstd_bias)

        self.to(device)

    def forward(self, rnn_states, obs):
        rnn_states = torch.nan_to_num(
            rnn_states.to(**self.tpdv),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        obs = torch.nan_to_num(
            obs.to(**self.tpdv),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        x = torch.cat([rnn_states, obs], dim=-1)
        x = torch.clamp(x, -10.0, 10.0)
        
        # 1. w_flow
        h_flow = self.branch_flow(x)
        mu_flow = self.flow_mu(h_flow)
        logstd_flow = self.flow_logstd(h_flow)

        # 2. w_shape
        h_shape = self.branch_shape(x)
        mu_shape = self.shape_mu(h_shape)
        logstd_shape = self.shape_logstd(h_shape)

        # 3. k_diff
        h_diff = self.branch_diff(x)
        mu_diff = self.diff_mu(h_diff)
        logstd_diff = self.diff_logstd(h_diff)

        # 4. beta
        h_beta = self.branch_beta(x)
        mu_beta = self.beta_mu(h_beta)
        logstd_beta = self.beta_logstd(h_beta)

        all_mu = torch.cat([
            mu_flow, mu_shape, mu_diff, mu_beta
        ], dim=-1)
        
        all_logstd = torch.cat([
            logstd_flow, logstd_shape, logstd_diff, logstd_beta
        ], dim=-1)

        all_mu = torch.nan_to_num(all_mu, nan=0.0, posinf=5.0, neginf=-5.0)
        all_mu = torch.clamp(all_mu, -5.0, 5.0)
        all_logstd = torch.nan_to_num(
            all_logstd,
            nan=self.logstd_bias,
            posinf=self.logstd_max,
            neginf=self.logstd_min,
        )
        all_logstd = torch.clamp(all_logstd, self.logstd_min, self.logstd_max)
        all_std = torch.exp(all_logstd)

        action_dist = Normal(all_mu, all_std, validate_args=False)
        
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
