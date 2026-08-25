import torch
import torch.nn as nn
from supervisor_controller.utils.util import init, adj_init
from supervisor_controller.utils.util import to_torch, init
from torch.distributions import Normal
import torch.nn.functional as F
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
        
        # [w_target, w_center, w_rand, k_diff, lambda_anchor, lambda_release]
        expected_act_dim = 6 
        assert act_dim == expected_act_dim, f"DiagGaussian expecting act_dim={expected_act_dim}, but got {act_dim}"

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        # --- 超参数配置 ---
        input_dim = rnn_input_dim + obs_dim
        branch_hidden_size = 64
        activation_func = nn.Tanh

        # ==========================================================
        # 独立多头 (Multi-Heads)
        # ==========================================================
        
        # Head 1: w_target (指向救援目标的对流强度)
        self.branch_target = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.target_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.target_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)

        # Head 2: w_center (指向合成驻波中点的对流强度)
        self.branch_center = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.center_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.center_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)

        # Head 3: w_rand (随机搜索/覆盖探索的强度)
        self.branch_rand = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.rand_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.rand_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)

        # Head 4: k_diff (扩散项：负责避障与中继链的拓扑撑开)
        self.branch_diff = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.diff_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.diff_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)

        # Head 5: lambda_anchor (反应项：由 Responder 切换为 Relay 的概率)
        # 对应觅食中的 lambda_pick
        self.branch_anchor = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.anchor_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.anchor_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)

        # Head 6: lambda_release (反应项：由 Relay 切换回 Responder 的概率)
        # 对应觅食中的 lambda_drop
        self.branch_release = nn.Sequential(
            get_mlp_layer(input_dim, branch_hidden_size, gain=np.sqrt(2)),
            activation_func()
        )
        self.release_mu = get_mlp_layer(branch_hidden_size, 1, gain=0.01)
        self.release_logstd = get_mlp_layer(branch_hidden_size, 1, gain=0.01)

        self.to(device) 

    def forward(self, rnn_states, obs):
        rnn_states = rnn_states.to(**self.tpdv)
        obs = obs.to(**self.tpdv)
        
        x = torch.cat([rnn_states, obs], dim=-1) 
        
        # 1. 计算 w_target
        h_target = self.branch_target(x)
        mu_target = self.target_mu(h_target)
        logstd_target = self.target_logstd(h_target)

        # 2. 计算 w_center
        h_center = self.branch_center(x)
        mu_center = self.center_mu(h_center)
        logstd_center = self.center_logstd(h_center)

        # 3. 计算 w_rand
        h_rand = self.branch_rand(x)
        mu_rand = self.rand_mu(h_rand)
        logstd_rand = self.rand_logstd(h_rand)

        # 4. 计算 k_diff
        h_diff = self.branch_diff(x)
        mu_diff = self.diff_mu(h_diff)
        logstd_diff = self.diff_logstd(h_diff)

        # 5. 计算 lambda_anchor
        h_anchor = self.branch_anchor(x)
        mu_anchor = self.anchor_mu(h_anchor)
        logstd_anchor = self.anchor_logstd(h_anchor)

        # 6. 计算 lambda_release
        h_release = self.branch_release(x)
        mu_release = self.release_mu(h_release)
        logstd_release = self.release_logstd(h_release)

        all_mu = torch.cat([
            mu_target, mu_center, mu_rand, mu_diff, 
            mu_anchor, mu_release
        ], dim=-1)
        
        all_logstd = torch.cat([
            logstd_target, logstd_center, logstd_rand, logstd_diff, 
            logstd_anchor, logstd_release
        ], dim=-1)

        all_std = torch.exp(all_logstd)

        # 构造多维正态分布
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

