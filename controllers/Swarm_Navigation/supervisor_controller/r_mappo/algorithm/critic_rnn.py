import torch.nn as nn
import torch
from supervisor_controller.utils.util import init, adj_init

class critic_RNNBase(nn.Module):
    """
    修改后的 Critic Base 网络。
    它现在接收形状为 [时间步, 批量大小, 智能体数量, 特征维度] 的结构化全局状态，
    并通过一个聚合层来处理它，然后再送入 RNN。
    """
    def __init__(self, args, input_shape, hidden_size, out_shape, device=torch.device("cuda:0")):
        nn.Module.__init__(self)
        self.args = args
        self.use_ReLU = self.args.use_ReLU
        self.tpdv = dict(dtype=torch.float32, device=device) 
        self.use_orthogonal = self.args.use_orthogonal
        self._use_feature_normalization = args.use_feature_normalization
        self.hidden_size = hidden_size
        self.device = device
        
        self.agent_feature_dim = input_shape 

        active_func = [nn.Tanh(), nn.ReLU()][self.use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][self.use_ReLU])
        
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.mlp = nn.Sequential(
            init_(nn.Linear(self.agent_feature_dim, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
            init_(nn.Linear(hidden_size, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size)
        )
        
        self.rnn = nn.GRU(hidden_size, out_shape)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self.use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
                    
        self.rnn_norm = nn.LayerNorm(out_shape)
        
        self.to(device) 

    
    def forward(self, inputs, rnn_states):

        original_dim = inputs.dim()
        if original_dim == 3:
            inputs = inputs.unsqueeze(0)

        seq_len, batch_size, num_agents, feats = inputs.shape
      
        x = inputs.reshape(-1, feats) 
        x = self.mlp(x)
        x = x.reshape(seq_len, -1, self.hidden_size) 
        
        rnn_states = rnn_states.reshape(-1, self.hidden_size)
        rnn_states = rnn_states.unsqueeze(0)

        self.rnn.flatten_parameters()
        
        x, hid = self.rnn(x, rnn_states)
        x = self.rnn_norm(x)

        if original_dim == 3:
            x = x.squeeze(0) 

        return x, hid[0], False

  

  

# import torch.nn as nn
# import torch
# from supervisor_controller.utils.util import init, adj_init

# class critic_RNNBase(nn.Module):
#     """ Identical to rnn_agent, but does not compute value/probability for each action, only the hidden state. """
#     def __init__(self, args, input_shape, hidden_size, out_shape, device=torch.device("cuda:0")):
#         nn.Module.__init__(self)
#         self.args = args
#         self.use_ReLU = self.args.use_ReLU
#         self.tpdv = dict(dtype=torch.float16, device=device)
#         self.use_orthogonal = self.args.use_orthogonal
#         self._use_feature_normalization = args.use_feature_normalization
#         self.agent_loc_dim = self.args.agent_loc_dim
#         self.mutual_dim = self.args.mutual_dim
#         self.num_obs_targets = self.args.num_obs_targets
#         self.num_obs_agents = self.args.num_obs_agents
#         self.hidden_size = hidden_size
#         self.device= device
#         active_func = [nn.Tanh(), nn.ReLU()][self.use_ReLU]
#         init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal]
#         gain = nn.init.calculate_gain(['tanh', 'relu'][self.use_ReLU])
#         def init_(m):
#             return init(m, init_method, lambda x: nn.init.constant_(x, 0),gain=gain)
#         # self.fc1 = nn.Sequential(init_(nn.Linear(self.mutual_dim, hidden_size-self.agent_loc_dim)), active_func, nn.LayerNorm(hidden_size-self.agent_loc_dim))
#         self.fc1 = nn.Sequential(init_(nn.Linear(input_shape, hidden_size)), active_func, nn.LayerNorm(hidden_size))
#         self.fc2 = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
#         self.rnn = nn.GRU(hidden_size, out_shape)
#         for name, param in self.rnn.named_parameters():
#             if 'bias' in name:
#                 nn.init.constant_(param, 0)
#             elif 'weight' in name:
#                 if self.use_orthogonal:
#                     nn.init.orthogonal_(param)
#                 else:
#                     nn.init.xavier_uniform_(param)
#         self.norm = nn.LayerNorm(input_shape)
#         self.rnn_norm = nn.LayerNorm(out_shape)
#        # self.norm = nn.LayerNorm(out_shape)
#         self.to(device) 

#     def forward(self, inputs, rnn_states):

#         # bs = inputs.shape[0]
#         # x_loc = inputs[:,:self.agent_loc_dim]
#         # x_corr = self.fc1(inputs[:,self.agent_loc_dim:].reshape(-1,self.mutual_dim))
#         # x_corr[obs_neighbor_num.reshape(-1)==0] = 0
#         # obs_num = torch.where(obs_neighbor_num.sum(-1)==0,1,obs_neighbor_num.sum(-1))
#         # mean_corr = x_corr.reshape(bs,self.num_obs_targets+self.num_obs_agents,-1).sum(1) / obs_num.unsqueeze(-1)
#         # x_nn = torch.cat((x_loc,mean_corr),dim=1)
#         no_sequence = False
#         if len(inputs.shape) == 2:
#             inputs = inputs[None]
#         if len(rnn_states.shape) == 2:
#             rnn_states = rnn_states[None]
#         # if self._use_feature_normalization:
#         #     inputs = self.norm(inputs)
#         x = self.fc1(inputs)
#         x = self.fc2(x)
#         self.rnn.flatten_parameters()
#         x, hid = self.rnn(x, rnn_states)
#         x = self.rnn_norm(x)
#         #import pdb;pdb.set_trace()
#         return x, hid[0,:,:], no_sequence
