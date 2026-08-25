import torch
import torch.nn as nn
from supervisor_controller.utils.util import init, adj_init
from supervisor_controller.utils.util import to_torch
from supervisor_controller.utils.popart import PopArt

class VFunction(nn.Module):
    """
    Individual agent q network (MLP).
    :param args: (namespace) contains information about hyperparameters and algorithm configuration
    :param input_dim: (int) dimension of input to q network
    :param act_dim: (int) dimension of the action space
    :param device: (torch.Device) torch device on which to do computations
    """
    def __init__(self, args, input_dim, hidden_dim, act_dim, device):
        super(VFunction, self).__init__()
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_ReLU = args.use_ReLU
        self.use_orthogonal = args.use_orthogonal
        self.use_popart = args.use_popart
        active_func = [nn.Tanh(), nn.ReLU()][self.use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        if self.use_popart:
            self.output_layer = init_(PopArt(input_dim, act_dim, self.device))
        else:
            self.output_layer = init_(nn.Linear(input_dim,act_dim))
        self.to(device) 

    def forward(self, x):
        """
        Compute q values for every action given observations and rnn states.
        :param x: (torch.Tensor) observations from which to compute q values.

        :return q_outs: (torch.Tensor) q values for every action
        """
        # make sure input is a torch tensor
        x = to_torch(x).to(**self.tpdv)

        # pass outputs through linear layer
        v_value = self.output_layer(x)


        return v_value
