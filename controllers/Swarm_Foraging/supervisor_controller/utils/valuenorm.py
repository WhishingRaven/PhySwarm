import torch
import torch.nn as nn
import numpy as np

class ValueNorm(nn.Module):
    def __init__(self, input_shape, norm_axes=1, beta=0.995, per_element_update=False, epsilon=1e-8):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update

        self.register_buffer('running_mean', torch.zeros(input_shape))
        self.register_buffer('running_mean_sq', torch.zeros(input_shape))
        self.register_buffer('debiasing_term', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0.0)) 

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()
        self.count.zero_()

    @torch.no_grad() 
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector).float()
        input_vector = input_vector.to(self.running_mean.device)

        batch_mean = input_vector.mean() 
        batch_sq_mean = (input_vector ** 2).mean()

        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[:self.norm_axes])
            weight = self.beta ** batch_size
        else:
            weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))
        self.count += 1
    
    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = debiased_mean_sq - debiased_mean ** 2
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector).float()
        input_vector = input_vector.to(self.running_mean.device)

        mean, var = self.running_mean_var()
        std = torch.sqrt(var.clamp(min=self.epsilon))
        
        return (input_vector - mean[(None,) * self.norm_axes]) / std[(None,) * self.norm_axes]

    def denormalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector).float()
        input_vector = input_vector.to(self.running_mean.device)

        mean, var = self.running_mean_var()
        std = torch.sqrt(var.clamp(min=self.epsilon))
        
        out = input_vector * std[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        
        return out.cpu().numpy()