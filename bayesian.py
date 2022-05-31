'''
    This code is based on https://arxiv.org/abs/1701.05369
'''
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Linear layers
class GaussianLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, 
                 local_reparameterization=True, log_sigma_init=-5.0, sample_on_eval=False, deterministic=False):
        self.log_sigma = None
        self.log_sigma_init = log_sigma_init
        self.local_reparameterization = local_reparameterization
        self.sample_on_eval = sample_on_eval 
        self.threshold = None
        self.log_sigma_clamp = 1.0
        self.deterministic = deterministic
        
        super(GaussianLinear, self).__init__(in_features, out_features, bias)
        
        self.log_sigma = nn.Parameter(torch.empty_like(self.weight))
        self._init_sigma()

    def reset_parameters(self):
        super(GaussianLinear, self).reset_parameters()
        self._init_sigma()
            
    def _init_sigma(self):
        if self.log_sigma is not None:
            self.log_sigma.data.normal_(self.log_sigma_init, 0.1)
        
    def forward(self, input):
        weight, log_sigma = self._threshold_weights()

        if (not self.training and not self.sample_on_eval) or self.deterministic:
            return F.linear(input, weight, self.bias)
        
        if self.local_reparameterization: 
            gamma = F.linear(input, weight, self.bias)
            delta = F.linear(input ** 2, torch.exp(2 * log_sigma))

            distr = Normal(gamma, torch.sqrt(torch.abs(delta) + 1e-8))
            return distr.rsample()
        else:
            distr = Normal(weight, torch.exp(log_sigma))
            sampled_weights = distr.rsample()

            return F.linear(input, sampled_weights, self.bias)

    def _threshold_weights(self):
        if self.log_sigma_clamp is not None:
            log_sigma = torch.clamp(self.log_sigma, max=self.log_sigma_clamp)
        else:
            log_sigma = self.log_sigma

        if self.threshold is not None:
            mask = (torch.abs(self.weight) > self.threshold).type_as(self.weight)
            return mask * self.weight, mask * log_sigma + (1 - mask) * (-20)

        return self.weight, log_sigma

# Convolution layers
class GaussianConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 local_reparameterization=True, log_sigma_init=-5.0, sample_on_eval=False, deterministic=False):
        self.log_sigma = None
        self.log_sigma_init = log_sigma_init
        self.local_reparameterization = local_reparameterization
        self.sample_on_eval = sample_on_eval 
        self.threshold = None
        self.log_sigma_clamp = 1.0
        self.deterministic = deterministic 
        
        super(GaussianConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                             stride, padding, dilation, groups, bias)
        
        self.log_sigma = nn.Parameter(torch.empty_like(self.weight)) 
        self._init_sigma()
            
    def _init_sigma(self):
        if self.log_sigma is not None:
            self.log_sigma.data.normal_(self.log_sigma_init, 0.1)

    def reset_parameters(self):
        super(GaussianConv2d, self).reset_parameters()
        self._init_sigma()
        
    def _conv2d(self, input, weight, bias=None):
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def forward(self, input):
        weight, log_sigma = self._threshold_weights()

        if (not self.training and not self.sample_on_eval) or self.deterministic:
            return self._conv2d(input, weight, self.bias)
        
        if self.local_reparameterization: 
            gamma = self._conv2d(input, weight, self.bias)
            delta = self._conv2d(input ** 2, torch.exp(2 * log_sigma))
            distr = Normal(gamma, torch.sqrt(torch.abs(delta) + 1e-8))
            return distr.rsample()
        else:
            distr = Normal(weight, torch.exp(log_sigma))
            sampled_weights = distr.rsample()
            return self._conv2d(input, sampled_weights, self.bias)

    def _threshold_weights(self):
        if self.log_sigma_clamp is not None:
            log_sigma = torch.clamp(self.log_sigma, max=self.log_sigma_clamp)
        else:
            log_sigma = self.log_sigma

        if self.threshold is not None:
            mask = (torch.abs(self.weight) > self.threshold).type_as(self.weight)
            return mask * self.weight, mask * log_sigma + (1 - mask) * (-20) 

        return self.weight, log_sigma

