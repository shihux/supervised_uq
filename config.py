# The training and testing scripts share the same config file 

import os
import numpy as np
from collections import OrderedDict

config_path = os.path.realpath(__file__)

input_channels = 1
num_classes = 1
num_filters = [32,64,128,192]
latent_dim = 6
no_convs_fcomb = 4
beta = 1.0 # for kl[q(z|x,y) || p(z|x)]
beta_w = 1.0 # for kl[q(w) || p(w)]
adam_lr = 1e-6
adam_weight_decay = 0
epochs = 801
l2_reg_coeff = 1e-5
train_retention_rate = 1.
dataset_location = '/home/shu/images/lidc/'
