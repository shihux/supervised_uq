import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet
from importlib.machinery import SourceFileLoader
from ged import ged

parser = argparse.ArgumentParser(description='Supervised uncertainty quantification')
parser.add_argument('--random_seed', type=int, default=123, help='random seed')
parser.add_argument('--ckpt_filename', type=str, help='checkpoint file')
opt = parser.parse_args()
print(opt)

config_filename = 'config.py'
cf = SourceFileLoader('cf', config_filename).load_module()

np.random.seed(opt.random_seed) # NOTE seed needs to be the same in training due to data splitting

torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed(opt.random_seed)
torch.backends.cudnn.deterministic=True

# used in DataLoader
def _init_fn(worker_id):
    np.random.seed(opt.random_seed + worker_id)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location=cf.dataset_location)

dataset_size = len(dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)

split_1 = int(0.7 * dataset_size)
split_2 = int(0.85 * dataset_size)
train_indices = indices[:split_1]
val_indices = indices[split_1:split_2]
test_indices = indices[split_2:]

train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler)

val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler, worker_init_fn=_init_fn)

test_sampler = SubsetRandomSampler(test_indices)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler, worker_init_fn=_init_fn)

print("Number of train/val/test patches:", (len(train_indices), len(val_indices), len(test_indices)))


def load_net(cf, ckpt_filename):
    net = ProbabilisticUnet(input_channels=cf.input_channels, num_classes=cf.num_classes, num_filters=cf.num_filters, latent_dim=cf.latent_dim, no_convs_fcomb=cf.no_convs_fcomb, beta=cf.beta)
    net.to(device)

    net.load_state_dict(torch.load(ckpt_filename))
    net.eval() 
    net.unet._sample_on_eval(True)
    return net

def eval(cf):
    net = load_net(cf, opt.ckpt_filename)
    data_loader = test_loader

    # ged value seems to converge after 50 samples
    ged(net, test_loader, 50)


if __name__ == "__main__":
    model = eval(cf)
