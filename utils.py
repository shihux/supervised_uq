import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def save_mask_prediction_example(mask, pred, iter):
	plt.imshow(pred[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_prediction.png")
	plt.imshow(mask[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_mask.png")

# Below for ged
def calc_confusion(labels, samples, class_ixs, loss_mask=None):
    """
    Compute confusion matrix for each class across the given arrays.
    Assumes classes are given in integer-valued encoding.
    :param labels: 4/5D array
    :param samples: 4/5D array
    :param class_ixs: integer or list of integers specifying the classes to evaluate
    :param loss_mask: 4/5D array
    :return: 2D array
    """
    try:
        assert labels.shape == samples.shape
    except:
        raise AssertionError('shape mismatch {} vs. {}'.format(labels.shape, samples.shape))

    if isinstance(class_ixs, int):
        num_classes = class_ixs
        class_ixs = range(class_ixs)
    elif isinstance(class_ixs, list):
        num_classes = len(class_ixs)
    else:
        raise TypeError('arg class_ixs needs to be int or list, not {}.'.format(type(class_ixs)))

    if loss_mask is None:
        shp = labels.shape
        loss_mask = np.zeros(shape=(shp[0], 1, shp[2], shp[3]))

    conf_matrix = np.zeros(shape=(num_classes, 4), dtype=np.float32)
    for i,c in enumerate(class_ixs):

        pred_ = (samples == c).astype(np.uint8)
        labels_ = (labels == c).astype(np.uint8)

        conf_matrix[i,0] = int(((pred_ != 0) * (labels_ != 0) * (loss_mask != 1)).sum()) # TP
        conf_matrix[i,1] = int(((pred_ != 0) * (labels_ == 0) * (loss_mask != 1)).sum()) # FP
        conf_matrix[i,2] = int(((pred_ == 0) * (labels_ == 0) * (loss_mask != 1)).sum()) # TN
        conf_matrix[i,3] = int(((pred_ == 0) * (labels_ != 0) * (loss_mask != 1)).sum()) # FN

    return conf_matrix


def metrics_from_conf_matrix(conf_matrix):
    """
    Calculate IoU per class from a confusion_matrix.
    :param conf_matrix: 2D array of shape (num_classes, 4)
    :return: dict holding 1D-vectors of metrics
    """
    tps = conf_matrix[:,0]
    fps = conf_matrix[:,1]
    fns = conf_matrix[:,3]

    metrics = {}
    metrics['iou'] = np.zeros_like(tps, dtype=np.float32)

    # iterate classes
    for c in range(tps.shape[0]):
        # unless both the prediction and the ground-truth is empty, calculate a finite IoU
        if tps[c] + fps[c] + fns[c] != 0:
            metrics['iou'][c] = tps[c] / (tps[c] + fps[c] + fns[c])
        else:
            metrics['iou'][c] = np.nan

    return metrics
