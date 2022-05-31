import numpy as np
import torch

def iou_distance(outputs, labels):
    smooth = 1e-10
    outputs = outputs.squeeze(1)
    labels = labels.squeeze(1)
    outputs = outputs.type(torch.IntTensor) 
    labels = labels.type(torch.IntTensor) 

    #If both mask and prediction are empty (no lesion) return distance 0
    if torch.sum(labels) == 0 and torch.sum(outputs) == 0:
        return 0
    else:
        intersection = (outputs & labels).float().sum()
        union = (outputs | labels).float().sum()
        iou = (intersection + smooth) / (union + smooth)
        return 1-iou.item()

def generalized_energy_distance_iou(predictions, masks):
    m = masks.shape[0]
    num_preds = predictions.shape[0]

    iou_dist = 0
    for i in range(num_preds):
        for j in range(m):
            iou_dist += iou_distance(predictions[i], masks[j])

    iou_dist_pred = 0
    for i in range(num_preds):
        for j in range(num_preds):
            iou_dist_pred += iou_distance(predictions[i], predictions[j])

    iou_dist_mask = 0
    for i in range(m):
        for j in range(m):
            iou_dist_mask += iou_distance(masks[i], masks[j])

    return (2/(m*num_preds)*iou_dist) - \
           (1/(num_preds**2)*(iou_dist_pred)) - \
           (1/(m**2)*iou_dist_mask)

# Assuming batch size of data_loader is 1
def ged(net, data_loader, num_preds):
    print('num ged samples', num_preds)

    iou_score = 0
    total_steps = 500
    net.unet._sample_on_eval(True)
    for step, (patch, masks, _) in enumerate(data_loader): 
        masks = torch.squeeze(masks,0)
        patch = patch.cuda()
        net.forward(patch, segm=None, training=False)
        predictions = []

        for i in range(num_preds):
            mask_pred = net.sample(patch, None, testing=True)
            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
            mask_pred = torch.squeeze(mask_pred, 0)
            predictions.append(mask_pred)

        predictions = torch.cat(predictions, 0)
        patch = torch.squeeze(patch, 0)
        ged_score = generalized_energy_distance_iou(predictions, masks)
        iou_score += ged_score

    ged = iou_score/len(data_loader)
    print('num samples ' + str(num_preds) + ', ged ' + str(round(ged,6)))
