import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
from utils.others import *

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os
import pandas as pd


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


def dice_loss(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 1.
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
def dice_coeff3(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
class SoftDiceLoss(nn.Module):
    """
    这个与前面的dice_loss差不多
    """
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        num = targets.size(0)  # Number of batches

        score = dice_coeff3(probs, targets)
        score = 1 - score.sum() / num
        return score



"""

以下都是attunet作者的

"""
def _fast_hist(label_true, label_pred, n_class):
    """

    :param label_true:
    :param label_pred:
    :param n_class:
    :return:
    """
    # print('label_true:\n{}'.format(label_true))
    # print('label_pred:\n{}'.format(label_pred))
    mask = (label_true >= 0) & (label_true < n_class)
    a = label_true[mask].astype(int)
    b = label_pred[mask].astype(int)
    c = n_class * a + b
    # print('a:\n{}'.format(a))
    # print('b:\n{}'.format(b))
    # print('c:\n{}'.format(c))
    hist = np.bincount(c, minlength=n_class**2)
    # print('hist1:\n{}'.format(hist))
    hist = hist.reshape(n_class, n_class)
    # print('hist2:\n{}'.format(hist))
    return hist

def segmentation_scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - 交集：np.diag取hist的对角线元素
      - 并集：hist.sum(1)和hist.sum(0)分别按两个维度相加，而对角线元素加了两次，因此减一次

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    # print(hist)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return {'overall_acc': acc,
            'mean_acc': acc_cls,
            'freq_w_acc': fwavacc,
            'mean_iou': mean_iu}

def dice_score_list(label_gt, label_pred, n_class):
    """

    :param label_gt: [WxH] (2D images)
    :param label_pred: [WxH] (2D images)
    :param n_class: number of label classes
    :return:
    """
    epsilon = 1.0e-6
    assert len(label_gt) == len(label_pred)
    batchSize = len(label_gt)
    dice_scores = np.zeros((batchSize, n_class), dtype=np.float32)
    for batch_id, (l_gt, l_pred) in enumerate(zip(label_gt, label_pred)):
        for class_id in range(n_class):
            img_A = np.array(l_gt == class_id, dtype=np.float32).flatten()
            img_B = np.array(l_pred == class_id, dtype=np.float32).flatten()
            score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
            dice_scores[batch_id, class_id] = score

    return np.mean(dice_scores, axis=0)

def segmentation_stats(pred_seg, target):
    n_classes = pred_seg.size(1)
    n_classes = n_classes + 1
    # print('n_classes:{}'.format(n_classes))
    # pred_lbls = pred_seg.data.max(1)[1].cpu().numpy().astype(np.int16)
    pred_lbls = pred_seg.data.max(1)[0].cpu().numpy()
    gt = np.squeeze(target.data.cpu().numpy(), axis=1)
    gts, preds = [], []
    for gt_, pred_ in zip(gt, pred_lbls):
        gts.append(gt_)
        preds.append(pred_)
    # print('preds:\n{}'.format(preds))
    # print('gts:\n{}'.format(gts))
    iou = segmentation_scores(gts, preds, n_class=n_classes)
    dice = dice_score_list(gts, preds, n_class=n_classes)
    return iou, dice

def get_segmentation_stats(prediction, target):
    seg_scores, dice_score = segmentation_stats(prediction, target)
    seg_stats = [('Overall_Acc', seg_scores['overall_acc']), ('Mean_IOU', seg_scores['mean_iou'])]
    for class_id in range(dice_score.size):
        seg_stats.append(('Class_{}'.format(class_id), dice_score[class_id]))
    return OrderedDict(seg_stats)

"""
CENet's dice_coeff
"""
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)     # y_true里面的全部元素相加
            j = torch.sum(y_pred)
            # print('i:{}'.format(i))
            # print('j:{}'.format(j))
            intersection = torch.sum(y_true * y_pred)   # 两者相乘之后，再全部元素
            # print('intersection:{}'.format(intersection))
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # print('score_mean:{}'.format(score.mean()))
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return b


'''
if __name__ == '__main__':
    pred = torch.Tensor([[[[1, 0, 1],
                           [1, 0, 0],
                           [0, 0, 1]]]])
    gt = torch.Tensor([[[[1, 1, 1],
                         [0, 1, 0],
                         [1, 0, 1]]]])
    # print('gt:\n'.format(gt))
    # print('pred:\n'.format(pred))

    # test iou
    iou, dice = segmentation_stats(gt, pred)
    # print('iou:{}'.format(iou))
    # print('dice:{}'.format(dice))

    # test dice
    loss = dice_bce_loss()
    dice_bce = loss.soft_dice_coeff(gt, pred)
    # print('dice_bce:\n{}'.format(dice_bce))
'''



def log_rmse(gt, pred):

    a, b = segmentation_stats(gt, pred)
    mean_iou = a['mean_iou']
    mean_acc = a['mean_acc']
    dic_coeff = dice_bce_loss()
    b = dic_coeff.soft_dice_coeff(gt, pred)
    dic = b.cpu().item()
    # print("mean_iou:\n{}".format(mean_iou))
    # print("mean_acc:\n{}".format(mean_acc))
    # print("dice coeff:\n{}".format(dic))
    return mean_iou, mean_acc, dic

