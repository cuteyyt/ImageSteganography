from skimage.measure import compare_psnr

import numpy as np


class AverageLoss(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val != np.nan and val != np.inf:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def cal_single_psnr(ori, target):
    ori_height = ori.shape[0]
    ori_width = ori.shape[1]
    target_height = target.shape[0]
    target_width = target.shape[1]
    if ori_width != target_width or ori_height != target_height:
        raise ValueError(
            'ori size ({}, {}) and target size ({}, {}) should be the same'.format(ori_height, ori_width, target_height,
                                                                                   target_width))

    if ori.dtype != np.uint8:
        ori = ori.astype(np.uint8)
    if target.dtype != np.uint8:
        target = target.astype(np.uint8)
    psnr = compare_psnr(ori, target, 255)
    return psnr
