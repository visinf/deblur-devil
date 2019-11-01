# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

# Naive PyTorch implementation of the scale and translation optimized PSNR metric used in
# R. Köhler, M. Hirsch, B. Mohler, B. Schölkopf, and S. Harmeling,
# Recording and playback of camera shake: benchmarking blind deconvolution with a real-world database, ECCV 2012.
# http://webdav.is.mpg.de/pixel/benchmark4camerashake/

# While the scale optimization is identical, the translation optimization is not based on a FFT.
# Instead, we just compute the mse for shifted windows in a given search range, and keep the minimum error.
# On GPU, this is plenty fast enough. Subsequently, this optimized mse is converted to PSNR.
#
# NOTE: The optimized PSNR metric should not be used as a training loss.

import numpy as np
import torch
import torch.nn as nn

from contrib.colorspace import ToGrayscale


class _OptimizedMSE(nn.Module):
    """
    Scale and translation optimized MSE. Not supposed to be used as a loss function for training !
    """

    def __init__(self, grayscale=True, scale_opt=False, trans_opt=False, trans_range=5, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.grayscale = grayscale
        self.scale_opt = scale_opt
        self.trans_opt = trans_opt
        self.trans_range = trans_range
        self.trans_range = trans_range
        if grayscale:
            self.rgb2gray = ToGrayscale()

    def translation_optimized_mse(self, prediction, gt):
        h, w = prediction.size()[2:4]
        min_mse = prediction.new_ones((1,)) * np.finfo(np.float32).max
        for y0 in range(-self.trans_range, self.trans_range + 1):
            pred_ymin = max(0, y0)
            pred_ymax = min(h, h + y0)
            gt_ymin = max(0, -y0)
            gt_ymax = min(h, h - y0)
            for x0 in range(-self.trans_range, self.trans_range + 1):
                pred_xmin = max(0, x0)
                pred_xmax = min(w, w + x0)
                gt_xmin = max(0, -x0)
                gt_xmax = min(w, w - x0)
                shifted_pred = prediction[:, :, pred_ymin:pred_ymax, pred_xmin:pred_xmax]
                shifted_gt = gt[:, :, gt_ymin:gt_ymax, gt_xmin:gt_xmax]
                translated_mse = self.mse(shifted_pred, shifted_gt)
                min_mse = torch.min(min_mse, translated_mse)
        return min_mse

    @staticmethod
    def optimize_scale(prediction, gt):
        nom = (prediction * gt).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True).sum(dim=1, keepdim=True)
        denom = (gt * gt).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True).sum(dim=1, keepdim=True)
        scale_factor = nom / denom
        return scale_factor * prediction

    def forward(self, prediction, gt):  # x2 supposed to be gt !
        if self.grayscale and prediction.size(1) == 3:
            prediction, gt = self.rgb2gray(torch.cat((prediction, gt), dim=0)).chunk(chunks=2, dim=0)
        if self.scale_opt:
            prediction = self.optimize_scale(prediction, gt)
        if self.trans_opt:
            mse = self.translation_optimized_mse(prediction, gt)
        else:
            mse = self.mse(prediction, gt)
        return mse


class PSNR(nn.Module):
    def __init__(self, grayscale=True, scale_opt=False, trans_opt=False, trans_range=5, reduction='mean'):
        super().__init__()
        self.mse = _OptimizedMSE(
            grayscale=grayscale,
            scale_opt=scale_opt,
            trans_opt=trans_opt,
            trans_range=trans_range,
            reduction=reduction)

    def forward(self, prediction, gt):
        mse = self.mse(prediction, gt)
        psnr = - 10 * torch.log10(mse)
        return psnr
