# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import torch
from torch import nn

from contrib.koehler_psnr import PSNR
from losses import factory


class DBNLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mse = nn.MSELoss(reduction='sum')
        self.train_psnr = PSNR(scale_opt=False, trans_opt=False)  # No optimizations in training
        self.valid_psnr = PSNR(scale_opt=True, trans_opt=True)

    def forward(self, model_dict, example_dict):
        output = model_dict["output1"]
        target = example_dict["target1"]

        with torch.no_grad():
            if self.training:
                psnr = self.train_psnr(output.clamp(0, 1), target)
            else:
                psnr = self.valid_psnr(output, target)

        total_loss = 0.5 * self.mse(output, target)

        return {"total_loss": total_loss, "psnr": psnr}


factory.register("DBNLoss", DBNLoss)
