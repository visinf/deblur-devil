# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import logging
import os
import warnings

from torchvision.utils import save_image

from utils import system
from visualizers import factory
from visualizers.visualizer import Visualizer


class GoProInference(Visualizer):
    def __init__(self,
                 args,
                 model_and_loss,
                 optimizer,
                 param_scheduler,
                 lr_scheduler,
                 train_loader,
                 validation_loader,
                 save="directory",
                 ext=".png"):
        super().__init__()
        self.args = args
        self.save = save
        self.ext = ext
        self.optimizer = optimizer
        self.param_scheduler = param_scheduler
        self.validation_loader = validation_loader
        self.lr_scheduler = lr_scheduler
        self.model = model_and_loss.model
        self.num_train_steps = len(train_loader) if train_loader is not None else 0
        self.num_valid_steps = len(validation_loader) if validation_loader is not None else 0
        if save == "directory":
            logging.info("Choose save directory!")
            quit()

    def on_step_finished(self, example_dict, model_dict, loss_dict, train, step, total_steps):
        if train:
            warnings.warn("DBNInference is supposed to be used at test time", UserWarning)
        else:
            basenames = example_dict['basename']
            outputs = model_dict['output1']

            batch_size, c, h, w = outputs.size()
            for b in range(batch_size):
                basename = basenames[b]
                output = outputs[b, ...]
                filename = os.path.join(self.save, basename + self.ext)
                system.ensure_dir(filename)
                save_image(output, filename=filename, nrow=1, padding=0, normalize=False)


factory.register('GoProInference', GoProInference)
