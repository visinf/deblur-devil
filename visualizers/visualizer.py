# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

from torch import nn


# ------------------------------------------
# That is how a Visualizer looks like
# ------------------------------------------
class Visualizer(nn.Module):
    # ------------------------------------------
    # on epoch initialization
    # ------------------------------------------
    def on_train_epoch_init(self, lr, epoch, total_epochs):
        pass

    def on_valid_epoch_init(self, lr, epoch, total_epochs):
        pass

    def on_epoch_init(self, lr, train, epoch, total_epochs):
        if train:
            self.on_train_epoch_init(lr, epoch, total_epochs)
        else:
            self.on_valid_epoch_init(lr, epoch, total_epochs)

    # ------------------------------------------
    # on step initialization
    # ------------------------------------------
    def on_train_step_init(self, example_dict, step, total_steps):
        pass

    def on_valid_step_init(self, example_dict, step, total_steps):
        pass

    def on_step_init(self, example_dict, train, step, total_steps):
        if train:
            self.on_train_step_init(example_dict, step, total_steps)
        else:
            self.on_valid_step_init(example_dict, step, total_steps)

    # ------------------------------------------
    # on step finished
    # ------------------------------------------
    def on_train_step_finished(self, example_dict, model_dict, loss_dict, step, total_steps):
        pass

    def on_valid_step_finished(self, example_dict, model_dict, loss_dict, step, total_steps):
        pass

    def on_step_finished(self, example_dict, model_dict, loss_dict, train, step, total_steps):
        if train:
            self.on_train_step_finished(example_dict, model_dict, loss_dict, step, total_steps)
        else:
            self.on_valid_step_finished(example_dict, model_dict, loss_dict, step, total_steps)

    # ------------------------------------------
    # on epoch finished
    # ------------------------------------------
    def on_train_epoch_finished(self, avg_loss_dict, epoch, total_epochs):
        pass

    def on_valid_epoch_finished(self, avg_loss_dict, epoch, total_epochs):
        pass

    def on_epoch_finished(self, avg_loss_dict, train, epoch, total_epochs):
        if train:
            self.on_train_epoch_finished(avg_loss_dict, epoch, total_epochs)
        else:
            self.on_valid_epoch_finished(avg_loss_dict, epoch, total_epochs)
