# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import logging

import torch
import torch.nn as nn
from torch.nn import functional as tf

from models import factory
from models.deblurring.dbn import DBNImpl
from utils import checkpoints


class _Meshgrid(nn.Module):
    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.width = -1
        self.height = -1
        self.grid = torch.zeros([1, 1])

    def type_as(self, tensor):
        self.grid = self.grid.type_as(tensor)
        return self

    def update_grid(self, width, height):
        if self.width != width or self.height != height:
            rangex = torch.arange(0, width)
            rangey = torch.arange(0, height)
            xx = rangex.repeat(height, 1).contiguous()
            yy = rangey.repeat(width, 1).t().contiguous()
            if self.normalize:
                xx = (2.0 / (width - 1.0)) * self.xx.float() - 1.0
                yy = (2.0 / (height - 1.0)) * self.yy.float() - 1.0
            xx = xx.view(1, 1, height, width)
            yy = yy.view(1, 1, height, width)
            self.grid = torch.cat((xx, yy), dim=1).type_as(self.grid)
            self.width = width
            self.height = height
        return self.grid

    def forward(self, width, height):
        grid = self.update_grid(width=width, height=height)
        return grid


class _WarpLayer(nn.Module):
    def __init__(self, mode='bilinear', padding='border'):
        super().__init__()
        self.meshgrid = _Meshgrid()
        self.mode = mode
        self.padding = padding

    def forward(self, x, flo):
        b, c, h, w = x.size()
        grid = self.meshgrid.type_as(x)(width=w, height=h)

        vgrid = grid + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(w - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(h - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)

        output = tf.grid_sample(x, vgrid, mode=self.mode, padding_mode=self.padding)
        return output


class WarpingNetwork(nn.Module):
    def __init__(self,
                 args,
                 pretrained_flownet="filename",
                 pretrained_pwcnet="filename"):
        super().__init__()
        self.network = None

        if pretrained_flownet != "filename" and pretrained_pwcnet != "filename":
            logging.info('Either PWCNet or FlowNetS checkpoint must be given!')
            quit()

        if pretrained_flownet != "filename":
            from models.flow.flownet1s import FlowNet1S
            self.network = FlowNet1S(args, div_flow=0.05)
            logging.info("Loading FlowNet1s checkpoint from {}".format(pretrained_flownet))
            checkpoints.restore_module_from_filename(
                self.network,
                pretrained_flownet,
                translations={'model._flownets': 'flownets'},
                fuzzy_translation_keys=['conv1.0.weight'])

        elif pretrained_pwcnet != "filename":
            from models.flow.pwcnet import PWCNet
            self.network = PWCNet(args, div_flow=0.05)
            logging.info("Loading PWCNet checkpoint from {}".format(pretrained_pwcnet))
            checkpoints.restore_module_from_filename(
                self.network,
                pretrained_pwcnet,
                translations={'model.pwcnet': 'pwcnet'})
        else:
            logging.info('Flow network is initialized randomly!')

        for param in self.network.parameters():
            param.requires_grad = False

        self.warp_img = _WarpLayer(padding='border')

    def run_flow_network(self, x1, x2):
        self.network.eval()
        output_dict = self.network({'input1': x1, 'input2': x2})
        return output_dict['flow1']

    def forward(self, inputs):
        n = len(inputs)
        refidx = n // 2
        refinput = inputs[refidx]
        inputs_x1 = []
        inputs_x2 = []
        for i, inputx in enumerate(inputs):
            if i != refidx:
                inputs_x1.append(refinput)
                inputs_x2.append(inputx)
        x1 = torch.cat(inputs_x1, dim=0)
        x2 = torch.cat(inputs_x2, dim=0)
        flow = self.run_flow_network(x1, x2)
        warped_x2 = self.warp_img(x2, flow)
        warped = warped_x2.chunk(chunks=n - 1, dim=0)
        return list(warped)


def _determine_sequence_length(args):
    if hasattr(args, 'validation_dataset_sequence_length'):
        return args.validation_dataset_sequence_length
    elif hasattr(args, 'training_dataset_sequence_length'):
        return args.training_dataset_sequence_length
    else:
        raise ValueError('Could not determine sequence length from datasets')


class FlowDBNImpl(nn.Module):
    def __init__(self,
                 args,
                 pretrained_flownet="filename",
                 pretrained_pwcnet="filename",
                 output_channels=3):
        super().__init__()
        self.sequence_length = _determine_sequence_length(args)
        num_input_images = 2 * self.sequence_length - 1
        self.dbn_net = DBNImpl(args, input_channels=num_input_images * 3, output_channels=output_channels)
        self.warping_network = WarpingNetwork(
            args,
            pretrained_flownet=pretrained_flownet,
            pretrained_pwcnet=pretrained_pwcnet)

    def forward(self, inputs):
        with torch.no_grad():
            warped = self.warping_network(inputs)

        list_of_inputs = inputs + warped
        model_dict = self.dbn_net(list_of_inputs)

        names = []
        reference_index = len(warped) // 2
        for i in range(len(inputs)):
            if i != reference_index:
                names.append('warped{}'.format(i + 1))
        for name, warp in zip(names, warped):
            model_dict[name] = warp

        return model_dict


class FlowDBN(nn.Module):
    def __init__(self,
                 args,
                 pretrained_flownet="filename",
                 pretrained_pwcnet="filename"):
        super().__init__()
        self.flowdbn = FlowDBNImpl(
            args,
            pretrained_flownet=pretrained_flownet,
            pretrained_pwcnet=pretrained_pwcnet,
            output_channels=3)

    def forward(self, input_dict):
        input_keys = sorted([x for x in input_dict.keys() if 'input' in x])
        inputs = [input_dict[key] for key in input_keys]
        reference_index = len(inputs) // 2
        ref = inputs[reference_index]

        output_dict = self.flowdbn(inputs)
        f15 = output_dict['f15']
        output = f15 + ref

        if not self.training:
            output.clamp_(0, 1)

        output_dict['output1'] = output

        return output_dict


factory.register("FlowDBN", FlowDBN)
