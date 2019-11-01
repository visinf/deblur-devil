# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import torch
import torch.nn as nn
import torch.nn.functional as tf

from contrib import weight_init
from contrib.interpolation import resize2d_as
from models import factory


def conv(in_planes, out_planes, kernel_size, stride, pad, nonlinear, bias):
    if nonlinear:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=pad, bias=bias),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)


def deconv(in_planes, out_planes, kernel_size, stride, pad, nonlinear, bias):
    if nonlinear:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=pad, bias=bias),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)


def concatenate_as(tensor_list, tensor_as, dim, mode="bilinear"):
    tensor_list = [resize2d_as(x, tensor_as, mode=mode) for x in tensor_list]
    return torch.cat(tensor_list, dim=dim)


def resize_tensors_as(tensor_list, tensor_as, mode="bilinear", align_corners=True):
    if isinstance(tensor_list, torch.Tensor):
        return resize2d_as(tensor_list, tensor_as, mode=mode, align_corners=align_corners)
    else:
        return [resize2d_as(x, tensor_as, mode=mode, align_corners=align_corners) for x in tensor_list]


def upsample2d_as(inputs, target_as, mode="bilinear", align_corners=True):
    _, _, h, w = target_as.size()
    return tf.interpolate(inputs, [h, w], mode=mode, align_corners=align_corners)


def upsample_flow_as(inputs, target_as, mode="bilinear", align_corners=True):
    _, _, h, w = target_as.size()
    factor = h / inputs.size(2)
    return factor * tf.upsample(inputs, [h, w], mode=mode, align_corners=align_corners)


class FlowNetS(nn.Module):
    def __init__(self, args, num_pred=2):
        super().__init__()
        self.args = args
        self.num_pred = num_pred

        def make_conv(in_planes, out_planes, kernel_size, stride):
            pad = kernel_size // 2
            return conv(in_planes, out_planes, kernel_size=kernel_size,
                        stride=stride, pad=pad, nonlinear=True, bias=True)

        self.conv1 = make_conv(6, 64, kernel_size=7, stride=2)
        self.conv2 = make_conv(64, 128, kernel_size=5, stride=2)
        self.conv3 = make_conv(128, 256, kernel_size=5, stride=2)
        self.conv3_1 = make_conv(256, 256, kernel_size=3, stride=1)
        self.conv4 = make_conv(256, 512, kernel_size=3, stride=2)
        self.conv4_1 = make_conv(512, 512, kernel_size=3, stride=1)
        self.conv5 = make_conv(512, 512, kernel_size=3, stride=2)
        self.conv5_1 = make_conv(512, 512, kernel_size=3, stride=1)
        self.conv6 = make_conv(512, 1024, kernel_size=3, stride=2)
        self.conv6_1 = make_conv(1024, 1024, kernel_size=3, stride=1)

        def make_deconv(in_planes, out_planes):
            return deconv(in_planes, out_planes, kernel_size=4, stride=2, pad=1,
                          nonlinear=True, bias=False)

        self.deconv5 = make_deconv(1024, 512)
        self.deconv4 = make_deconv(1024 + num_pred, 256)
        self.deconv3 = make_deconv(768 + num_pred, 128)
        self.deconv2 = make_deconv(384 + num_pred, 64)

        def make_predict(in_planes, out_planes):
            return conv(in_planes, out_planes, kernel_size=3, stride=1, pad=1,
                        nonlinear=False, bias=True)

        self.predict_flow6 = make_predict(1024, num_pred)
        self.predict_flow5 = make_predict(1024 + num_pred, num_pred)
        self.predict_flow4 = make_predict(768 + num_pred, num_pred)
        self.predict_flow3 = make_predict(384 + num_pred, num_pred)
        self.predict_flow2 = make_predict(192 + num_pred, num_pred)

        def make_upsample(in_planes, out_planes):
            return deconv(in_planes, out_planes, kernel_size=4, stride=2, pad=1,
                          nonlinear=False, bias=False)

        self.upsample_flow6_to_5 = make_upsample(num_pred, num_pred)
        self.upsample_flow5_to_4 = make_upsample(num_pred, num_pred)
        self.upsample_flow4_to_3 = make_upsample(num_pred, num_pred)
        self.upsample_flow3_to_2 = make_upsample(num_pred, num_pred)

        weight_init.msra_(self.modules(), mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3_1 = self.conv3_1(self.conv3(conv2))
        conv4_1 = self.conv4_1(self.conv4(conv3_1))
        conv5_1 = self.conv5_1(self.conv5(conv4_1))
        conv6_1 = self.conv6_1(self.conv6(conv5_1))

        predict_flow6 = self.predict_flow6(conv6_1)

        upsampled_flow6_to_5 = self.upsample_flow6_to_5(predict_flow6)
        deconv5 = self.deconv5(conv6_1)
        concat5 = concatenate_as((conv5_1, deconv5, upsampled_flow6_to_5), conv5_1, dim=1)
        predict_flow5 = self.predict_flow5(concat5)

        upsampled_flow5_to_4 = self.upsample_flow5_to_4(predict_flow5)
        deconv4 = self.deconv4(concat5)
        concat4 = concatenate_as((conv4_1, deconv4, upsampled_flow5_to_4), conv4_1, dim=1)
        predict_flow4 = self.predict_flow4(concat4)

        upsampled_flow4_to_3 = self.upsample_flow4_to_3(predict_flow4)
        deconv3 = self.deconv3(concat4)
        concat3 = concatenate_as((conv3_1, deconv3, upsampled_flow4_to_3), conv3_1, dim=1)
        predict_flow3 = self.predict_flow3(concat3)

        upsampled_flow3_to_2 = self.upsample_flow3_to_2(predict_flow3)
        deconv2 = self.deconv2(concat3)
        concat2 = concatenate_as((conv2, deconv2, upsampled_flow3_to_2), conv2, dim=1)
        predict_flow2 = self.predict_flow2(concat2)

        if self.training:
            return predict_flow2, predict_flow3, predict_flow4, predict_flow5, predict_flow6
        else:
            return predict_flow2


class FlowNet1S(nn.Module):
    def __init__(self, args, div_flow=1.0):
        super().__init__()
        self.flownets = FlowNetS(args)
        self.div_flow = div_flow

    def forward(self, input_dict):
        im1 = input_dict['input1']
        im2 = input_dict['input2']
        inputs = torch.cat((im1, im2), dim=1)

        output_dict = {}
        if self.training:
            flow2, flow3, flow4, flow5, flow6 = self.flownets(inputs)
            output_dict['flow2'] = flow2
            output_dict['flow3'] = flow3
            output_dict['flow4'] = flow4
            output_dict['flow5'] = flow5
            output_dict['flow6'] = flow6
        else:
            flow2 = self.flownets(inputs)
            output_dict['flow1'] = upsample2d_as(flow2, im1, mode="bilinear") / self.div_flow

        return output_dict


factory.register("FlowNet1S", FlowNet1S)
