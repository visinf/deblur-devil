import torch
import torch.nn as nn

from contrib import weight_init
from contrib.interpolation import FlowWarpingLayer
from contrib.interpolation import resize2d_as
from contrib.spatial_correlation_sampler import Correlation
from models import factory


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, is_relu=True):
    if is_relu:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class FeatureExtractor(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.num_chs = num_inputs
        self.convs = nn.ModuleList()
        for l, (ch_in, ch_out) in enumerate(zip(num_inputs[:-1], num_inputs[1:])):
            layer = nn.Sequential(conv(ch_in, ch_out, stride=2), conv(ch_out, ch_out))
            self.convs.append(layer)

    def forward(self, x):
        features = []
        for layer in self.convs:
            x = layer(x)
            features.append(x)
        return features[::-1]


class WarpingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.warp = FlowWarpingLayer()

    def forward(self, value, flow, height_in, width_in):
        b, c, height_out, width_out = value.size()
        u, v = flow.chunk(chunks=2, dim=1)
        u = u * (width_out / width_in)
        v = v * (height_out / height_in)
        scaled_flow = torch.cat((u, v), dim=1)
        warped = self.warp(value, scaled_flow)
        return warped


class FlowEstimatorDense(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.conv1 = conv(num_inputs, 128)
        self.conv2 = conv(num_inputs + 128, 128)
        self.conv3 = conv(num_inputs + 256, 96)
        self.conv4 = conv(num_inputs + 352, 64)
        self.conv5 = conv(num_inputs + 416, 32)
        self.conv_last = conv(num_inputs + 448, 2, is_relu=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class ContextNetwork(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.convs = nn.Sequential(
            conv(num_inputs, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 2, is_relu=False)
        )

    def forward(self, x):
        return self.convs(x)


class PWCNetImpl(nn.Module):
    def __init__(self, args, div_flow):
        super().__init__()
        self.args = args
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.flow_estimators = nn.ModuleList()
        self.warping_layers = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.corrn = nn.ModuleList()
        self.div_flow = div_flow
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        for i, channels in enumerate(self.num_chs[::-1]):
            if i > self.output_level:
                break
            num_ch_in = self.dim_corr if i == 0 else self.dim_corr + channels + 2
            self.flow_estimators.append(FlowEstimatorDense(num_ch_in))
            self.warping_layers.append(WarpingLayer())
            self.relus.append(nn.LeakyReLU(0.1))
            self.corrn.append(Correlation())

        self.context_network = ContextNetwork(self.dim_corr + 32 + 2 + 2 + 448)
        weight_init.msra_(self.modules(), mode='fan_in', nonlinearity='leaky_relu')

    def extract_pyramid_featyres(self, input1, input2):
        feat1 = []
        feat2 = []
        pyramid_features = self.feature_pyramid_extractor(torch.cat((input1, input2), dim=0))
        for feat in pyramid_features:
            f1, f2 = feat.chunk(2, dim=0)
            feat1.append(f1)
            feat2.append(f2)
        feat1.append(input1)
        feat2.append(input2)
        return feat1, feat2

    def forward(self, input1, input2):
        height_im, width_im = input1.size()[2:4]
        im1_pyramid, im2_pyramid = self.extract_pyramid_featyres(input1, input2)
        all_flows = []
        flow = None
        feat = None
        x2_warp = im2_pyramid[0]
        for i, (x1, x2) in enumerate(zip(im1_pyramid, im2_pyramid)):
            # initially warp layers with current flow estimate
            if i > 0:
                flow = resize2d_as(flow, x1, mode="bilinear", align_corners=False)
                x2_warp = self.warping_layers[i](x2, flow / self.div_flow, height_im, width_im)
            # apply correlation layer
            out_corr = self.relus[i](self.corrn[i](x1, x2_warp))
            # estimate flow for current level
            estimator_features = out_corr if i == 0 else torch.cat((out_corr, x1, flow), dim=1)
            feat, flow = self.flow_estimators[i](estimator_features)
            # check if we are done
            if i == self.output_level:
                break
            # append level flow
            all_flows.append(flow)

        # run context network
        flow_residual = self.context_network(torch.cat([feat, flow], dim=1))
        flow += flow_residual
        all_flows.append(flow)

        return all_flows


class PWCNet(nn.Module):
    def __init__(self, args, div_flow=1.0):
        super().__init__()
        self.pwcnet_impl = PWCNetImpl(args, div_flow=div_flow)
        self.div_flow = div_flow

    def forward(self, input_dict):
        im1 = input_dict['input1']
        im2 = input_dict['input2']

        output_dict = {}
        flow6, flow5, flow4, flow3, flow2 = self.pwcnet_impl(im1, im2)
        if self.training:
            output_dict['flow2'] = flow2
            output_dict['flow3'] = flow3
            output_dict['flow4'] = flow4
            output_dict['flow5'] = flow5
            output_dict['flow6'] = flow6
        else:
            output_dict['flow1'] = resize2d_as(flow2, im1, mode="bilinear") / self.div_flow
        return output_dict


factory.register("PWCNet", PWCNet)
