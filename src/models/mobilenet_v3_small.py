import os

import torch
import torch.nn as nn
import hydra

from .backbones import mobilenetv3_small


class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small"""
    def __init__(self, cfg):
        super().__init__()

        self.deconv_with_bias = cfg.model.deconv_with_bias

        self.backbone = mobilenetv3_small()
        
        cwd = hydra.utils.get_original_cwd()
        self.backbone.load_state_dict(torch.load(
            os.path.join(cwd, 'models/pretrained/mobilenetv3-small-55df8e1f.pth')))

        self.inplanes = 576

        self.deconv_layers = self._make_deconv_layer(
            cfg.model.num_deconv_layers,
            cfg.model.num_deconv_filters,
            cfg.model.num_deconv_kernels,
        )

        self.final_layer = nn.Conv2d(
            in_channels=cfg.model.num_deconv_filters[-1],
            out_channels=cfg.model.num_joints,
            kernel_size=cfg.model.final_conv_kernel,
            stride=1,
            padding=1 if cfg.model.final_conv_kernel == 3 else 0,
        )

        self._init_weights()

    @staticmethod
    def _get_deconv_cfg(kernel_size):
        if kernel_size == 4:
            padding = 1
            output_padding = 0
        elif kernel_size == 3:
            padding = 1
            output_padding = 1
        elif kernel_size == 2:
            padding = 0
            output_padding = 0
        else:
            raise NotImplementedError

        return kernel_size, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.conv(x)

        x = self.deconv_layers(x)
        out = self.final_layer(x)

        return out

    def _init_weights(self):
        # init deconv layers
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # init final layer
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


def mobilenet_v3_small(cfg):
    return MobileNetV3Small(cfg)
