# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation for Positional Encoding as Spatial Inductive Bias in GANs.

In this module, we provide necessary components to conduct experiments
mentioned in the paper: Positional Encoding as Spatial Inductive Bias in GANs.
More details can be found in: https://arxiv.org/pdf/2012.05217.pdf
"""
from functools import partial

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmgen.models.builder import MODULES, build_module
from .generator_discriminator import SinGANMultiScaleGenerator
from .modules import GeneratorBlock


@MODULES.register_module()
class SinGANMSGeneratorPENF(SinGANMultiScaleGenerator):
    """Multi-Scale Generator used in SinGAN with positional encoding.

    More details can be found in: Positional Encoding as Spatial Inductvie Bias
    in GANs, CVPR'2021.

    Notes:

    - In this version, we adopt the interpolation function from the official
      PyTorch APIs, which is different from the original implementation by the
      authors. However, in our experiments, this influence can be ignored.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        num_scales (int): The number of scales/stages in generator. Note
            that this number is counted from zero, which is the same as the
            original paper.
        kernel_size (int, optional): Kernel size, same as :obj:`nn.Conv2d`.
            Defaults to 3.
        padding (int, optional): Padding for the convolutional layer, same as
            :obj:`nn.Conv2d`. Defaults to 0.
        num_layers (int, optional): The number of convolutional layers in each
            generator block. Defaults to 5.
        base_channels (int, optional): The basic channels for convolutional
            layers in the generator block. Defaults to 32.
        min_feat_channels (int, optional): Minimum channels for the feature
            maps in the generator block. Defaults to 32.
        out_act_cfg (dict | None, optional): Configs for output activation
            layer. Defaults to dict(type='Tanh').
        padding_mode (str, optional): The mode of convolutional padding, same
            as :obj:`nn.Conv2d`. Defaults to 'zero'.
        positional_encoding (dict | None, optional): Configs for the positional
            encoding. Defaults to None.
        first_stage_in_channels (int | None, optional): The input channel of
            the first generator block. If None, the first stage will adopt the
            same input channels as other stages. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_scales,
                 kernel_size=3,
                 padding=0,
                 num_layers=5,
                 base_channels=32,
                 min_feat_channels=32,
                 out_act_cfg=dict(type='Tanh'),
                 padding_mode='zero',
                 positional_encoding=None,
                 first_stage_in_channels=None,
                 **kwargs):
        super(SinGANMultiScaleGenerator, self).__init__()

        self.with_positional_encode = positional_encoding is not None
        if self.with_positional_encode:
            self.head_position_encode = build_module(positional_encoding)

        self.pad_head = int((kernel_size - 1) / 2 * num_layers)
        self.blocks = nn.ModuleList()

        self.upsample = partial(
            F.interpolate, mode='bicubic', align_corners=True)

        for scale in range(num_scales + 1):
            base_ch = min(base_channels * pow(2, int(np.floor(scale / 4))),
                          128)
            min_feat_ch = min(
                min_feat_channels * pow(2, int(np.floor(scale / 4))), 128)

            if scale == 0:
                in_ch = (
                    first_stage_in_channels
                    if first_stage_in_channels else in_channels)
            else:
                in_ch = in_channels

            self.blocks.append(
                GeneratorBlock(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    num_layers=num_layers,
                    base_channels=base_ch,
                    min_feat_channels=min_feat_ch,
                    out_act_cfg=out_act_cfg,
                    padding_mode=padding_mode,
                    **kwargs))

        if padding_mode == 'zero':
            self.padding_layer = nn.ZeroPad2d(self.pad_head)
            self.noise_padding_layer = nn.ZeroPad2d(self.pad_head)
            self.img_padding_layer = nn.ZeroPad2d(self.pad_head)
            self.mask_padding_layer = nn.ReflectionPad2d(self.pad_head)
        elif padding_mode == 'reflect':
            self.padding_layer = nn.ReflectionPad2d(self.pad_head)
            self.noise_padding_layer = nn.ReflectionPad2d(self.pad_head)
            self.img_padding_layer = nn.ReflectionPad2d(self.pad_head)
            self.mask_padding_layer = nn.ReflectionPad2d(self.pad_head)
            mmcv.print_log('Using Reflection padding', 'mmgen')
        else:
            raise NotImplementedError(
                f'Padding mode {padding_mode} is not supported')

    def forward(self,
                init_res,
                noises,
                noise_weights,
                curr_scale,
                num_batches=1,
                get_prev_res=False,
                noise_rs_mode="pad",
                prev_rs_mode="pad"):
        """Forward function.

        Args:
            input_sample (Tensor | None): The input for generator. In the
                original implementation, a tensor filled with zeros is adopted.
                If None is given, we will construct it from the first fixed
                noises.
            fixed_noises (list[Tensor]): List of the fixed noises in SinGAN.
            noise_weights (list[float]): List of the weights for random noises.
            rand_mode (str): Choices from ['rand', 'recon']. In ``rand`` mode,
                it will sample from random noises. Otherwise, the
                reconstruction for the single image will be returned.
            curr_scale (int): The scale for the current inference or training.
            num_batches (int, optional): The number of batches. Defaults to 1.
            get_prev_res (bool, optional): Whether to return results from
                previous stages. Defaults to False.
            return_noise (bool, optional): Whether to return noises tensor.
                Defaults to False.

        Returns:
            Tensor | dict: Generated image tensor or dictionary containing \
                more data.
        """
        if get_prev_res:
            prev_res_list = []

        g_res = init_res

        for stage in range(curr_scale + 1):
            noise_ = noises[stage]

            g_res_rs = self.rescale_input(g_res, prev_rs_mode)
            noise_rs = self.rescale_input(noise_, noise_rs_mode)

            if self.with_positional_encode and stage == 0:
                head_grid = self.head_position_encode(noise_rs)
                noise_rs = noise_rs + head_grid

            assert g_res_rs.shape[-2:] == noise_rs.shape[-2:]

            if stage == 0 and self.with_positional_encode:
                noise = noise_rs * noise_weights[stage]
            else:
                noise = noise_rs * noise_weights[stage] + g_res_rs
            g_res = self.blocks[stage](noise.detach(), g_res)

            if get_prev_res and stage != curr_scale:
                prev_res_list.append(g_res)

            # upsample, here we use interpolation from PyTorch
            if stage != curr_scale:
                h_next, w_next = noises[stage + 1].shape[-2:]
                g_res = self.upsample(g_res, (h_next, w_next))

        if get_prev_res:
            output_dict = dict(
                fake_img=g_res,
                prev_res_list=prev_res_list,)
            return output_dict

        return g_res
    
    def rescale_input(self, x, rescale_mode="pad"):
        if rescale_mode == "pad":
            return self.padding_layer(x)
        elif rescale_mode == "interp":
            size = x.shape[-2:]
            size = (size[0] + 2 * self.pad_head,
                    size[1] + 2 * self.pad_head)
            return self.upsample(x, size)
        else:
            raise NotImplementedError("Unknown rescale method")