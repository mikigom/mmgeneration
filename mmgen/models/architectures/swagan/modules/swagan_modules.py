import torch
import torch.nn as nn
from mmcv.ops.upfirdn2d import upfirdn2d

from mmgen.core.runners.fp16_utils import auto_fp16
from mmgen.models.architectures.stylegan.modules import (ConvDownLayer,
                                                         DownsampleUpFIRDn)
from mmgen.models.architectures.stylegan.modules import \
    ModulatedToRGB as ModulatedToRGB_


def get_haar_wavelet():
    haar_wav_l = 1 / (2**0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2**0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h

    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh


class HaarTransform(nn.Module):

    def __init__(self):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet()

        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)

    def forward(self, input):
        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)

        return torch.cat((ll, lh, hl, hh), 1)


class InverseHaarTransform(nn.Module):

    def __init__(self):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet()

        self.register_buffer('ll', ll)
        self.register_buffer('lh', -lh)
        self.register_buffer('hl', -hl)
        self.register_buffer('hh', hh)

    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)
        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))

        return ll + lh + hl + hh


class ModulatedToRGB(ModulatedToRGB_):
    """To RGB layer.

    This module is designed to output image tensor in SWAGAN.

    Args:
        in_channels (int): Input channels.
        style_channels (int): Channels for the style codes.
        out_channels (int, optional): Output channels. Defaults to 3.
        upsample (bool, optional): Whether to adopt upsampling in features.
            Defaults to False.
        blur_kernel (list[int], optional): Blurry kernel.
            Defaults to [1, 3, 3, 1].
        style_mod_cfg (dict, optional): Configs for style modulation module.
            Defaults to dict(bias_init=1.).
        style_bias (float, optional): Bias value for style code.
            Defaults to 0..
        fp16_enabled (bool, optional): Whether to use fp16 training in this
            module. Defaults to False.
        conv_clamp (float, optional): Clamp the convolutional layer results to
            avoid gradient overflow. Defaults to `256.0`.
        out_fp32 (bool, optional): Whether to convert the output feature map to
            `torch.float32`. Defaults to `True`.
    """

    def __init__(self,
                 in_channels,
                 style_channels,
                 out_channels=3,
                 upsample=True,
                 blur_kernel=[1, 3, 3, 1],
                 style_mod_cfg=dict(bias_init=1.),
                 style_bias=0.,
                 fp16_enabled=False,
                 conv_clamp=256,
                 out_fp32=True):
        super(ModulatedToRGB,
              self).__init__(in_channels, style_channels, 4 * out_channels,
                             upsample, blur_kernel, style_mod_cfg, style_bias,
                             fp16_enabled, conv_clamp, out_fp32)

        if upsample:
            self.iwt = InverseHaarTransform()
            self.dwt = HaarTransform()

    @auto_fp16(apply_to=('x', 'style'))
    def forward(self, x, style, skip=None):
        """Forward Function.

        Args:
            x ([Tensor): Input features with shape of (N, C, H, W).
            style (Tensor): Style latent with shape of (N, C).
            skip (Tensor, optional): Tensor for skip link. Defaults to None.

        Returns:
            Tensor: Output features with shape of (N, C, H, W)
        """
        out = self.conv(x, style)
        out = out + self.bias.to(x.dtype)

        if self.fp16_enabled:
            out = torch.clamp(out, min=-self.conv_clamp, max=self.conv_clamp)

        # Here, Tero adopts FP16 at `skip`.
        if skip is not None:
            skip = self.iwt(skip)
            skip = self.upsample(skip)
            skip = self.dwt(skip)
            out = out + skip
        return out


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 blur_kernel=[1, 3, 3, 1],
                 fp16_enabled=False,
                 convert_input_fp32=True):
        super().__init__()

        self.fp16_enabled = fp16_enabled
        self.convert_input_fp32 = convert_input_fp32

        self.conv1 = ConvDownLayer(in_channel, in_channel, 3)
        self.conv2 = ConvDownLayer(
            in_channel,
            out_channel,
            3,
            downsample=True,
            blur_kernel=blur_kernel)

    @auto_fp16()
    def forward(self, input):
        if not self.fp16_enabled and self.convert_input_fp32:
            input = input.to(torch.float32)

        out = self.conv1(input)
        out = self.conv2(out)

        return out


class ModulatedFromRGB(nn.Module):

    def __init__(self,
                 out_channel,
                 downsample=True,
                 blur_kernel=[1, 3, 3, 1],
                 fp16_enabled=False,
                 convert_input_fp32=True):
        super().__init__()

        self.fp16_enabled = fp16_enabled
        self.convert_input_fp32 = convert_input_fp32

        self._downsample = downsample

        if self._downsample:
            self.iwt = InverseHaarTransform()
            self.downsample = DownsampleUpFIRDn(blur_kernel)
            self.dwt = HaarTransform()

        self.conv = ConvDownLayer(3 * 4, out_channel, 1)

    @auto_fp16()
    def forward(self, input, skip=None):
        if not self.fp16_enabled and self.convert_input_fp32:
            input = input.to(torch.float32)

        if self._downsample:
            input = self.iwt(input)
            input = self.downsample(input)
            input = self.dwt(input)

        out = self.conv(input)

        if skip is not None:
            out = out + skip

        return input, out
