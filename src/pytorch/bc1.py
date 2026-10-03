import torch
import torch.nn as nn
import numpy as np
from src.pytorch.common import get_device
from src.pytorch.texture import triFetch, upsample
from src.bc1_export import export_bc1

# Tensors used to quantize the color endpoint tensors
quantScales = torch.tensor([31, 63, 31], dtype=torch.float32).to(get_device())
quantInvScales = torch.tensor([1 / 31, 1 / 63, 1 / 31], dtype=torch.float32).to(get_device())

# Straight through estimator for BC1 quantization
class STEBC1Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e0, e1, a):
        e0 = torch.round(e0 * quantScales) * quantInvScales
        e1 = torch.round(e1 * quantScales) * quantInvScales
        a  = torch.round(a * 3.0) / 3.0
        return (e0, e1, a)

    @staticmethod
    def backward(ctx, e0Out, e1Out, aOut):
        return (e0Out, e1Out, aOut)

# Class that represents a BC1 mip chain
class BC1Texture(nn.Module):
    def __init__(self, w: int, h: int, m: int, data: bytes | None = None, quantized: bool = False):
        super().__init__()
        self.width = w
        self.height = h
        self.mip_count = m
        self.channels = 3

        # Is this texture being initialized with quantized data?
        self.quantized = quantized

        self.e0 = []
        self.e1 = []
        self.a = []

        # w, h, and d must match data
        if data is not None:
            self.load_from_binary(data)
            return

        # For each mip, create 2 color endpoint textures, and one alpha interpolation texture
        # Each is randomly initialized ([0-1))
        for i in range(m):
            mip_w = w >> i
            mip_h = h >> i

            e0 = nn.parameter.Parameter(torch.rand(mip_w // 4, mip_h // 4, 3))
            self.register_parameter(f'e0-{i}', e0)
            self.e0.append(e0)

            e1 = nn.parameter.Parameter(torch.rand(mip_w // 4, mip_h // 4, 3))
            self.register_parameter(f'e1-{i}', e1)
            self.e1.append(e1)

            a = nn.parameter.Parameter(torch.rand(mip_h, mip_w, 1))
            self.register_parameter(f'a-{i}', a)
            self.a.append(a)

    # Returns a bytearray representing all mips in the array
    @torch.no_grad()
    def export(self):
        if not self.quantized:
            self.quantize()

        e0_np = []
        e1_np = []
        a_np = []
        for i in range(len(self.e0)):
            e0_np.append(self.e0[i].cpu().numpy())
            e1_np.append(self.e1[i].cpu().numpy())
            a_np.append(self.a[i].cpu().numpy())

        return export_bc1(e0_np, e1_np, a_np)

    # Load bc1 texture from binary data
    @torch.no_grad()
    def load_from_binary(self, data: bytes):
        offset = 0
        for i in range(self.mip_count):
            w = self.width >> i
            h = self.height >> i
            if w % 4 or h % 4:
                self.mip_count = i
                break

            # Single out the mip's blocks from the buffer
            count = (w * h) // 16
            mip = np.frombuffer(data, '<u8', count, offset)
            mip = mip.reshape((w // 4, h // 4, 1))
            offset += count

            # Load color endpoint and normalize them
            e0 = mip & 65535
            e0 = np.concatenate([
                ((e0 >> 11) & 31) / 31,
                ((e0 >> 5) & 63) / 63,
                (e0 & 31) / 31
            ], axis = 2).astype(np.float32)

            e1 = (mip >> 16) & 65535
            e1 = np.concatenate([
                ((e1 >> 11) & 31) / 31,
                ((e1 >> 5) & 63) / 63,
                (e1 & 31) / 31
            ], axis = 2).astype(np.float32)

            self.e0.append(torch.from_numpy(e0))
            self.e1.append(torch.from_numpy(e1))
            self.a.append(torch.zeros((w, h, 1)))

            a_offset = 32
            for x in range(4):
                for y in range(4):
                    a = (mip >> a_offset) & 3
                    # Switch from DXT1 specification color indices to alpha interpolation values
                    a = a ^ ((a >> 1) ^ 1)
                    a = (a - 1) & 3
                    a = torch.from_numpy((a / 3).astype(np.float32))

                    self.a[i][x::4,y::4] = a
                    a_offset += 2

            self.e0[i] = nn.Parameter(self.e0[i])
            self.e1[i] = nn.Parameter(self.e1[i])
            self.a[i] = nn.Parameter(self.a[i])

    # Quantization applied to all mips after training
    @torch.no_grad()
    def quantize(self):
        self.quantized = True
        for i in range(self.mip_count):
            e0 = self.e0[i]
            e1 = self.e1[i]
            a  = self.a[i]

            e0 = torch.round(torch.sigmoid(e0) * quantScales) * quantInvScales
            e1 = torch.round(torch.sigmoid(e1) * quantScales) * quantInvScales
            a  = torch.round(torch.sigmoid(a) * 3.0) / 3.0

            self.e0[i] = e0
            self.e1[i] = e1
            self.a[i]  = a


    # Quantizes the provided e0, e1, and a values during training
    def quantize_mip(self, e0: torch.tensor, e1: torch.tensor, a: torch.tensor):
        # Sigmoid activation to clamp between (0-1)
        e0 = torch.sigmoid(e0)
        e1 = torch.sigmoid(e1)
        a  = torch.sigmoid(a)

        # Straight through estimator so we get full f32 gradients
        e0, e1, a = STEBC1Quantize.apply(e0, e1, a)

        return (e0, e1, a)

    # Decodes the two nearest mips into regular RGB format
    def decode_mips(self, s: torch.tensor):
        s0 = torch.floor(s).int()
        s1 = torch.clamp(s0 + 1, max=self.mip_count - 1)

        # Color endpoints are at quarter res, so upscale them so they can be lerped by a
        e0s0 = upsample(self.e0[s0], 4)
        e0s1 = upsample(self.e0[s1], 4)
        e1s0 = upsample(self.e1[s0], 4)
        e1s1 = upsample(self.e1[s1], 4)
        as0 = self.a[s0]
        as1 = self.a[s1]

        # Quantize if not already quantized
        if not self.quantized:
            e0s0, e1s0, as0 = self.quantize_mip(e0s0, e1s0, as0)
            e0s1, e1s1, as1 = self.quantize_mip(e0s1, e1s1, as1)

        # Decode to RGB values
        mips0 = torch.lerp(e0s0, e1s0, as0)
        mips1 = torch.lerp(e0s1, e1s1, as1)

        return (mips0, mips1)

    # Performs trilinear interpolation on this BC1 mip chain
    def sample(self, uv: torch.tensor, s: torch.tensor, align: bool = False):
        s = torch.clamp(s, min=0.0, max=self.mip_count)
        mips0, mips1 = triFetch(self.decode_mips(s), uv, torch.frac(s), 'bilinear', align)

        return torch.lerp(mips0, mips1, torch.frac(s))

    def forward(self, x):
        return self.sample(*x)

