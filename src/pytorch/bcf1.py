import torch
from src.pytorch.bc1 import BC1Texture

BCF1_CHANNELS = 12

# Helper class that concatenates the output of four BC1 textures
class BCf1(torch.nn.Module):
    def __init__(self, parameters, initialize_latents: bool = True):
        super().__init__()
        w, h, m, var_a = parameters
        self.width = w
        self.height = h
        self.mip_count = m
        self.channels = BCF1_CHANNELS

        self.var_a = var_a

        if not initialize_latents:
            return

        # Create the 4 latent textures
        # t0 is always at full res
        self.t0 = BC1Texture(w, h, m)
        if var_a: 
            self.t1 = BC1Texture(w, h, m)
            self.t2 = BC1Texture(max(w // 2, 4), max(h // 2, 4), max(m - 1, 1))
            self.t3 = BC1Texture(max(w // 2, 4), max(h // 2, 4), max(m - 1, 1))
        else:
            self.t1 = BC1Texture(max(w // 2, 4), max(h // 2, 4), max(m - 1, 1))
            self.t2 = BC1Texture(max(w // 4, 4), max(h // 4, 4), max(m - 2, 1))
            self.t3 = BC1Texture(max(w // 8, 4), max(h // 8, 4), max(m - 3, 1))

    # Concatenates the exported bytearrays of the 4 latent textures
    def export(self):
        out = bytearray()
        out += self.t0.export()
        out += self.t1.export()
        out += self.t2.export()
        out += self.t3.export()
        return out

    # Quantizes all 4 latent textures
    def quantize(self):
        self.t0.quantize()
        self.t1.quantize()
        self.t2.quantize()
        self.t3.quantize()

    # Samples all 4 latent textures, and concatenates the output
    def sample(self, uv: torch.tensor, s: torch.tensor, align: bool = False):
        # Negative mip bias for smaller latent textures
        s0 = torch.clamp(s, min=0, max=self.t0.mip_count - 1)
        if (self.var_a):
            s1 = torch.clamp(s, min=0, max=self.t1.mip_count - 1)
            s2 = torch.clamp(s - 1, min=0, max=self.t2.mip_count - 1)
            s3 = torch.clamp(s - 1, min=0, max=self.t3.mip_count - 1)
        else:
            s1 = torch.clamp(s - 1, min=0, max=self.t1.mip_count - 1)
            s2 = torch.clamp(s - 2, min=0, max=self.t2.mip_count - 1)
            s3 = torch.clamp(s - 3, min=0, max=self.t3.mip_count - 1)

        # Sample BC1Textures
        t0 = self.t0((uv, s0, align))
        t1 = self.t1((uv, s1, align))
        t2 = self.t2((uv, s2, align))
        t3 = self.t3((uv, s3, align))
        out = torch.cat([t0, t1, t2, t3], axis=2)
        return out

    # Wrapper around the sample method
    def forward(self, x):
        return self.sample(x[0], x[1])

