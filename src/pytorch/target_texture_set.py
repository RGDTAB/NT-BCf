import numpy as np
import torch
from src.input_loader import bulk_load_textures
from src.pytorch.common import get_device
from src.pytorch.texture import triFetch

# Class representing the target textures for optimization
class TargetTextureSet():
    def __init__(self, standard: list|None, normals: list|None, single_chan: list|None, max_mip: int, gen_mips: bool):
        device = get_device()

        # Load target textures from disk
        self.mip_chain, self.texture_channels = bulk_load_textures(
            standard, normals, single_chan,
            max_mip, gen_mips,
        )

        # Convert them to torch tensors
        for i in range(len(self.mip_chain)):
            self.mip_chain[i] = torch.from_numpy(self.mip_chain[i]).to(device)

        self.width = self.mip_chain[0].shape[0]
        self.height = self.mip_chain[0].shape[1]
        self.mip_count = len(self.mip_chain)
        self.channels = self.mip_chain[0].shape[2]

    # Performs bicubic interpolation, with linear interpolation between mips
    def sample(self, uv: torch.tensor, s: torch.tensor):
        mips0, mips1, = triFetch(self.mip_chain, uv, s, 'bicubic')
        return torch.lerp(mips0, mips1, torch.frac(s))
