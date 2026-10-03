import torch
import torch.nn.functional as Functional
import random
import math
from enum import Enum

# PyTorch texture functions expect tensors with a shape of (number, channels, height width)
# That can get a bit confusing, so we just permute the tensors temporarily
def toCHW(mip: torch.tensor) -> torch.tensor:
    return mip.permute(*torch.arange(mip.ndim - 1, -1, -1)).unsqueeze(0)

def toWHC(mip: torch.tensor) -> torch.tensor:
    return mip.permute(*torch.arange(mip.ndim - 1, -1, -1)).squeeze(-1)

# Fetch grid of samples for closest mips to lod-level s
def triFetch(mips: list, uv: torch.tensor, s: torch.tensor, mode: str, align_corners: bool = False) -> torch.tensor:
    s0 = torch.floor(s).int()
    s1 = torch.clamp(s0 + 1, max=len(mips) - 1)

    mips0 = toCHW(mips[s0])
    mips1 = toCHW(mips[s1])

    mips0 = Functional.grid_sample(mips0, uv, mode=mode, padding_mode='border', align_corners=align_corners)
    mips1 = Functional.grid_sample(mips1, uv, mode=mode, padding_mode='border', align_corners=align_corners)

    mips0 = toWHC(mips0)
    mips1 = toWHC(mips1)

    return mips0, mips1

# Upscale a tensor to be n times as large than it currently is
def upsample(mip: torch.tensor, n: int):
    mip = toCHW(mip)
    # Uses nearest neighbor interpolation
    mip = Functional.interpolate(mip, scale_factor=n)
    mip = toWHC(mip)
    return mip

# Create a uniform grid of locations with a random offset, and a random lod level
def get_batch(width, height, depth, batch_size: int, device: torch.device):
    # Nvidia NTC mip sampling scheme - mostly exponential, with occasional uniform sampling
    if random.random() > 0.05:
        # 95% chance of exponential distribution
        s = min(depth - 1.0, -math.log(random.random(), 4.0))
    else:
        # 5% chance to uniformly samply lod
        s = (depth - 1.0) * random.random()
    s0 = int(s)
    s = torch.tensor(s, dtype=torch.float32)

    # Create a uniform grid, with each sample one pixel apart
    # Keeping the individual samples close helps cache coherency
    grid_u, grid_v = torch.meshgrid(
        torch.linspace(0.0, batch_size / (width >> s0), batch_size, device=device),
        torch.linspace(0.0, batch_size / (height >> s0), batch_size, device=device), indexing='xy'
    )

    grid_u = grid_u.unsqueeze(2)
    grid_v = grid_v.unsqueeze(2)
    grid = torch.cat((grid_u, grid_v), 2)

    # Add a random offset
    grid += torch.rand(2, device=device)
    # Contain the grid in (-1, 1)
    grid = torch.frac(grid) * 2.0 - 1.0

    return (torch.unsqueeze(grid, 0), s)

# Creates a grid of samples for every pixel
def get_full_grid(width, height, device: torch.device):
    grid_u, grid_v = torch.meshgrid(
        torch.linspace(-1.0, 1.0, width, device=device),
        torch.linspace(-1.0, 1.0, height, device=device), indexing='xy'
    )
    grid_u = grid_u.unsqueeze(2)
    grid_v = grid_v.unsqueeze(2)

    grid = torch.cat((grid_u, grid_v), 2)

    return torch.unsqueeze(grid, 0)

