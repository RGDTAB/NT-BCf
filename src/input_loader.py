import sys
from pathlib import Path
import numpy as np
import cv2
try:
    import OpenEXR
except ImportError:
    OpenEXR = None
from enum import Enum

class InputTextureType(Enum):
    STANDARD = 0 # Use all channels
    NORMAL_MAP = 1 # Use first 2 channels
    SINGLE_CHANNEL = 2 # Use first channel

# Loads the passed textures into a list of ndarrays, one ndarray per mip
# Also returns the 
def bulk_load_textures(standard: list|None, normals: list|None, single_chan: list|None, mip_count: int, generate_mips: bool):
    individual_mip_chains = []
    # Load the textures in as ndarrays
    width = height = None
    if standard is not None:
        standard_textures, width, height = load_mip_chains(standard, InputTextureType.STANDARD, mip_count, generate_mips, (width, height))
        individual_mip_chains += standard_textures
    if normals is not None:
        normal_textures, width, height = load_mip_chains(normals, InputTextureType.NORMAL_MAP, mip_count, generate_mips, (width, height))
        individual_mip_chains += normal_textures
    if single_chan is not None:
        single_textures, width, height = load_mip_chains(single_chan, InputTextureType.SINGLE_CHANNEL, mip_count, generate_mips, (width, height))
        individual_mip_chains += single_textures

    if len(individual_mip_chains) == 0:
        print(f'ERROR: No target textures were passed! Exiting...', file=sys.stderr)
        exit()

    texture_channels = []
    for chain in individual_mip_chains:
        texture_channels.append(chain[0].shape[2])

    mip_chain = []
    for i in range(len(individual_mip_chains[0])):
        mip_chain.append(np.concatenate([mip[i] for mip in individual_mip_chains], axis=2))

    return (mip_chain, texture_channels)

# Returns the pixel data from an OpenEXR file as a numpy array
def load_exr(file: Path):
    if OpenEXR is None:
        print(f'ERROR: OpenEXR is not supported (did you install it?) Exiting...', file=sys.stderr)
        return None

    tex = None
    # Open OpenEXR file
    with OpenEXR.File(str(file)) as exr_file:
        # EXR files are occasionally split into parts, currently only single part files are supported.
        # This is mostly due to the fact that multipart files can have different resolutions in different parts
        if len(exr_file.parts) != 1:
            print(f'ERROR: Issue opening {file}! Multipart EXR files are not supported. Exiting...', file=sys.stderr)
            exit()

        # Concatenate pixels for different channels together
        for name,channel in exr_file.channels().items():
            tex = channel.pixels if tex is None else np.concatenate((tex, channel.pixels), axis=2)

    return tex

# Returns the ndarray loaded from the texture file
def load_texture(file: Path, tex_type: InputTextureType) -> np.ndarray:
    if not file.lower().endswith('.exr'):
        tex = cv2.imread(file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        # Opencv loads multiple channel images as 3d arrays
        if tex.ndim == 3:
            if tex.shape[2] == 3:
                tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
            elif tex.shape[2] == 4:
                tex = cv2.cvtColor(tex, cv2.COLOR_BGRA2RGBA)
        # ... and loads single channel images as 2d arrays
        else:
            tex = np.expand_dims(tex, axis=2)
    else:
        tex = load_exr(file)

    if tex is None:
        print(f'ERROR: Issue opening {file}! Exiting...', file=sys.stderr)
        exit()

    if tex.shape[0] % 4 != 0 or tex.shape[1] % 4 != 0:
        print(f'ERROR: {file}\'s resolution isn\'t divisible by 4! Exiting...', file=sys.stderr)
        exit()

    # Scale the input files to [0..1]
    if tex.dtype == np.uint8:
        tex = tex.astype(np.float32) * 1/255
    elif tex.dtype == np.uint16:
        tex = tex.astype(np.float32) * 1/65536
    else:
        tex = tex.astype(np.float32)

    # Normal map - keep only 1st and 2nd channel, reconstruct 3rd at runtime
    if tex_type == InputTextureType.NORMAL_MAP:
        tex = np.ascontiguousarray(tex[:,:,0:min(tex.shape[2],2)])
    # Single channel - discard all but the first channel
    elif tex_type == InputTextureType.SINGLE_CHANNEL:
        tex = np.ascontiguousarray(tex[:,:,0:1])

    return tex

# Checks to see that all mips are at the proper resolution
def is_mip_chain_valid(mip_chain: list) -> bool:
    prev_width = mip_chain[0].shape[0]
    prev_height = mip_chain[0].shape[1]
    for mip in range(1, len(mip_chain)):
        if mip_chain[mip].shape[0] != prev_width // 2 or mip_chain[mip].shape[1] != prev_height // 2:
            return False
        if mip_chain[mip].shape[0] % 4 != 0 or mip_chain[mip].shape[1] % 4 != 0:
            return False
        prev_width = mip_chain[mip].shape[0]
        prev_height = mip_chain[mip].shape[1]
        
    return True

# Load all mip chains of a given InputTextureType, and return them as a list of ndarrays
def load_mip_chains(files: list, tex_type: InputTextureType, mip_count: int, generate_mips: bool, mip0_dims):
    mip_chains = []

    mip0_width, mip0_height = mip0_dims
    for file in files:
        # Mip chains are passed as a single string, with individual paths being comma delimited
        chain_paths = file.split(',')

        chain = []
        # Load all mips into the chain
        for mip in chain_paths:
            chain.append(load_texture(mip, tex_type))
        # Sort the mips from largest to smallest
        chain.sort(reverse = True, key = lambda e: e.shape[0])

        # Confirm that all mips are of the proper resolution
        if not is_mip_chain_valid(chain):
            print(f'ERROR: {file} is a malformed mip chain, exiting', file=sys.stderr)
            exit()

        # This is the first mip chain
        if mip0_width is None or mip0_height is None:
            mip0_width = chain[0].shape[0]
            mip0_height = chain[0].shape[1]
        # The dimensions of this mip chain must match that of the previous chains
        elif mip0_width != chain[0].shape[0] or mip0_height != chain[0].shape[1]:
            print(f'ERROR: Mip0 of {file} has different dimensions compared to previous mip chains, exiting', file=sys.stderr)
            exit()

        max_mip = get_highest_mip(mip0_width, mip0_height)
        # mip_count == -1 means 'use as many mips as possible'
        if mip_count == -1:
            mip_count = max_mip
        else:
            mip_count = min(mip_count, max_mip)

        # We're missing mips
        if len(chain) < mip_count:
            if not generate_mips:
                print(f'ERROR: {file} doesn\'t have the required number of mips, exiting', file=sys.stderr)
                exit()
            else:
                append_missing_mips(chain, mip_count)

        # We have too many mips? Discard the extras
        elif len(chain) > mip_count:
            chain = chain[:mip_count]

        mip_chains.append(chain)

    return (mip_chains, mip0_width, mip0_height)

# Gets the largest mip allowed with a mip0 of size (width, height)
def get_highest_mip(width: int, height: int):
    mips = min(width.bit_length() - 2, height.bit_length() - 2)
    for i in range(1, mips):
        if (width >> i) % 4 != 0 or (height >> i) % 4 != 0:
            mips = i
            break
    return mips

# Generates mips to complete a mip chain
def append_missing_mips(mip_chain: list, required_mip_count: int):
    width = mip_chain[0].shape[0]
    height = mip_chain[0].shape[1]
    old_len = len(mip_chain)

    missing_mips = required_mip_count - len(mip_chain)
    for i in range(missing_mips):
        # Inter-area is good enough as a downsampling operator
        mip = cv2.resize(mip_chain[0], (width >> (i + old_len), height >> (i + old_len)), interpolation=cv2.INTER_AREA)

        # Single channel images get squeezed down to a 2d array for some reason
        if mip.ndim == 2:
            mip = np.expand_dims(mip, axis=2)
        mip_chain.append(mip)

