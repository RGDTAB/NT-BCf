#!/usr/bin/env python

import argparse
import random
import torch
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2

from src.args import parse_args
from src.pytorch.common import get_device
from src.pytorch.texture import get_batch, get_full_grid
from src.pytorch.target_texture_set import TargetTextureSet
from src.pytorch.decoder_mlp import DecoderMLP
from src.pytorch.bcf1 import BCf1


def main():
    args = parse_args('encoder-pytorch.py')
    if args.output is None:
        print('ERROR: No output file set! Exiting...', file=sys.stderr)
        exit()

    # Load target textures
    target_set = TargetTextureSet(args.standard_tex, args.normal_map, args.single_channel, args.max_mips, args.generate_mips)

    width = target_set.width
    height = target_set.height
    mip_count = target_set.mip_count
    channels = target_set.channels

    # Properly configure latent textures if we want half res latents
    if not args.half_resolution:
        latent_parameters = (width, height, mip_count, args.format == 'BCf1A')
    else:
        latent_parameters = (max(width // 2, 4), max(height // 2, 4), max(mip_count - 1, 1), args.format == 'BCf1A')

    device = get_device()
    latent_textures = BCf1(latent_parameters).to(device)
    decoder = DecoderMLP(latent_textures.channels, args.mlp_width, target_set.channels).to(device)

    # Get the desired loss metric
    # 'reduction' is set to 'sum' to be kept consistent with SlangPy
    if not args.l1:
        loss_fn = torch.nn.MSELoss(reduction = 'sum')
    else:
        loss_fn = torch.nn.L1Loss(reduction = 'sum')

    # Create the ADAM optimizer
    optim = torch.optim.Adam([{'params' : latent_textures.parameters(), 'lr' : args.texture_lr},
                              {'params' : decoder.parameters(), 'lr' : args.mlp_lr}])

    # Create a cosine annealing scheduler if it was requested
    if args.cos:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.training_epochs)

    batch_size = args.batch_size

    loss_accum = 0.0
    start = time.time()
    # Training loop
    for e in range(args.training_epochs):
        batch, s = get_batch(width, height, mip_count, batch_size, device)

        # Forward pass
        y = target_set.sample(batch, s)
        latents = latent_textures((batch, s))
        y_h = decoder(latents)
        loss = loss_fn(y, y_h)

        if not args.quiet:
            loss_accum += loss
            if e % args.reporting_period == 0 and e != 0:
                loss_mean = loss_accum.item() / (args.batch_size ** 2 * channels * args.reporting_period)
                psnr = -10 * np.log10(loss_mean)
                print(f'EPOCH {e}: {psnr:.4f}dB')
                loss_accum = 0.0

        # Backwards pass
        loss.backward()
        # Step
        optim.step()
        optim.zero_grad()

        if args.cos:
            scheduler.step()

    end = time.time()
    if not args.quiet:
        print(f'Completed in {end - start:.2f} seconds')
        print(f'Each epoch took approximately {(end - start) / args.training_epochs * 1e3:.2f}ms')
        print(args.format, args.training_epochs)

    out_len = 0
    # Exporting
    if args.output is not None:
        with open(args.output, 'bw') as f:
            out_data = bytearray()

            # Neural texture details
            out_data += latent_parameters[0].to_bytes(2, 'little') # width
            out_data += latent_parameters[1].to_bytes(2, 'little') # height
            out_data += target_set.channels.to_bytes(1, 'little')  # channels
            out_data += latent_parameters[2].to_bytes(1, 'little') # max mip chain length
            out_data += latent_parameters[3].to_bytes(1, 'little') # Var A?
            out_data += args.mlp_width.to_bytes(1, 'little') # hidden layer width
            out_data += len(target_set.texture_channels).to_bytes(1, 'little') # num textures
            for i in range(len(target_set.texture_channels)):
                out_data += target_set.texture_channels[i].to_bytes(1, 'little') # channels in texture

            # Learned data
            out_data += latent_textures.export() # latent textures
            for param in decoder.parameters():
                numpy_param = param.detach().cpu().numpy().astype('<f2')
                out_data += numpy_param.tobytes() # decoder weights and biases
            out_len = len(out_data) // 1024
            f.write(out_data)

    print(f'Size: {out_len}kb')

    with torch.no_grad():
        texture_count = len(target_set.texture_channels)

        # Print PSNR values
        cumulative_y = np.empty(0)
        cumulative_y_h = np.empty(0)
        for i in range(mip_count):
            mip_batch = get_full_grid(width >> i, height >> i, device)
            mip_y = target_set.mip_chain[i]
            mip_y = mip_y.cpu().numpy()
            mip_latents = latent_textures.sample(mip_batch, torch.tensor([i], dtype=torch.float32, device=device), True)
            mip_y_h = decoder(mip_latents)
            mip_y_h = mip_y_h.cpu().numpy()
            print(f'Mip {i} PSNR: {cv2.PSNR(mip_y, mip_y_h, 1.0):.4f}db')
            cumulative_y = np.append(cumulative_y, mip_y.ravel())
            cumulative_y_h = np.append(cumulative_y_h, mip_y_h.ravel())
        print(f'Cumulative PSNR: {cv2.PSNR(cumulative_y, cumulative_y_h, 1.0):.4f}db')

        if args.no_gui:
            exit()

        # Display Mip0 results
        s = 0.0
        batch = get_full_grid(width >> int(s), height >> int(s), device)
        y = target_set.mip_chain[0]
        latents = latent_textures.sample(batch, torch.tensor([s], dtype=torch.float32, device=device), True)
        y_h = decoder(latents)

        y = y.cpu().numpy()
        y = np.clip(y, 0.0, 1.0)
        y_h = y_h.cpu().numpy()
        y_h = np.clip(y_h, 0.0, 1.0)

        plt.figure()
        channel_start = 0
        for i in range(texture_count):
            channel_end = channel_start + target_set.texture_channels[i]

            img = y[:,:,channel_start : channel_end]
            neural_img = y_h[:,:,channel_start : channel_end]
            if img.shape[2] == 2:
                zeros = np.zeros((img.shape[0], img.shape[1], 1))
                img = np.concatenate((img, zeros), axis=2)
                neural_img = np.concatenate((neural_img, zeros), axis=2)

            # Top row is the target, and bottom row is the reconstruction
            plt.subplot(2,texture_count,i + 1)
            plt.imshow(img, vmin=0.0, vmax=1.0)
            plt.subplot(2,texture_count,i + 1 + texture_count)
            plt.imshow(neural_img, vmin=0.0, vmax=1.0)
            channel_start = channel_end

        # Display latent textures in a separate window
        latents = latents.cpu().numpy()
        plt.figure()
        plt.subplot(1,4,1)
        plt.imshow(latents[:,:,0:3], vmin=0.0, vmax=1.0)
        plt.subplot(1,4,2)
        plt.imshow(latents[:,:,3:6], vmin=0.0, vmax=1.0)
        plt.subplot(1,4,3)
        plt.imshow(latents[:,:,6:9], vmin=0.0, vmax=1.0)
        plt.subplot(1,4,4)
        plt.imshow(latents[:,:,9:12], vmin=0.0, vmax=1.0)
        plt.show()

if __name__ == '__main__':
    main()
