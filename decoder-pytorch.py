#!/usr/bin/env python

import torch
import argparse
from src.output_parser import parse_texture_params
from src.pytorch.texture import get_full_grid
from src.pytorch.bc1 import BC1Texture
from src.pytorch.bcf1 import BCf1
from src.pytorch.decoder_mlp import DecoderMLP
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        prog = 'decoder-pytorch.py',
        description = 'Neural texture compression with block compressed features - DECODER'
    )

    parser.add_argument('file', help='Path to neural texture you would like to view.')
    args = parser.parse_args()


    f = open(args.file, 'rb')
    data = f.read()
    f.close()

    # Reading texture parameters from the data buffer
    texture_params, input_channels, latent_params, latent_offsets, mlp_offsets = parse_texture_params(data)
    width, height, channels, mip_count, var_a, mlp_width, input_texture_count = texture_params
    t0_params, t1_params, t2_params, t3_params = latent_params
    t0_offset, t1_offset, t2_offset, t3_offset, t3_end = latent_offsets
    weight0_offset, bias0_offset, weight1_offset, bias1_offset, bias1_end = mlp_offsets

    # Reading latent textures from the buffer
    t0 = BC1Module(*t0_params, data[t0_offset:t1_offset], True)
    t1 = BC1Module(*t1_params, data[t1_offset:t2_offset], True)
    t2 = BC1Module(*t2_params, data[t2_offset:t3_offset], True)
    t3 = BC1Module(*t3_params, data[t3_offset:t3_end], True)

    # Combining them into a BCf1 for convenience
    latent_parameters = (width, height, mip_count, var_a)
    latent_textures = BCf1(latent_parameters, False)
    latent_textures.t0 = t0
    latent_textures.t1 = t1
    latent_textures.t2 = t2
    latent_textures.t3 = t3

    # Reading the MLP weights and biases from the buffer
    weights = [
        np.frombuffer(data, dtype='<f2', count=12 * mlp_width, offset=weight0_offset).astype(np.float32),
        np.frombuffer(data, dtype='<f2', count=mlp_width * channels, offset=weight1_offset).astype(np.float32)
    ]

    biases = [
        np.frombuffer(data, dtype='<f2', count=mlp_width, offset=bias0_offset).astype(np.float32),
        np.frombuffer(data, dtype='<f2', count=channels, offset=bias1_offset).astype(np.float32)
    ]

    # Loading the MLP data into the actual decoder
    decoder = DecoderMLP(12, mlp_width, channels)
    decoder.lin1.weight.data = torch.from_numpy(weights[0].reshape((mlp_width, 12)))
    decoder.lin2.weight.data = torch.from_numpy(weights[1].reshape((channels, mlp_width)))
    decoder.lin1.bias.data = torch.from_numpy(biases[0])
    decoder.lin2.bias.data = torch.from_numpy(biases[1])

    device = 'cpu'

    # Display the neural textures
    with torch.no_grad():
        batch = get_full_grid(width, height, device)
        latents = latent_textures((batch, torch.zeros(1)))
        y_h = decoder(latents)
        y_h = y_h.numpy()
        y_h = np.clip(y_h, 0.0, 1.0)

        # Display reconstructed textures in one window
        plt.figure()
        channel_start = 0
        for i in range(input_texture_count):
            channel_end = channel_start + input_channels[i]

            neural_img = y_h[:,:,channel_start : channel_end]
            if neural_img.shape[2] == 2:
                zeros = np.zeros((neural_img.shape[0], neural_img.shape[1], 1))
                neural_img = np.concatenate((neural_img, zeros), axis=2)

            plt.subplot(1,input_texture_count,i + 1)
            plt.imshow(neural_img, vmin=0.0, vmax=1.0)

            channel_start = channel_end

        # ... and the latent textures in another
        latents = latents.numpy()
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
