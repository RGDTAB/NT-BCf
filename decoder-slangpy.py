#!/usr/bin/env python

import argparse
import slangpy as spy
from src.slangpy.decoder_window import DecoderWindow

def main():
    parser = argparse.ArgumentParser(
        prog = 'decoder-slangpy.py',
        description = 'Neural texture compression with block compressed features - DECODER'
    )

    parser.add_argument('file', help='Path to neural texture you would like to view.')
    args = parser.parse_args()

    # Vulkan for consistency
    device = spy.Device(type=spy.DeviceType.vulkan)

    app = spy.App(device)
    window = DecoderWindow(args, app)
    app.run()

if __name__ == '__main__':
    main()
