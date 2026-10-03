#!/usr/bin/env python

import slangpy as spy
from src.args import parse_args
from src.slangpy.encoder import BCf1Encoder
from src.slangpy.encoder_window import EncoderWindow

def main():
    # DX12 doesn't support atomic floats
    device = spy.Device(type=spy.DeviceType.vulkan)

    args = parse_args('encoder-slangpy.py')
    if not args.no_gui:
        # GUI path
        app = spy.App(device)
        window = EncoderWindow(args, app)
        app.run()
    else:
        # No GUI path
        encoder = BCf1Encoder(args, device=device)
        while encoder.epoch <= args.training_epochs:
            encoder.train_epoch()

if __name__ == '__main__':
    main()
