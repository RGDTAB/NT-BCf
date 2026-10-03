import argparse
import json
from pathlib import Path

# Parses command line arguments into an object
def parse_args(prog_name: str):
    parser = argparse.ArgumentParser(
        prog = prog_name,
        description = 'Neural texture compression with block compressed features - ENCODER'
    )
    
    parser.add_argument('-j', '--json', help='Read in arguments from this JSON file')

    # Arguments related to the target textures
    target_group = parser.add_argument_group('Target Textures', 'Each texture to be compressed. '
        + 'If a texture has artist created mip-maps, they can be used by '
        + 'combining the full chain into a comma separated string.'
    )

    target_group.add_argument('-i', '--standard-tex', help='Textures that don\'t require special considerations, ' +
        'unlike normal maps or single channel textures. All channels of the texture are used',
        nargs='*'
    )
    target_group.add_argument('-n', '--normal-map', help='Normal map textures. '
        + 'The z-component is discarded before training as it can trivially be reconstructed from the x and y components.',
        nargs='*'
    )
    target_group.add_argument('-s', '--single-channel', help='Single channel textures. Only the first channel is used for training.',
        nargs='*'
    )
    target_group.add_argument('-mm', '--max-mips', help='Deepest mip level to create/use', default=-1, type=int)
    target_group.add_argument('-gm', '--generate-mips', help='Should missing mips be automatically generated?', default=True, action=argparse.BooleanOptionalAction)

    # Arguments related to the desired latent textures
    latent_group = parser.add_argument_group('Latent Textures')
    latent_group.add_argument('-f', '--format', help='Quality vs. size tradeoff. BCf1A has higher quality, but uses more space compared to BCf1B.', default='BCf1A', choices=['BCf1A', 'BCf1B'])
    latent_group.add_argument('-hr', '--half-resolution', help='Store the latent textures at half of the original\'s resolution', action='store_true')

    # Arguments related to the desired MLP decoder
    mlp_group = parser.add_argument_group('MLP')
    mlp_group.add_argument('-mw', '--mlp-width', help='Width of the hidden layers in the MLP', default=16, type=int)

    # Arguments related to how the neural texture should be trained
    training_group = parser.add_argument_group('Training')
    training_group.add_argument('-e', '--training-epochs', help='Number of epochs to train. Ex: "15K"', default='1K')
    training_group.add_argument('-b', '--batch-size', help='Size (width and height) of each batch', default=512, type=int)
    training_group.add_argument('-tlr', '--texture-lr', help='Learning rate for the latent textures', default=0.01, type=float)
    training_group.add_argument('-mlr', '--mlp-lr', help='Learning rate for the decoder MLP weights and biases', default=0.001, type=float)
    training_group.add_argument('-cos', help='Use cosine annealing', action='store_true')
    training_group.add_argument('-l1', help='Optimize the neural texture using L1 loss instead of L2', action='store_true')

    # Arguments related to the output of the of the encoder
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-r', '--reporting-period', help='The number of epochs between each loss report', default=100, type=int)
    output_group.add_argument('-q', '--quiet', help='Silences training output until the very end', action='store_true')
    output_group.add_argument('--no-gui', help='No GUI is displayed during or after training', action='store_true')
    output_group.add_argument('-o', '--output', help='Output file containing weights and latent textures.')

    args = parser.parse_args()

    if args.json is not None:
        args = load_args_from_json(args.json)
        # Fun fact - parser.parse_args() will only replace values in the
        # provided namespace when they are explicitly set in the command line
        parser.parse_args(namespace=args)

    # Parse training epochs - Ends with 'k' = multiply by 1,000
    if type(args.training_epochs) is str:
        if args.training_epochs.isdigit():
            args.training_epochs = int(args.training_epochs)
        elif args.training_epochs.upper().endswith('K'):
            args.training_epochs = int(args.training_epochs.strip('kK')) * 1000

    args.training_epochs = max(args.training_epochs, 1)
    args.max_mips = max(args.max_mips, -1)
    args.batch_size = max(args.batch_size, 1)
    args.reporting_period = max(args.reporting_period, 1)

    return args

# Load the arguments stored in a JSON file into a Namespace object
def load_args_from_json(file: Path):
    with open(file, 'r') as f:
        args = argparse.Namespace(**json.loads(f.read()))
    return args
