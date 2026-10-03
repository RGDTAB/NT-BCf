import random
import math
import numpy as np
import time
import cv2
import json
import slangpy as spy
from enum import Enum
from pathlib import Path
from typing import Sequence
from src.args import parse_args, load_args_from_json
from src.input_loader import get_highest_mip, load_exr, InputTextureType
from src.slangpy.encoder import BCf1Encoder
from src.slangpy.emulated_file_dialog import *

class EncoderWindow(spy.AppWindow):
    # Subclass used to represent each texture and its mips
    class TextureItem():
        def __init__(self, texture_group: spy.ui.Widget, file: Path, mip_callback, type = InputTextureType.STANDARD):
            self.files = [str(file)]
            self.should_remove = False
            self.type = type

            # Calculate the maximum number of mips that this texture can support
            self.max_mips = 1
            try:
                if file.suffix != '.exr':
                    tex = cv2.imread(file)
                else:
                    tex = load_exr(file)

                self.width = tex.shape[0]
                self.height = tex.shape[1]
                self.max_mips = get_highest_mip(self.width, self.height)
            except:
                print(f'ERROR: There was an issue opening {file}', file=sys.stderr)
                self.should_remove = True

            # Group to store all the texture info
            self.group = spy.ui.Group(texture_group, file.name)

            # Should update when mip is added or type is changed
            self.should_update = False

            # Sets the texture type and sets the update flag
            def type_callback(value: int):
                self.type = InputTextureType(value)
                self.should_update = True

            texture_types = ['Standard', 'Normal Map', 'Single-Channel']
            spy.ui.ComboBox(
                self.group,
                'Texture Type',
                type.value,
                type_callback,
                texture_types,
            )

            # Each mip is just a text label with the file name
            self.mip_chain = []
            mip0 = spy.ui.Text(
                self.group,
                file.name
            )
            self.mip_chain.append(mip0)

            # Goes through the encoder window to open a file dialog
            def add_mip_callback():
                mip_callback(self)
            self.add_mip = spy.ui.Button(
                self.group,
                'Add Mip',
                add_mip_callback
            )
            # Mip pos is at the end of the chain - mips are added from the top down (0->N)
            self.new_mip_pos = self.group.child_index(self.add_mip)

            # Remove the smallest mip
            def remove_callback():
                # Remove the texture if on mip 0
                if len(self.mip_chain) == 1:
                    self.should_remove = True
                else:
                    # Remove mip from the UI
                    self.group.remove_child(self.mip_chain[len(self.mip_chain) - 1])
                    # Remove mip from lists
                    self.mip_chain.pop(len(self.mip_chain) - 1)
                    self.files.pop(len(self.files) - 1)

                    self.new_mip_pos -= 1
                    self.add_mip.enabled = True
                    self.should_update = True

            spy.ui.Button(
                self.group,
                'Remove Mip',
                remove_callback,
            )

        # Updates the number of mips being displayed
        def update_mip_chain(self, mip_count: int):
            # -1 means as many mips as possible
            if mip_count == -1:
                mip_count = self.max_mips

            if mip_count <= len(self.mip_chain):
                self.add_mip.enabled = False
            elif len(self.mip_chain) < self.max_mips:
                self.add_mip.enabled = True

            mip_count = min(mip_count, len(self.mip_chain))

            for i in range(mip_count):
                self.mip_chain[i].visible = True
            for i in range(mip_count, len(self.mip_chain)):
                self.mip_chain[i].visible = False

        # Add mip to chains - called by the encoder window
        def append_mip(self, file: Path):
            # Already have full chain
            if len(self.mip_chain) >= self.max_mips:
                return

            # Add to file location list
            self.files.append(str(file))

            # Add to UI
            mip = spy.ui.Text(None)
            mip.text = file.name
            self.group.add_child_at(mip, self.new_mip_pos)

            self.new_mip_pos += 1
            self.mip_chain.append(mip)

            self.should_update = True

            # Disable add mip button if chain is full
            if len(self.mip_chain) >= self.max_mips:
                self.add_mip.enabled = False

        # Return string representing the full mip chain
        def get_files(self):
            # Mip chains are parsed as a comma delimited string
            return ','.join(self.files)

    def __init__(self, args, app):
        super().__init__(app)

        # The args Namespace object is updated by the UI, and later passed to the encoder
        self.args = args

        # Initial window shader program is different than the usual one,
        # as we don't have any textures loaded yet to show
        window_program = self.device.load_program('shaders/empty_encoder.slang', ['main'])
        self.window_kernel = self.device.create_compute_kernel(window_program)

        self.init_ui()

        self.rerender_period = 10 # How many epochs before the neural texture should be rerendered
        self.render_texture = None # Window kernel output
        self.needs_rerender = True # Should rerender to self.render_texture
        self.render_mip_level = 0 # Texture mip level that is displayed
        self.show_loss = 0 # 0 - no loss, 1 - L1, 2 - L2
        self.show_texture = 0 # 0 - encoder output, 1 - target, 2 - latent textures
        self.zoom = 1.0 # Window zoom factor
        self.view_offset = [0.0, 0.0] # XY coords of top left corner
        self.old_cursor_pos = None # Used to track mouse movement when panning
        self.lmb_down = False # Check if should pan texture

        self.is_training = False # Is the encoder currently training

        self.add_texture_dialog = None # Add texture (mip 0) dialog
        self.add_mip_dialog = None # Add mip file dialog
        self.add_mip_source = None # Texture mip file dialog belongs to
        self.set_output_dialog = None # Output location file dialog
        self.load_json_dialog = None # Load json profile file dialog
        self.save_json_dialog = None # Save to json profile file dialog
        self.export_dialog = None # After encoder finished - export to file dialog

        self.results_window = None # Window showing PSNR values and such

        self.encoder = None # BCf1 Encoder

    # Are any of the emulated file dialogs open
    def is_file_dialog_open(self):
        return (
            self.add_texture_dialog is not None
            or self.add_mip_dialog is not None
            or self.set_output_dialog is not None
            or self.load_json_dialog is not None
            or self.save_json_dialog is not None
            or self.export_dialog is not None
        )

    # Create encoder and start training
    def start_training_callback(self):
        self.encoder = BCf1Encoder(self.args, self.device)

        self.texture_aspect = self.encoder.w / self.encoder.h
        self.texture_count = max(len(self.encoder.texture_channels), 4)
        # Create a numpy array with each target texture's channel count
        # and offset bitpacked into a single value
        channel_offset = 0
        packed_texture_channels = np.zeros(32, dtype=np.uint32)
        for i, channels in enumerate(self.encoder.texture_channels):
            packed_texture_channels[i] = channels | (channel_offset << 3)
            channel_offset += channels
        self.packed_texture_channels = self.device.create_buffer(
            size=32 * 4,
            format=spy.Format.r32_uint,
            usage=spy.BufferUsage.shader_resource,
            label=f'packed_texture_channels',
            data=packed_texture_channels,
        )

        # Compile the encoder window UI program
        constants = f'export static const int MLP_WIDTH={self.args.mlp_width}; export static const int CHANNELS={self.encoder.c};'
        window_program = self.device.load_program('shaders/encoder_window.slang', ['main'], constants)
        self.window_kernel = self.device.create_compute_kernel(window_program)

        # Disable start button - prevents accidental restarts
        self.start_button.enabled = False
        # Enable pause and stop buttons
        self.pause_button.enabled = True
        self.stop_button.enabled = True

        # Disable training parameter UI - prevent changes mid training
        self.training_parameters.enabled = False
        self.texture_group.enabled = False
        self.output_group.enabled = False

        self.is_training = True
        self.needs_rerender = True

    # Temporarily suspend training, or resume if already paused
    def pause_training_callback(self):
        if self.is_training:
            self.is_training = False
            self.pause_button.label = 'Resume'
        elif self.encoder is not None and not (self.encoder.complete or self.encoder.stopped):
            self.is_training = True
            self.pause_button.label = 'Pause'

    # Stop training,
    def stop_training_callback(self):
        # Enable start button
        self.start_button.enabled = True
        # Disable pause and stop buttons
        self.pause_button.enabled = False
        self.stop_button.enabled = False

        # Enable training parameter UI
        self.training_parameters.enabled = True
        self.texture_group.enabled = True
        self.output_group.enabled = True

        self.encoder.stopped = True
        self.is_training = False

    # Initialize UI for training parameters
    def init_ui(self):
        self.window = spy.ui.Window(self.screen, 'Configuration')

        self.progress_bar = spy.ui.ProgressBar(self.window, 0.0)
        self.current_psnr = spy.ui.Text(self.window, '0.0dB')

        # Start-Pause-Stop buttons
        self.start_button = spy.ui.Button(self.window, 'Start', self.start_training_callback)
        self.pause_button = spy.ui.Button(self.window, 'Pause', self.pause_training_callback)
        self.pause_button.enabled = False
        self.stop_button = spy.ui.Button(self.window, 'Stop', self.stop_training_callback)
        self.stop_button.enabled = False

        # Group for training parameters not related directly to the latent textures
        self.training_parameters = spy.ui.Group(self.window, 'Training Parameters')

        # Number of training epochs
        def training_epochs_callback(value: int):
            self.args.training_epochs = value
        self.training_epochs = spy.ui.InputInt(
            self.training_parameters,
            'Training epochs',
            self.args.training_epochs,
            training_epochs_callback
        )

        # MLP hidden layer width
        def mlp_width_callback(value: int):
            self.args.mlp_width = value
        self.mlp_width = spy.ui.InputInt(
            self.training_parameters,
            'MLP hidden layer width',
            self.args.mlp_width,
            mlp_width_callback,
        )

        # Should optimization be based on L1 or L2?
        def loss_callback(value: int):
            self.args.l1 = value == 0
        loss_options = ['L1', 'L2']
        selected_loss = 0 if self.args.l1 else 1
        self.loss = spy.ui.ComboBox(
            self.training_parameters,
            'Loss',
            selected_loss,
            loss_callback,
            loss_options
        )

        # Learning rate used for the latent textures
        def texture_lr_callback(value: float):
            self.args.texture_lr = value
        self.texture_lr = spy.ui.InputFloat(
            self.training_parameters,
            'Latent texture learning rate',
            self.args.texture_lr,
            texture_lr_callback,
        )

        # Learning rate used for the MLP weights and biases
        def mlp_lr_callback(value: float):
            self.args.mlp_lr = value
        self.mlp_lr = spy.ui.InputFloat(
            self.training_parameters,
            'MLP parameter learning rate',
            self.args.mlp_lr,
            mlp_lr_callback,
        )

        # Should cosine annealing be used?
        def cos_annealing_callback(value: bool):
            self.args.cos = value
        self.half_res = spy.ui.CheckBox(
            self.training_parameters,
            'Cosine annealing',
            self.args.cos,
            cos_annealing_callback,
        )

        # Width and height of each grid batch
        def batch_size_callback(value: int):
            self.args.batch_size = value
        self.batch_size = spy.ui.InputInt(
            self.training_parameters,
            'Batch size',
            self.args.batch_size,
            batch_size_callback,
        )

        # Neural texture parameters that are specific to the latent textures
        self.texture_group = spy.ui.Group(self.window, 'Target Textures')

        def generate_mips_callback(value):
            self.args.generate_mips = value
        self.generate_mips = spy.ui.CheckBox(self.texture_group, 'Generate missing mips', self.args.generate_mips, generate_mips_callback)

        def half_res_callback(value: bool):
            self.args.half_resolution = value
        self.half_res = spy.ui.CheckBox(
            self.training_parameters,
            'Half resolution latents',
            self.args.half_resolution,
            half_res_callback,
        )

        # Format used - BCf1A or BCf1B
        def format_callback(value: int):
            self.args.format = 'BCf1A' if value == 0 else 'BCf1B'
        format_options = ['Variant A', 'Variant B']
        selected_format = 0 if self.args.format == 'BCf1A' else 1
        self.format = spy.ui.ComboBox(
            self.training_parameters,
            'Neural texture format',
            selected_format,
            format_callback,
            format_options
        )

        # Set the maximum number of mips that must be learned
        def max_mips_callback(value):
            if value < -1:
                # -1 is the sentinel for 'as many as possible', other negatives are invalid
                self.max_mips.value = -1
                self.args.max_mips = -1
            elif value == 0:
                # The max mips can't be 0, just > 1 or -1
                if self.args.max_mips == -1:
                    self.max_mips.value = 1
                    self.args.max_mips = 1
                else:
                    self.max_mips.value = -1
                    self.args.max_mips = -1
            else:
                # Don't bother clamping to the actual maximum, not worth the effort
                self.args.max_mips = value
        self.max_mips = spy.ui.InputInt(self.texture_group, 'Mip count (-1 to use the maximum amount)', self.args.max_mips, max_mips_callback)

        # Group for all of the texture files
        self.texture_sources = spy.ui.Group(self.texture_group, 'Texture Files')
        self.texture_array = []

        # Add all of the passed textures to the UI
        self.args.standard_tex = [] if self.args.standard_tex is None else self.args.standard_tex
        self.args.normal_map = [] if self.args.normal_map is None else self.args.normal_map
        self.args.single_channel = [] if self.args.single_channel is None else self.args.single_channel
        for texture in self.args.standard_tex:
            chain = texture.split(',')
            mip0_path = Path(chain[0]).absolute()
            mip0 = self.TextureItem(self.texture_sources, mip0_path, self.add_mip_callback, InputTextureType.STANDARD)
            for path in chain[1:]:
                path = Path(path).absolute()
                mip0.append_mip(path)
            self.texture_array.append(mip0)
        for texture in self.args.normal_map:
            chain = texture.split(',')
            mip0_path = Path(chain[0]).absolute()
            mip0 = self.TextureItem(self.texture_sources, mip0_path, self.add_mip_callback, InputTextureType.NORMAL_MAP)
            for path in chain[1:]:
                path = Path(path).absolute()
                mip0.append_mip(path)
            self.texture_array.append(mip0)
        for texture in self.args.single_channel:
            chain = texture.split(',')
            mip0_path = Path(chain[0]).absolute()
            mip0 = self.TextureItem(self.texture_sources, mip0_path, self.add_mip_callback, InputTextureType.SINGLE_CHANNEL)
            for path in chain[1:]:
                path = Path(path).absolute()
                mip0.append_mip(path)
            self.texture_array.append(mip0)

        if len(self.texture_array) == 0:
            self.start_button.enabled = False
        else:
            self.start_button.enabled = True

        # Open a file dialog for opening an image file
        def add_texture_callback():
            if self.is_file_dialog_open():
                return

            filters = [
                spy.platform.FileDialogFilter('PNG', '*.png'),
                spy.platform.FileDialogFilter('JPEG', '*.jpg,*.jpeg'),
                spy.platform.FileDialogFilter('Webp', '*.webp'),
                spy.platform.FileDialogFilter('BMP', '*.bmp'),
                spy.platform.FileDialogFilter('AVIF', '*.avif'),
                spy.platform.FileDialogFilter('OpenEXR', '*.exr'),
                spy.platform.FileDialogFilter('Radiance HDR', '*.hdr,*.pic'),
            ]

            file_response = open_file_dialog(self.window, filters)
            if type(file_response) is EmulatedFileDialog:
                self.add_texture_dialog = file_response
            elif file_response is not None:
                self.add_texture(file_response)
        self.add_texture_button = spy.ui.Button(self.texture_group, 'Add Texture(s)', add_texture_callback)

        # Group for the output
        self.output_group = spy.ui.Group(self.window, 'Output')
        self.output_location = spy.ui.Text(self.output_group, self.args.output if self.args.output is not None else 'No output set!')

        # Set the file that the neural texture will be exported to
        def output_location_callback():
            if self.is_file_dialog_open():
                return

            file_response = save_file_dialog(self.window, [])
            if type(file_response) is EmulatedFileDialog:
                self.set_output_dialog = file_response
            elif file_response is not None:
                self.set_output_location(file_response)
        self.set_output_location_button = spy.ui.Button(self.output_group, '...', output_location_callback)

        # Save the neural textures parameters to JSON for later training
        def save_to_json_callback():
            if self.is_file_dialog_open():
                return

            file_response = save_file_dialog(self.window, [])
            if type(file_response) is EmulatedFileDialog:
                self.save_json_dialog = file_response
            elif file_response is not None:
                self.save_json(file_response)
        self.save_json_button = spy.ui.Button(self.output_group, 'Save arguments to JSON', save_to_json_callback)

        # Load neural texture parameters from a JSON file
        def load_from_json_callback():
            if self.is_file_dialog_open():
                return

            file_response = open_file_dialog(self.window, [])
            if type(file_response) is EmulatedFileDialog:
                self.load_json_dialog = file_response
            elif file_response is not None:
                self.load_json(file_response)
        self.load_json_button = spy.ui.Button(self.output_group, 'Load arguments from JSON', load_from_json_callback)

    # Dump arguments into a JSON file
    def save_json(self, file: Path):
        with open(file, 'w') as f:
            if hasattr(self.args, 'json'):
                delattr(self.args, 'json')

            f.write(json.dumps(vars(self.args), indent=4))

    # Load args from a json file
    def load_json(self, file: Path):
        self.screen.remove_all_children()
        self.args = load_args_from_json(file)
        self.init_ui()

    # Update args list to include textures to include all textures and their mips
    def update_texture_args(self):
        if len(self.texture_array) == 0:
            self.start_button.enabled = False
        else:
            self.start_button.enabled = True
        self.args.standard_tex = []
        self.args.normal_map = []
        self.args.single_channel = []
        for texture in self.texture_array:
            match texture.type:
                case InputTextureType.STANDARD:
                    self.args.standard_tex.append(texture.get_files())
                case InputTextureType.NORMAL_MAP:
                    self.args.normal_map.append(texture.get_files())
                case InputTextureType.SINGLE_CHANNEL:
                    self.args.single_channel.append(texture.get_files())

    # Creates a UI window to show the PSNR values and time spent on encoding
    def display_encoder_results(self):
        if self.results_window is not None:
            self.results_window.show()
            return

        # Create results window
        self.results_window = spy.ui.Window(self.screen, 'Results')

        # Duration section
        spy.ui.Text(self.results_window, f'Total CPU Time: {self.encoder.total_cpu_time:.4f}s')
        spy.ui.Text(self.results_window, f'Epoch Mean Duration: {self.encoder.epoch_mean_time:.4f}ms')
        spy.ui.Text(self.results_window, '------------------')

        # PSNR section
        spy.ui.Text(self.results_window, f'Total PSNR: {self.encoder.cumulative_psnr:.4f}dB')
        spy.ui.Text(self.results_window, '------------------')
        for mip, psnr in enumerate(self.encoder.mip_psnrs):
            spy.ui.Text(self.results_window, f'Mip {mip} PSNR: {psnr:.4f}dB')

        # If already saved to a file, then say so
        if self.encoder.args.output is not None:
            spy.ui.Text(self.results_window, f'Exported to {self.encoder.args.output}')

        # Open a file dialog to chose where to export the texture to
        def export_button_callback():
            if self.is_file_dialog_open():
                return

            file_response = save_file_dialog(self.screen, [])
            if type(file_response) is EmulatedFileDialog:
                self.export_dialog = file_response
            elif file_response is not None:
                self.encoder.export_neural_texture(file_response)
        spy.ui.Button(self.results_window, 'Export to...', export_button_callback)


    # Removes the results window when restarting the encoder
    def hide_encoder_results(self):
        if self.results_window is not None:
            self.results_window.remove_all_children()
            self.results_window.close()
            self.results_window = None

    # Callback to add a mip to a specific texture / mip chain
    def add_mip_callback(self, source: TextureItem):
        if self.is_file_dialog_open():
            return

        filters = [
            spy.platform.FileDialogFilter('PNG', '*.png'),
            spy.platform.FileDialogFilter('JPEG', '*.jpg,*.jpeg'),
            spy.platform.FileDialogFilter('Webp', '*.webp'),
            spy.platform.FileDialogFilter('BMP', '*.bmp'),
            spy.platform.FileDialogFilter('AVIF', '*.avif'),
            spy.platform.FileDialogFilter('OpenEXR', '*.exr'),
            spy.platform.FileDialogFilter('Radiance HDR', '*.hdr,*.pic'),
        ]
        file_response = open_file_dialog(self.window, filters)
        if type(file_response) is EmulatedFileDialog:
            self.add_mip_dialog = file_response
            self.add_mip_source = source
        elif file_response is not None:
            source.append_mip(file_response)

    def update_ui(self):
        # Don't allow the user to close the settings window
        self.window.show()

        if self.encoder and self.encoder.psnr is not None and not self.encoder.complete:
            self.current_psnr.text = f'{self.encoder.psnr:.4f}dB'
        else:
            self.current_psnr.text = '0.0dB'

        # Check on emulated add texture dialog
        if self.add_texture_dialog is not None:
            self.add_texture_dialog.render()
            if self.add_texture_dialog.complete:
                if self.add_texture_dialog.result is not None:
                    for file in self.add_texture_dialog.result:
                        self.add_texture(file)
                self.add_texture_dialog = None

        # Check on emulated add mip dialog
        if self.add_mip_dialog is not None:
            self.add_mip_dialog.render()
            if self.add_mip_dialog.complete:
                if self.add_mip_dialog.result is not None:
                    for file in self.add_mip_dialog.result:
                        self.add_mip_source.append_mip(file)
                self.add_mip_dialog = None

        # Check on emulated set output dialog
        if self.set_output_dialog is not None:
            self.set_output_dialog.render()
            if self.set_output_dialog.complete:
                if self.set_output_dialog.result is not None:
                    self.set_output_location(self.set_output_dialog.result)
                self.set_output_dialog = None

        # Check on emulated save JSON dialog
        if self.save_json_dialog is not None:
            self.save_json_dialog.render()
            if self.save_json_dialog.complete:
                if self.save_json_dialog.result is not None:
                    self.save_json(self.save_json_dialog.result)
                self.save_json_dialog = None

        # Check on emulated load JSON dialog
        if self.load_json_dialog is not None:
            self.load_json_dialog.render()
            if self.load_json_dialog.complete:
                if self.load_json_dialog.result is not None:
                    self.load_json(self.load_json_dialog.result[0])
                self.load_json_dialog = None

        # Check on emulated export dialog
        if self.export_dialog is not None:
            self.export_dialog.render()
            if self.export_dialog.complete:
                if self.export_dialog.result is not None:
                    self.encoder.export_neural_texture(self.export_dialog.result)
                self.export_dialog = None

        # Show or hide the results from the encoder
        if self.encoder is not None and self.encoder.complete:
            self.display_encoder_results()
        else:
            self.hide_encoder_results()

        # Iterate over the textures to see if the argument lists need to be updated
        should_update = False
        remove_at = [] 
        for i, texture in enumerate(self.texture_array):
            if texture.should_remove:
                remove_at.append(i - len(remove_at))
            else:
                texture.update_mip_chain(self.args.max_mips)
            if texture.should_update:
                should_update = True

        for loc in remove_at:
            self.texture_sources.remove_child_at(loc)
            self.texture_array.pop(loc)
        if len(remove_at) != 0 or should_update:
            self.update_texture_args()

    # Add a texture to the texture array
    def add_texture(self, file: Path, type = InputTextureType.STANDARD):
        self.texture_array.append(self.TextureItem(self.texture_sources, file, self.add_mip_callback, type))
        self.update_texture_args()

    # Updates the file that the neural texture will be exported to after compression
    def set_output_location(self, file: Path):
        self.output_location.text = str(file)
        self.args.output = str(file)

    # Reallocates the render texture to accomodate the new screen size
    # Note: Old one is freed automatically
    def realloc_render_texture(self, image):
        was_none = self.render_texture is None

        self.render_texture = self.device.create_texture(
            format=spy.Format.rgba16_float,
            width=image.width,
            height=image.height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label='render_texture',
        )

        # Center the settings menu on the top of the screen
        if was_none:
            width = image.width
            self.window.position = spy.math.float2(10,10)
            self.window.size = spy.math.float2(width - 20, self.window.size.y)

        self.needs_rerender = True

    # Renders the blank checkerboard
    def render_empty_encoder(self, command_encoder):
        self.window_kernel.dispatch(
            thread_count = [self.render_texture.width, self.render_texture.width, 1],
            vars = {
                'dst_texture': self.render_texture,
            },
            command_encoder = command_encoder,
        )

    # Renders the active encoder's results to self.render_texture
    def render_encoder_window(self, command_encoder):
        if self.encoder is None:
            self.render_empty_encoder(command_encoder)
            return

        window_aspect = self.render_texture.width / self.render_texture.height

        # Determine the best number of rows and columns to use while matching the screen's width
        rows = 0
        columns = 0
        while rows < self.texture_count:
            rows += 1
            columns = max(math.ceil(window_aspect * rows), math.ceil(self.texture_count / rows))
            combined_aspect = (columns / rows) * self.texture_aspect

            next_columns = max(math.ceil(window_aspect * (rows + 1)), math.ceil(self.texture_count / (rows + 1)))
            next_aspect = (next_columns / (rows + 1)) * self.texture_aspect

            if next_aspect < window_aspect or next_aspect >= combined_aspect:
                break

        # Render to the render texture
        self.window_kernel.dispatch(
            thread_count = [self.render_texture.width, self.render_texture.width, 1],
            vars = {
                'texture': {
                    't0': self.encoder.latent_texture_parameter(0),
                    't1': self.encoder.latent_texture_parameter(1),
                    't2': self.encoder.latent_texture_parameter(2),
                    't3': self.encoder.latent_texture_parameter(3),
                    'layer1': {
                        'weights': self.encoder.weight0_buffer,
                        'weightsGrad': self.encoder.weight0_grad,
                        'biases': self.encoder.bias0_buffer,
                        'biasesGrad': self.encoder.bias0_grad,
                    },
                    'layer2': {
                        'weights': self.encoder.weight1_buffer,
                        'weightsGrad': self.encoder.weight1_grad,
                        'biases': self.encoder.bias1_buffer,
                        'biasesGrad': self.encoder.bias1_grad,
                    },
                    'var_a': self.encoder.var_a,
                },
                'target': {
                    'texture':self.encoder.target_texture_array,
                    'sampler':self.encoder.target_sampler,
                },
                'mip_level': self.render_mip_level,
                'dst_texture': self.render_texture,
                'texture_channels': self.packed_texture_channels,
                'show_loss': self.show_loss,
                'show_texture': self.show_texture,
                'zoom': self.zoom,
                'view_offset': self.view_offset,
                'rows': rows,
                'columns': columns,
            },
            command_encoder = command_encoder,
        )

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if event.is_key_press():
            match event.key:
                case spy.KeyCode.key1: # 1 - Show L1 loss
                    self.show_loss = 1
                    self.needs_rerender = True
                case spy.KeyCode.key2: # 2 - Show L2 loss
                    self.show_loss = 2
                    self.needs_rerender = True
                case spy.KeyCode.o: # O - Show reconstructed textures
                    self.show_texture = 0
                    self.show_loss = 0
                    self.needs_rerender = True
                case spy.KeyCode.i: # I - Show target/input textures
                    self.show_texture = 1
                    self.show_loss = 0
                    self.needs_rerender = True
                case spy.KeyCode.l: # L - Show latent textures
                    self.show_texture = 2
                    self.show_loss = 0
                    self.needs_rerender = True
                case spy.KeyCode.up: # Up - Show larger mip
                    self.render_mip_level = max(self.render_mip_level - 1, 0)
                    self.needs_rerender = True
                case spy.KeyCode.down: # Down - Show smaller mip
                    if self.encoder is not None:
                        self.render_mip_level = min(self.render_mip_level + 1, self.encoder.mip_count - 1)
                        self.needs_rerender = True

    def on_mouse_event(self, event: spy.MouseEvent):
        # Zoom/Pan calculations require a render texture
        # There is a fraction of a second when we don't have a render texture initialized
        if self.render_texture is None:
            return

        # Pan
        if event.is_move() and self.lmb_down:
            diff = self.old_cursor_pos - event.pos
            # Positions are in pixels - not normalized
            diff.x /= self.render_texture.width * self.zoom
            diff.y /= self.render_texture.height * self.zoom
            self.view_offset[0] += diff.x
            self.view_offset[1] += diff.y
            # Clamp view offset to keep from panning away from textures
            self.view_offset[0] = min(max(self.view_offset[0], 0.0), 1 - 1 / self.zoom)
            self.view_offset[1] = min(max(self.view_offset[1], 0.0), 1 - 1 / self.zoom)

            self.old_cursor_pos = event.pos
            self.needs_rerender = True
        # Record LMB down
        elif event.is_button_down() and event.button == spy.MouseButton.left:
            self.lmb_down = True
            self.old_cursor_pos = event.pos
        # Record LMB up
        elif event.is_button_up() and event.button == spy.MouseButton.left:
            self.lmb_down = False
        # Zoom
        elif event.is_scroll():
            # Record the cursor pos - we'll zoom into / away from it
            pos = event.pos * 2
            pos.x /= self.render_texture.width
            pos.y /= self.render_texture.height

            zoom_factor = 0.5
            zoom_prev = self.zoom
            self.zoom = max(self.zoom + event.scroll.y * zoom_factor, 1.0)
            # This will keep what the mouse hovers over in the same exact position
            diff = (1 / zoom_prev - 1 / self.zoom) / 2
            self.view_offset[0] += diff * pos.x
            self.view_offset[1] += diff * pos.y

            # Clamp view offset to keep from panning away from textures
            self.view_offset[0] = min(max(self.view_offset[0], 0.0), 1 - 1 / self.zoom)
            self.view_offset[1] = min(max(self.view_offset[1], 0.0), 1 - 1 / self.zoom)
            self.needs_rerender = True

    def render(self, render_context: spy.AppWindow.RenderContext):
        image = render_context.surface_texture
        command_encoder = render_context.command_encoder

        self.update_ui();

        # Check if window has been resized, or if the program has just started
        if (self.render_texture is None
                or self.render_texture.width != image.width
                or self.render_texture.width != image.width):
            self.realloc_render_texture(image)

        # Update encoder
        if self.encoder is not None and self.is_training:
            self.encoder.train_epoch()
            self.progress_bar.fraction = self.encoder.epoch / self.args.training_epochs
            if self.encoder.epoch % self.rerender_period == 0 and self.encoder.epoch <= self.args.training_epochs:
                self.needs_rerender = True
            if self.encoder.epoch > self.args.training_epochs:
                self.stop_training_callback()

        # Rerender the encoder window if needed
        if self.needs_rerender:
            self.render_encoder_window(command_encoder)
            self.needs_rerender = False;

        # Prevent the FPS from skyrocketing
        if not self.is_training:
            time.sleep(0.001)

        # Copy the render texture to the AppWindow's surface texture
        command_encoder.blit(image, self.render_texture)

