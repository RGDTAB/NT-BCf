import time
import math
import numpy as np
import slangpy as spy
from src.output_parser import parse_texture_params

# Displays the decoded neural texture
# Since it uses Vulkan, this helps check that our BC1 textures are formatted properly
class DecoderWindow(spy.AppWindow):
    def __init__(self, args, app):
        super().__init__(app)

        # Load data from the passed file
        f = open(args.file, 'rb')
        data = f.read()
        f.close()

        # Reading texture parameters from the data buffer
        texture_params, texture_channels, latent_params, latent_offsets, mlp_offsets = parse_texture_params(data)

        self.width = texture_params[0]
        self.height = texture_params[1]
        self.texture_aspect = texture_params[0] / texture_params[1]
        self.mip_count = texture_params[3]
        self.var_a = texture_params[4]
        self.texture_count = max(len(texture_channels), 4)

        # Create a numpy array with each target texture's channel count
        # and offset bitpacked into a single value
        channel_offset = 0
        packed_texture_channels = np.zeros(32, dtype=np.uint32)
        for i, channels in enumerate(texture_channels):
            packed_texture_channels[i] = channels | (channel_offset << 3)
            channel_offset += channels
        self.packed_texture_channels = self.device.create_buffer(
            size=32 * 4,
            format=spy.Format.r32_uint,
            usage=spy.BufferUsage.shader_resource,
            label=f'packed_texture_channels',
            data=packed_texture_channels,
        )

        # Load neural texture data to the GPU
        self.load_latent_textures(latent_params, latent_offsets, data)
        self.create_samplers()
        self.load_mlp_parameters(texture_params, mlp_offsets, data)

        # Load the decoder window program
        channels = texture_params[2]
        mlp_width = texture_params[5]
        constants = f'export static const int MLP_WIDTH={mlp_width}; export static const int CHANNELS={channels};'
        window_program = self.device.load_program('shaders/decoder_window.slang', ['main'], constants)
        self.window_kernel = self.device.create_compute_kernel(window_program)

        self.render_texture = None # Window kernel output
        self.needs_rerender = True # Should rerender to self.render_texture
        self.mip_level = 0 # Texture mip level that is displayed
        self.show_latents = False # Should the latent textures be displayed?
        self.zoom = 1.0 # Window zoom factor
        self.view_offset = [0.0, 0.0] # XY coords of top left corner
        self.old_cursor_pos = None # Used to track mouse movement when panning
        self.lmb_down = False # Check if should pan texture

    # Loads the BC1 textures from the byte array to the GPU
    def load_latent_textures(self, latent_params, latent_offsets, data):
        self.latent_textures = []
        for i, t_param in enumerate(latent_params):
            # Allocate the texture
            self.latent_textures.append(self.device.create_texture(
                type=spy.TextureType.texture_2d,
                format=spy.Format.bc1_unorm,
                width=t_param[0],
                height=t_param[1],
                mip_count = t_param[2],
                usage=spy.TextureUsage.shader_resource,
                label=f't{i}',
            ))

            # Data gets loaded in mip by mip
            texture_data = np.frombuffer(data[latent_offsets[i]:latent_offsets[i+1]], dtype=np.uint64)
            mip_offset = 0
            for mip in range(t_param[2]):
                w = (t_param[0] >> mip) // 4
                h = (t_param[1] >> mip) // 4
                mip_size = w * h
                mip_data = texture_data[mip_offset:mip_offset + mip_size].copy()
                self.latent_textures[i].copy_from_numpy(
                    data=mip_data.copy(),
                    mip=mip,
                )

                mip_offset += mip_size

    # Create a bilinear and nearest neighbor sampler
    def create_samplers(self):
        self.bilinear_sampler = self.device.create_sampler(
            mag_filter=spy.TextureFilteringMode.linear,
            min_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
            address_w=spy.TextureAddressingMode.clamp_to_edge,
            label='bilinear_sampler',
        )
        self.nearest_sampler = self.device.create_sampler(
            mag_filter=spy.TextureFilteringMode.point,
            min_filter=spy.TextureFilteringMode.point,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
            address_w=spy.TextureAddressingMode.clamp_to_edge,
            label='nearest_sampler',
        )

    # Load decoder MLP parameters from the byte array to the GPU
    def load_mlp_parameters(self, texture_params, mlp_offsets, data):
        channels = texture_params[2]
        mlp_width = texture_params[5]

        weight0_offset, bias0_offset, weight1_offset, bias1_offset, bias1_end = mlp_offsets

        # Load weights and biases from the data bytearray as fp16 values
        weight0_data = np.frombuffer(data, dtype='<f2', count=12 * mlp_width, offset=weight0_offset).astype(np.float32)
        bias0_data = np.frombuffer(data, dtype='<f2', count=mlp_width, offset=bias0_offset).astype(np.float32)
        weight1_data = np.frombuffer(data, dtype='<f2', count=mlp_width * channels, offset=weight1_offset).astype(np.float32)
        bias1_data = np.frombuffer(data, dtype='<f2', count=channels, offset=bias1_offset).astype(np.float32)

        # Allocate the weight and bias buffers
        self.weight0_buffer = self.device.create_buffer(
            size = 12 * mlp_width * 4,
            format=spy.Format.r32_float,
            usage=spy.BufferUsage.shader_resource,
            label='weight_0',
            data=weight0_data,
        )
        self.bias0_buffer = self.device.create_buffer(
            size = mlp_width * 4,
            format=spy.Format.r32_float,
            usage=spy.BufferUsage.shader_resource,
            label='bias_0',
            data=bias0_data,
        )
        self.weight1_buffer = self.device.create_buffer(
            size = mlp_width * channels * 4,
            format=spy.Format.r32_float,
            usage=spy.BufferUsage.shader_resource,
            label='weight_1',
            data=weight1_data,
        )
        self.bias1_buffer = self.device.create_buffer(
            size = channels * 4,
            format=spy.Format.r32_float,
            usage=spy.BufferUsage.shader_resource,
            label='bias_1',
            data=bias1_data,
        )

    # Render the neural texture to the render texture
    def render_decoder_window(self, command_encoder):
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
                't0': self.latent_textures[0],
                't1': self.latent_textures[1],
                't2': self.latent_textures[2],
                't3': self.latent_textures[3],
                'bilinear_sampler': self.bilinear_sampler,
                'nearest_sampler': self.nearest_sampler,
                'var_a': self.var_a,
                'layer1': {
                    'weights': self.weight0_buffer,
                    'biases': self.bias0_buffer,
                },
                'layer2': {
                    'weights': self.weight1_buffer,
                    'biases': self.bias1_buffer,
                },
                'dst_texture': self.render_texture,
                'texture_channels': self.packed_texture_channels,
                'mip_level': self.mip_level,
                'show_latents': self.show_latents,
                'zoom': self.zoom,
                'view_offset': self.view_offset,
                'rows': rows,
                'columns': columns,
            },
            command_encoder = command_encoder,
        )

    # Reallocates the render texture to accomodate the new screen size
    # Note: Old one is freed automatically
    def realloc_render_texture(self, image):
        self.render_texture = self.device.create_texture(
            format=spy.Format.rgba16_float,
            width=image.width,
            height=image.height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label='render_texture',
        )
        self.needs_rerender = True

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if event.is_key_press():
            match event.key:
                case spy.KeyCode.l: # L - Show latent textures
                    self.show_latents = not self.show_latents
                    self.needs_rerender = True
                case spy.KeyCode.up: # Up - Show larger mip
                    self.mip_level = max(self.mip_level - 1, 0)
                    self.needs_rerender = True
                case spy.KeyCode.down: # Down - Show smaller mip
                    self.mip_level = min(self.mip_level + 1, self.mip_count - 1)
                    self.needs_rerender = True

    def on_mouse_event(self, event: spy.MouseEvent):
        # Zoom/Pan calculations require a render texture
        # There is a fraction of a second when we don't have a render texture initialized
        if self.render_texture is None:
            return

        # Pan
        if event.is_move() and self.lmb_down:
            diff = self.old_cursor_pos - event.pos
            # Positions are in pixels - not normalized!
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

        # Check if window has been resized, or if the program has just started
        if (self.render_texture is None
                or self.render_texture.width != image.width
                or self.render_texture.width != image.width):
            self.realloc_render_texture(image)

        # Render the encoder window if needed
        if self.needs_rerender:
            self.render_decoder_window(command_encoder)
            self.needs_rerender = False;

        # Prevent the FPS from skyrocketing
        time.sleep(0.001)

        # Copy the render texture to the AppWindow's surface texture
        command_encoder.blit(image, self.render_texture)


