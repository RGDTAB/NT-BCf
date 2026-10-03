import random
import sys
import math
import numpy as np
import slangpy as spy
from src.args import parse_args
from src.input_loader import bulk_load_textures
from src.bc1_export import export_bc1

# Surprised numpy doesn't have a built-in sigmoid
def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

# SlangPy based encoder that encapsulates all the textures and shaders needed to optimize a neural texture
class BCf1Encoder():
    def __init__(self, args, device: spy.Device):
        super().__init__()
        self.device = device
        self.args = args

        if self.args.output is None and not self.args.quiet and self.args.no_gui:
            print('ERROR: No output file set! Exiting...', file=sys.stderr)
            exit()

        # Load target textures as numpy arrays
        mip_chain, self.texture_channels = bulk_load_textures(self.args.standard_tex, self.args.normal_map, self.args.single_channel, self.args.max_mips, self.args.generate_mips)
        
        # Load argument data into member variable for convenience
        self.w = mip_chain[0].shape[0]
        self.h = mip_chain[0].shape[1]
        self.mip_count = len(mip_chain)
        self.c = mip_chain[0].shape[2]
        self.var_a = self.args.format == 'BCf1A'

        # Allocate the target texture array and destination texture array on the device
        self.allocate_target_textures(mip_chain)
        self.allocate_dst_textures()

        # Define the resolution for each latent texture
        latent_w = self.w if not self.args.half_resolution else max(self.w // 2, 4)
        latent_h = self.h if not self.args.half_resolution else max(self.h // 2, 4)
        latent_mips = self.mip_count if not self.args.half_resolution else max(self.mip_count - 1, 1)
        self.t_dims = [(latent_w, latent_h, latent_mips, 0)]
        if self.var_a:
            self.t_dims.append((latent_w, latent_h, latent_mips, 0))
            self.t_dims.append((max(latent_w // 2, 4), max(latent_h // 2, 4), max(latent_mips - 1, 1), 1))
            self.t_dims.append((max(latent_w // 2, 4), max(latent_h // 2, 4), max(latent_mips - 1, 1), 1))
        else:
            self.t_dims.append((max(latent_w // 2, 4), max(latent_h // 2, 4), max(latent_mips - 1, 1), 1))
            self.t_dims.append((max(latent_w // 4, 4), max(latent_h // 4, 4), max(latent_mips - 2, 1), 2))
            self.t_dims.append((max(latent_w // 8, 4), max(latent_h // 8, 4), max(latent_mips - 3, 1), 3))

        # Allocate the latent textures and all of their necessary buffers
        self.allocate_latent_textures(self.t_dims)
        self.gen_mip_offsets(self.t_dims)

        # Allocate buffers for the weights and biases of the decoder MLP
        self.allocate_mlp_buffers(self.args.mlp_width, self.c)

        # Create the loss buffer used to report optimization progress
        self.loss_buffer = self.device.create_buffer(
            size = 4,
            format = spy.Format.r32_float,
            usage = spy.BufferUsage.unordered_access,
            label = 'loss_buffer'
        )

        gpu_spec = f'{self.device.info.adapter_name} ({self.device.info.api_name})'

        # Double check to make sure that the device supports atomic floats
        if self.device.features.count(spy.Feature.atomic_float) == 0 or self.device.capabilities.count('SPV_EXT_shader_atomic_float_add') == 0:
            print(f'ERROR: {gpu_spec} doesn\'t support atomic operations on floating point values', file=sys.stderr)
            exit()

        # Load the compute kernels used during optimization
        constants = f'export static const int MLP_WIDTH={self.args.mlp_width}; export static const int CHANNELS={self.c};'
        grad_program = self.device.load_program('shaders/learn_grad.slang', ['LearnGradient'], constants)
        self.learn_grad_kernel = self.device.create_compute_kernel(grad_program)

        step_e_program = self.device.load_program('shaders/step_e.slang', ['GradientStepTextureE'])
        self.step_texture_e_kernel = self.device.create_compute_kernel(step_e_program)

        step_a_program = self.device.load_program('shaders/step_a.slang', ['GradientStepTextureA'])
        self.step_texture_a_kernel = self.device.create_compute_kernel(step_a_program)

        step_buf_program = self.device.load_program('shaders/step_buffer.slang', ['GradientStepBuffer'])
        self.step_buffer_kernel = self.device.create_compute_kernel(step_buf_program)

        decode_program = self.device.load_program('shaders/decoder.slang', ['main'], constants)
        self.decode_kernel = self.device.create_compute_kernel(decode_program)

        self.epoch = 1 # Current training epoch
        self.queries = self.device.create_query_pool( # Timestamp query for determining mean epoch duration
            spy.QueryType.timestamp,
            self.args.training_epochs * 2
        )
        self.t = spy.Timer() # Timer for determining CPU time
        self.texture_lr = self.args.texture_lr # Latent texture learning rate - can be altered by scheduler
        self.mlp_lr = self.args.mlp_lr # MLP learning rate - can be altered by scheduler
        self.stopped = False # UI helper - was the encoder stopped before it could complete?
        self.complete = False # UI helper - did the encoder finish its task?
        self.psnr = None # PSNR last calculated during training - often higher than final result

    # Allocate the target textures and copy the numpy data to them,
    # as well as creating the linear sampler for it
    def allocate_target_textures(self, mip_chains):
        # Create the bilinear sampler
        self.target_sampler = self.device.create_sampler(
            mag_filter=spy.TextureFilteringMode.linear,
            min_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
            address_w=spy.TextureAddressingMode.clamp_to_edge,
            label='target_sampler',
        )
        # Create the texture array on the device. Each channel gets its own array entry
        self.target_texture_array = self.device.create_texture(
            type=spy.TextureType.texture_2d_array,
            format=spy.Format.r32_float,
            width = self.w,
            height = self.h,
            mip_count = self.mip_count,
            array_length = self.c,
            usage = spy.TextureUsage.shader_resource,
            sampler = self.target_sampler,
            label='target_texture_array',
        )

        # Copy over numpy data from mip_chains
        for mip in range(self.mip_count):
            for layer in range(self.c):
                data = np.ascontiguousarray(mip_chains[mip][:,:,layer])
                self.target_texture_array.copy_from_numpy(
                    data=data,
                    layer = layer,
                    mip = mip,
                )

    # Create the destination textures for the final PSNR report
    def allocate_dst_textures(self):
        self.dst_texture_array = self.device.create_texture(
            type=spy.TextureType.texture_2d_array,
            format=spy.Format.r32_float,
            width = self.w,
            height = self.h,
            mip_count = self.mip_count,
            array_length = self.c,
            usage = spy.TextureUsage.unordered_access,
            label='dst_texture_array',
        )

        # Create the UAVs used to access each mip
        self.dst_texture_uavs = []
        for mip in range(self.target_texture_array.mip_count):
            self.dst_texture_uavs.append(self.dst_texture_array.create_view(
                mip=mip,
                mip_count=1,
                label=f'dst_texture_uav_{mip}',
            ))


    # Create buffers containing the element offset of each mip for each latent texture
    def gen_mip_offsets(self, t_dims):
        self.mip_offsets = []
        self.dev_mip_offsets = []

        for i, dim in enumerate(t_dims):
            cur_offset = 0
            t_mip_offsets = []
            for j in range(dim[2]):
                t_mip_offsets.append(cur_offset)
                w = dim[0] >> j
                h = dim[1] >> j
                cur_offset += w * h

            self.mip_offsets.append(np.asarray(t_mip_offsets).astype(np.uint32))
            self.dev_mip_offsets.append(self.device.create_buffer(
                element_count=self.mip_offsets[i].size,
                struct_size=4,
                format=spy.Format.r32_uint,
                usage=spy.BufferUsage.shader_resource,
                label=f'mip_offsets_{i}',
                data=self.mip_offsets[i],
            ))


    # Allocate the latent textures, their UAVs, and associated buffers
    def allocate_latent_textures(self, t_dims):
        self.t_e0 = []
        self.t_e1 = []
        self.t_a = []
        self.t_e0_uav = []
        self.t_e1_uav = []
        self.t_a_uav = []
        self.t_e0_grad = []
        self.t_e1_grad = []
        self.t_a_grad = []
        self.t_e0_adam = []
        self.t_e1_adam = []
        self.t_a_adam = []

        for i, dim in enumerate(t_dims):
            # Get the size in pixels for the entire mip chain
            size = 0
            for j in range(dim[2]):
                size += (dim[0] >> j) * (dim[1] >> j)

            # Allocate the textures themselves
            self.t_e0.append(self.device.create_texture(
                type=spy.TextureType.texture_2d,
                format=spy.Format.rgba32_float,
                width=dim[0] // 4,
                height=dim[1] // 4,
                mip_count = dim[2],
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label=f't{i}_e0',
            ))
            self.t_e1.append(self.device.create_texture(
                type=spy.TextureType.texture_2d,
                format=spy.Format.rgba32_float,
                width=dim[0] // 4,
                height=dim[1] // 4,
                mip_count=dim[2],
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label=f't{i}_e1',
            ))
            self.t_a.append(self.device.create_texture(
                format=spy.Format.r32_float,
                width = dim[0],
                height = dim[1],
                mip_count = dim[2],
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label=f't{i}_a',
            ))

            # Initialize the textures with random data [0, 1)
            for mip in range(self.t_e0[i].mip_count):
                w = self.t_e0[i].get_mip_width(mip)
                h = self.t_e0[i].get_mip_height(mip)
                e0_data = np.random.rand(w,h,4)
                e1_data = np.random.rand(w,h,4)
                self.t_e0[i].copy_from_numpy(e0_data.astype(np.float32), mip=mip)
                self.t_e1[i].copy_from_numpy(e1_data.astype(np.float32), mip=mip)
                w = self.t_a[i].get_mip_width(mip)
                h = self.t_a[i].get_mip_height(mip)
                a_data = np.random.rand(w,h)
                self.t_a[i].copy_from_numpy(a_data.astype(np.float32), mip=mip)

            # Create UAVs for each mip
            t_e0_uav = []
            t_e1_uav = []
            t_a_uav = []
            for j in range(dim[2]):
                t_e0_uav.append(self.t_e0[i].create_view(
                    mip = j,
                    mip_count = 1,
                    label=f't{i}_e0_uav_{j}',
                ))
                t_e1_uav.append(self.t_e1[i].create_view(
                    mip = j,
                    mip_count = 1,
                    label=f't{i}_e1_uav_{j}',
                ))
                t_a_uav.append(self.t_a[i].create_view(
                    mip = j,
                    mip_count = 1,
                    label=f't{i}_a_uav_{j}',
                ))
            self.t_e0_uav.append(t_e0_uav)
            self.t_e1_uav.append(t_e1_uav)
            self.t_a_uav.append(t_a_uav)

            # Allocate the gradient buffers
            self.t_e0_grad.append(self.device.create_buffer(
                size=size,
                format=spy.Format.rgba32_float,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                label=f't{i}_e0_grad',
            ))
            self.t_e1_grad.append(self.device.create_buffer(
                size=size,
                format=spy.Format.rgba32_float,
                usage=spy.BufferUsage.shader_resource |spy.BufferUsage.unordered_access,
                label=f't{i}_e1_grad',
            ))
            self.t_a_grad.append(self.device.create_buffer(
                size=size * 4,
                format=spy.Format.r32_float,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                label=f't{i}_a_grad',
            ))

            # Allocate the adam state buffers
            self.t_e0_adam.append(self.device.create_buffer(
                size=size * 2,
                format=spy.Format.rgba32_float,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                label=f't{i}_e0_adam',
            ))
            self.t_e1_adam.append(self.device.create_buffer(
                size=size * 2,
                format=spy.Format.rgba32_float,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                label=f't{i}_e1_adam',
            ))
            self.t_a_adam.append(self.device.create_buffer(
                size=size * 8,
                format=spy.Format.rg32_float,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                label=f't{i}_a_adam',
            ))


    # Allocate all the buffers for the MLP decoder
    def allocate_mlp_buffers(self, mlp_width: int, channels: int):
        weight0_data = np.random.rand(12 * mlp_width).astype(np.float32)
        weight1_data = np.random.rand(mlp_width * channels).astype(np.float32)
        bias0_data = np.random.rand(mlp_width).astype(np.float32)
        bias1_data = np.random.rand(channels).astype(np.float32)

        # He / Kaiming uniform initialization
        bound0 = (1.0 / math.sqrt(12))
        bound1 = (1.0 / math.sqrt(mlp_width))
        weight0_data = (weight0_data * bound0 * 2) - bound0
        weight1_data = (weight1_data * bound1 * 2) - bound1
        bias0_data = (bias0_data * bound0 * 2) - bound0
        bias1_data = (bias1_data * bound1 * 2) - bound1

        # Allocate the weight and bias buffers
        self.weight0_buffer = self.device.create_buffer(
            size = 12 * mlp_width * 4,
            format=spy.Format.r32_float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label='weight_0',
            data=weight0_data,
        )
        self.bias0_buffer = self.device.create_buffer(
            size = mlp_width * 4,
            format=spy.Format.r32_float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label='bias_0',
            data=bias0_data,
        )
        self.weight1_buffer = self.device.create_buffer(
            size = mlp_width * channels * 4,
            format=spy.Format.r32_float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label='weight_1',
            data=weight1_data,
        )
        self.bias1_buffer = self.device.create_buffer(
            size = channels * 4,
            format=spy.Format.r32_float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label='bias_1',
            data=bias1_data,
        )

        # Allocate the gradient buffers
        self.weight0_grad = self.device.create_buffer(
            size = 12 * mlp_width * 4,
            format=spy.Format.r32_float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label='weight_0_grad',
        )
        self.bias0_grad = self.device.create_buffer(
            size = mlp_width * 4,
            format=spy.Format.r32_float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label='bias_0_grad',
        )
        self.weight1_grad = self.device.create_buffer(
            size = mlp_width * channels * 4,
            format=spy.Format.r32_float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label='weight_1_grad',
        )
        self.bias1_grad = self.device.create_buffer(
            size = channels * 4,
            format=spy.Format.r32_float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label='bias_1_grad',
        )

        # Allocate the adam state buffers
        self.weight0_adam = self.device.create_buffer(
            size = 12 * mlp_width * 8,
            format=spy.Format.rg32_float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label='weight_1_adam',
        )
        self.bias0_adam = self.device.create_buffer(
            size = mlp_width * 8,
            format=spy.Format.rg32_float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label='bias_0_adam',
        )
        self.weight1_adam = self.device.create_buffer(
            size = mlp_width * channels * 8,
            format=spy.Format.rg32_float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label='weight_1_adam',
        )
        self.bias1_adam = self.device.create_buffer(
            size = channels * 8,
            format=spy.Format.rg32_float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label='bias_1_adam',
        )

    # Clear all gradient buffers
    def clear_buffers(self, command_encoder):
        for i in range(len(self.t_dims)):
            command_encoder.clear_buffer(self.t_e0_grad[i])
            command_encoder.clear_buffer(self.t_e1_grad[i])
            command_encoder.clear_buffer(self.t_a_grad[i])

        command_encoder.clear_buffer(self.weight0_grad)
        command_encoder.clear_buffer(self.bias0_grad)
        command_encoder.clear_buffer(self.weight1_grad)
        command_encoder.clear_buffer(self.bias1_grad)

    # Returns a dictionary defining the shader bindings for a latent texture
    def latent_texture_parameter(self, i: int):
        return {
            'e0Texture': self.t_e0[i],
            'e1Texture': self.t_e1[i],
            'aTexture': self.t_a[i],
            'e0AccumulateBuffer': self.t_e0_grad[i],
            'e1AccumulateBuffer': self.t_e1_grad[i],
            'aAccumulateBuffer': self.t_a_grad[i],
            'mipOffsets': self.dev_mip_offsets[i],
        }

    # Learn the gradients for a single batch
    def learn_gradients(self, command_encoder, sample_stride, sample_offset, s: float):
        self.learn_grad_kernel.dispatch(
            thread_count = [self.args.batch_size, self.args.batch_size, 1],
            vars = {
                'texture': {
                    't0': self.latent_texture_parameter(0),
                    't1': self.latent_texture_parameter(1),
                    't2': self.latent_texture_parameter(2),
                    't3': self.latent_texture_parameter(3),
                    'layer1': {
                        'weights': self.weight0_buffer,
                        'weightsGrad': self.weight0_grad,
                        'biases': self.bias0_buffer,
                        'biasesGrad': self.bias0_grad,
                    },
                    'layer2': {
                        'weights': self.weight1_buffer,
                        'weightsGrad': self.weight1_grad,
                        'biases': self.bias1_buffer,
                        'biasesGrad': self.bias1_grad,
                    },
                    'var_a': self.var_a,
                },
                'target': {
                    'texture':self.target_texture_array,
                    'sampler':self.target_sampler,
                },
                'sample_stride': sample_stride,
                'sample_offset': sample_offset,
                'mip_level': s,
                'loss_buffer': self.loss_buffer,
                'l1': self.args.l1,
                'report_loss': not (self.args.quiet and self.args.no_gui),
                'no_grad': False,
            },
            command_encoder = command_encoder
        )

    # Update the learning rates with cosine annealing
    def cosine_annealing(self):
        self.texture_lr = 0.5 * self.args.texture_lr * (1 + math.cos((self.epoch - 1) * math.pi / self.args.training_epochs))
        self.mlp_lr = 0.5 * self.args.mlp_lr * (1 + math.cos((self.epoch - 1) * math.pi / self.args.training_epochs))

    # Update the latent textures using the adam optimizer
    def step_latent_texture(self, command_encoder, i: int, t_s0: int, t_s1: int):
        w = self.t_dims[i][0] >> t_s0
        h = self.t_dims[i][1] >> t_s0

        # Update the lower mip
        # E0
        self.step_texture_e_kernel.dispatch(
            thread_count = [w // 4, h // 4, 1],
            vars = {
                'mip_offset':self.mip_offsets[i][t_s0],
                'iteration': self.epoch,
                'learning_rate': self.texture_lr,
                'grad_buffer': self.t_e0_grad[i],
                'adam_state_buffer': self.t_e0_adam[i],
                'dst_texture': self.t_e0_uav[i][t_s0],
            },
            command_encoder = command_encoder
        )
        # E1
        self.step_texture_e_kernel.dispatch(
            thread_count = [w // 4, h // 4, 1],
            vars = {
                'mip_offset':self.mip_offsets[i][t_s0],
                'iteration': self.epoch,
                'learning_rate': self.texture_lr,
                'grad_buffer': self.t_e1_grad[i],
                'adam_state_buffer': self.t_e1_adam[i],
                'dst_texture': self.t_e1_uav[i][t_s0],
            },
            command_encoder = command_encoder
        )
        # A
        self.step_texture_a_kernel.dispatch(
            thread_count = [w, h, 1],
            vars = {
                'mip_offset':self.mip_offsets[i][t_s0],
                'iteration': self.epoch,
                'learning_rate': self.texture_lr,
                'grad_buffer': self.t_a_grad[i],
                'adam_state_buffer': self.t_a_adam[i],
                'dst_texture': self.t_a_uav[i][t_s0],
            },
            command_encoder = command_encoder
        )

        # Update the upper mip (if it's a different mip)
        if t_s0 != t_s1:
            w = self.t_dims[i][0] >> t_s1
            h = self.t_dims[i][1] >> t_s1

            # E0
            self.step_texture_e_kernel.dispatch(
                thread_count = [w // 4, h // 4, 1],
                vars = {
                    'mip_offset': self.mip_offsets[i][t_s1],
                    'iteration': self.epoch,
                    'learning_rate': self.texture_lr,
                    'grad_buffer': self.t_e0_grad[i],
                    'adam_state_buffer': self.t_e0_adam[i],
                    'dst_texture': self.t_e0_uav[i][t_s1],
                },
                command_encoder = command_encoder
            )
            # E1
            self.step_texture_e_kernel.dispatch(
                thread_count = [w // 4, h // 4, 1],
                vars = {
                    'mip_offset':self.mip_offsets[i][t_s1],
                    'iteration': self.epoch,
                    'learning_rate': self.texture_lr,
                    'grad_buffer': self.t_e1_grad[i],
                    'adam_state_buffer': self.t_e1_adam[i],
                    'dst_texture': self.t_e1_uav[i][t_s1],
                },
                command_encoder = command_encoder
            )
            # A
            self.step_texture_a_kernel.dispatch(
                thread_count = [w, h, 1],
                vars = {
                    'mip_offset':self.mip_offsets[i][t_s1],
                    'iteration': self.epoch,
                    'learning_rate': self.texture_lr,
                    'grad_buffer': self.t_a_grad[i],
                    'adam_state_buffer': self.t_a_adam[i],
                    'dst_texture': self.t_a_uav[i][t_s1],
                },
                command_encoder = command_encoder
            )

    # Update the MLP decoder using the ADAM optimizer
    def step_mlp_buffers(self, command_encoder):
        weight0_count = 12 * self.args.mlp_width
        bias0_count = self.args.mlp_width
        weight1_count = self.args.mlp_width * self.c
        bias1_count = self.c

        # Weight0
        self.step_buffer_kernel.dispatch(
            thread_count = [weight0_count, 1, 1],
            vars = {
                'count': weight0_count,
                'iteration': self.epoch,
                'learning_rate': self.mlp_lr,
                'grad_buffer': self.weight0_grad,
                'adam_state_buffer': self.weight0_adam,
                'dst_buffer': self.weight0_buffer,
            },
            command_encoder = command_encoder,
        )
        # Bias0
        self.step_buffer_kernel.dispatch(
            thread_count = [bias0_count, 1, 1],
            vars = {
                'count': bias0_count,
                'iteration': self.epoch,
                'learning_rate': self.mlp_lr,
                'grad_buffer': self.bias0_grad,
                'adam_state_buffer': self.bias0_adam,
                'dst_buffer': self.bias0_buffer,
            },
            command_encoder = command_encoder,
        )
        # Weight1
        self.step_buffer_kernel.dispatch(
            thread_count = [weight1_count, 1, 1],
            vars = {
                'count': weight1_count,
                'iteration': self.epoch,
                'learning_rate': self.mlp_lr,
                'grad_buffer': self.weight1_grad,
                'adam_state_buffer': self.weight1_adam,
                'dst_buffer': self.weight1_buffer,
            },
            command_encoder = command_encoder,
        )
        # Bias1
        self.step_buffer_kernel.dispatch(
            thread_count = [bias1_count, 1, 1],
            vars = {
                'count': bias1_count,
                'iteration': self.epoch,
                'learning_rate': self.mlp_lr,
                'grad_buffer': self.bias1_grad,
                'adam_state_buffer': self.bias1_adam,
                'dst_buffer': self.bias1_buffer,
            },
            command_encoder = command_encoder,
        )

    # Decode the neural texture for the final PSNR report
    def decode_neural_texture(self, command_encoder):
        for i in range(self.mip_count):
            self.decode_kernel.dispatch(
                thread_count = [self.w >> i, self.h >> i, 1],
                vars = {
                    'texture': {
                        't0': self.latent_texture_parameter(0),
                        't1': self.latent_texture_parameter(1),
                        't2': self.latent_texture_parameter(2),
                        't3': self.latent_texture_parameter(3),
                        'layer1': {
                            'weights': self.weight0_buffer,
                            'weightsGrad': self.weight0_grad,
                            'biases': self.bias0_buffer,
                            'biasesGrad': self.bias0_grad,
                        },
                        'layer2': {
                            'weights': self.weight1_buffer,
                            'weightsGrad': self.weight1_grad,
                            'biases': self.bias1_buffer,
                            'biasesGrad': self.bias1_grad,
                        },
                        'var_a': self.var_a,
                    },
                    'mip_level': i,
                    'dst_texture': self.dst_texture_uavs[i],
                },
                command_encoder = command_encoder,
            )

    # Readback data from the GPU and save it to the file
    def export_neural_texture(self, file: str):
        output = bytearray()

        # Basic texture info
        if not self.args.half_resolution:
            output += self.w.to_bytes(2, 'little') # width
            output += self.h.to_bytes(2, 'little') # height
            output += self.c.to_bytes(1, 'little')  # channels
            output += self.mip_count.to_bytes(1, 'little') # max mip chain length
        else:
            output += max(self.w // 2, 4).to_bytes(2, 'little') # width
            output += max(self.w // 2, 4).to_bytes(2, 'little') # height
            output += self.c.to_bytes(1, 'little')  # channels
            output += max(self.mip_count - 1, 1).to_bytes(1, 'little') # max mip chain length
        output += self.var_a.to_bytes(1, 'little') # Var A?
        output += self.args.mlp_width.to_bytes(1, 'little') # hidden layer width
        output += len(self.texture_channels).to_bytes(1, 'little') # num textures
        for tex_channel in self.texture_channels:
            output += tex_channel.to_bytes(1, 'little') # channels in texture

        # Readback and quantize the latent textures
        quant_scales = np.array([31, 63, 31])
        quant_inv_scales = np.array([1 / 31, 1 / 63, 1 / 31])
        for t, dim in enumerate(self.t_dims):
            e0_np = []
            e1_np = []
            a_np = []
            for mip in range(dim[2]):
                e0 = self.t_e0[t].to_numpy(mip=mip)[:,:,0:3]
                e1 = self.t_e1[t].to_numpy(mip=mip)[:,:,0:3]
                a = np.expand_dims(self.t_a[t].to_numpy(mip=mip), axis=2)

                e0 = np.round(sigmoid(e0) * quant_scales) * quant_inv_scales
                e1 = np.round(sigmoid(e1) * quant_scales) * quant_inv_scales
                a = np.round(sigmoid(a) * 3) * (1 / 3)

                e0_np.append(e0)
                e1_np.append(e1)
                a_np.append(a)
            output += export_bc1(e0_np, e1_np, a_np)

        # Readback the MLP weights and biases, and convert to half floats
        weight0_np = self.weight0_buffer.to_numpy().astype('<f2')
        weight1_np = self.weight1_buffer.to_numpy().astype('<f2')
        bias0_np = self.bias0_buffer.to_numpy().astype('<f2')
        bias1_np = self.bias1_buffer.to_numpy().astype('<f2')

        output += weight0_np.tobytes()
        output += bias0_np.tobytes()
        output += weight1_np.tobytes()
        output += bias1_np.tobytes()

        # Write to file
        with open(file, 'bw') as f:
            f.write(output)

        # Print output size in kb
        if not self.args.quiet and self.args.no_gui:
            print(f'Size: {len(output) // 1024}kb')


    # Train the neural texture for a single epoch
    def train_epoch(self):
        if self.epoch > self.args.training_epochs:
            return

        # Nvidia NTC mip sampling scheme - mostly exponential, with occasional uniform sampling
        if random.random() > 0.05:
            # 95% chance of exponential distribution
            s = min(self.mip_count - 1.0, -math.log(random.random(), 4.0))
        else:
            # 5% chance to uniformly samply lod
            s = (self.mip_count - 1.0) * random.random()
        s0 = int(s)
        s1 = int(min(s + 1, self.mip_count - 1.0))

        # Batch is sampled as a uniform grid with a random offset
        sample_stride = np.asarray([1.0 / (self.w >> s0), 1.0 / (self.h >> s0)]).astype(np.float32)
        sample_offset = np.random.rand(2).astype(np.float32)

        # Clear grad buffers
        command_encoder = self.device.create_command_encoder()
        self.clear_buffers(command_encoder)
        self.device.submit_command_buffer(command_encoder.finish())

        # Learn gradients
        command_encoder = self.device.create_command_encoder()
        command_encoder.write_timestamp(self.queries, (self.epoch - 1) * 2 + 0)
        self.learn_gradients(command_encoder, sample_stride, sample_offset, s)
        self.device.submit_command_buffer(command_encoder.finish())
        
        # Update learning rates if needed
        if self.args.cos:
            self.cosine_annealing()

        # Step latent textures
        command_encoder = self.device.create_command_encoder()
        for i, dim in enumerate(self.t_dims):
            t_s0 = int(max(s0 - dim[3], 0))
            t_s1 = int(max(s1 - dim[3], 0))
            self.step_latent_texture(command_encoder, i, t_s0, t_s1)

        # Step the MLP parameters
        self.step_mlp_buffers(command_encoder)
        command_encoder.write_timestamp(self.queries, (self.epoch - 1) * 2 + 1)

        self.device.submit_command_buffer(command_encoder.finish())
        self.device.wait()

        # If we should report the loss
        if not (self.args.quiet and self.args.no_gui) and self.epoch % self.args.reporting_period == 0:
            loss = self.loss_buffer.to_numpy()[0]

            # The loss metric is accumulated as a sum
            loss /= (self.args.batch_size ** 2) * self.c * self.args.reporting_period
            self.psnr = -10 * np.log10(loss)

            if self.args.no_gui:
                print(f'EPOCH {self.epoch}: {self.psnr:.4f}dB')


            # Clear out the loss buffer
            temp_command_encoder = self.device.create_command_encoder()
            temp_command_encoder.clear_buffer(self.loss_buffer)
            self.device.submit_command_buffer(temp_command_encoder.finish())
            self.device.wait()

        self.epoch += 1

        # Optimization is complete
        if self.epoch > self.args.training_epochs:
            # Print out the training duration, and mean epoch time
            self.total_cpu_time = self.t.elapsed_s()
            times = np.asarray(self.queries.get_timestamp_results(0, self.args.training_epochs * 2))
            self.epoch_mean_time = np.mean(times[1::2] - times[0::2]) * 1e3
            if self.args.no_gui and not self.args.quiet:
                print(f'Completed in {self.total_cpu_time:.2f} seconds')
                print(f'Each epoch took {self.epoch_mean_time:.2f}ms')
                print(self.args.format, self.args.training_epochs)
            
            # Export the neural texture if we have an output configured
            if self.args.output is not None:
                self.export_neural_texture(self.args.output)

            # Fully decode the neural texture
            temp_command_encoder = self.device.create_command_encoder()
            self.decode_neural_texture(temp_command_encoder)
            self.device.submit_command_buffer(temp_command_encoder.finish())
            self.device.wait()

            self.mip_psnrs = []
            target_data = np.empty(0)
            output_data = np.empty(0)
            # Calculate PSNR for each mip
            for mip in range(self.mip_count):
                y = np.concatenate([np.ravel(self.target_texture_array.to_numpy(layer=channel, mip=mip)) for channel in range(self.c)])
                y_h = np.concatenate([np.ravel(self.dst_texture_array.to_numpy(layer=channel, mip=mip)) for channel in range(self.c)])
                target_data = np.append(target_data, y)
                output_data = np.append(output_data, y_h)
                mse = np.mean((y - y_h) ** 2)
                psnr = -10 * np.log10(mse)
                self.mip_psnrs.append(psnr)
            # Cumulative PSNR is calculated by concatenating the data of all the mips
            # together, and calculating the total MSE
            cumulative_mse = np.mean((target_data - output_data) ** 2)
            self.cumulative_psnr = -10 * np.log10(cumulative_mse)

            # Print results to stdout if appropriate
            if self.args.no_gui and not self.args.quiet:
                for mip, mip_psnr in enumerate(self.mip_psnrs):
                    print(f'Mip {mip} PSNR: {mip_psnr:.4f}dB')
                print(f'Cumulative PSNR: {self.cumulative_psnr:.4f}dB')

            self.needs_rerender = True
            self.complete = True

