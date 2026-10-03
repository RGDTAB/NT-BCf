# Gets the number of bytes in a bc1 mip chain
# Levels in mip chain must be known
def bc1_calc_mip_size(width, height, mip_count):
    size = 0
    n_mips = max(0, min(width.bit_length() - 2, height.bit_length() - 2, mip_count))
    for _ in range(n_mips):
        size += (width * height) // 2
        width //= 2
        height //= 2

    return size

# Returns the neural texture's metadata 
def parse_texture_params(data : bytes):
    # Reading parameters from the data buffer
    width = data[0] + (data[1] << 8)
    height = data[2] + (data[3] << 8)
    channels = data[4]
    mip_count = data[5]
    var_a = data[6] != 0
    mlp_width = data[7]
    input_texture_count = data[8]
    texture_params = (width, height, channels, mip_count, var_a, mlp_width, input_texture_count)

    input_channels = []
    input_channel_offset = 9
    for _ in range(input_texture_count):
        input_channels.append(data[input_channel_offset])
        input_channel_offset += 1

    # Resolution of each latent texture
    t0_params = (width, height, mip_count)
    if var_a:
        t1_params = (width, height, mip_count)
        t2_params = (max(width // 2, 4), max(height // 2, 4), max(mip_count - 1, 1))
        t3_params = (max(width // 2, 4), max(height // 2, 4), max(mip_count - 1, 1))
    else:
        t1_params = (max(width // 2, 4), max(height // 2, 4), max(mip_count - 1, 1))
        t2_params = (max(width // 4, 4), max(height // 4, 4), max(mip_count - 2, 1))
        t3_params = (max(width // 8, 4), max(height // 8, 4), max(mip_count - 3, 1))
    latent_params = (t0_params, t1_params, t2_params, t3_params)

    # Byte offsets into the data buffer
    t0_offset = input_channel_offset
    t1_offset = t0_offset + bc1_calc_mip_size(*t0_params)
    t2_offset = t1_offset + bc1_calc_mip_size(*t1_params)
    t3_offset = t2_offset + bc1_calc_mip_size(*t2_params)
    t3_end = t3_offset + bc1_calc_mip_size(*t3_params)
    latent_offsets = (t0_offset, t1_offset, t2_offset, t3_offset, t3_end)

    weight0_offset = t3_end
    bias0_offset = weight0_offset + 12 * mlp_width * 2
    weight1_offset = bias0_offset + mlp_width * 2
    bias1_offset = weight1_offset + mlp_width * channels * 2
    bias1_end = bias1_offset + channels * 2
    mlp_offsets = (weight0_offset, bias0_offset, weight1_offset, bias1_offset, bias1_end)

    return (texture_params, input_channels, latent_params, latent_offsets, mlp_offsets)
