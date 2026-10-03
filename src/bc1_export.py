import numpy as np

# Encodes a BC1 texture into a byte array
def export_bc1(e0_arr, e1_arr, a_arr):
    out = bytearray()

    # Integer values for quantization
    quant_scales = np.array([31, 63, 31])
    for i in range(len(e0_arr)):
        e0 = e0_arr[i]
        e1 = e1_arr[i]
        a = a_arr[i]

        # Pack e0 into 2 bytes
        e0 = np.round(e0 * quant_scales).astype(np.uint16)
        e0 = ((e0[:,:,0] & 31) << 11) | ((e0[:,:,1] & 63) << 5) | (e0[:,:,2] & 31)
        e0 = e0.astype('<u8')

        # Same with e1
        e1 = np.round(e1 * quant_scales).astype(np.uint16)
        e1 = ((e1[:,:,0] & 31) << 11) | ((e1[:,:,1] & 63) << 5) | (e1[:,:,2] & 31)
        e1 = e1.astype('<u8')

        lt = np.less(e0, e1).astype(np.ubyte)
        e_min = np.minimum(e0, e1)
        e_max = np.maximum(e0, e1)
        # The odds that e0 and e1 are equal are low... but never 0!
        n_eq = np.not_equal(e_min, e_max).astype('<u8')
        e_min = e_min ^ ((e_min ^ 0xFFFF) * (n_eq ^ 1))

        # Quantize the alpha values
        a = np.round(a * 3.0).squeeze(2).astype('<u8')
        # Switch from alpha interpolation values to DXT1 specification color indices
        # https://wikis.khronos.org/opengl/S3_Texture_Compression#DXT1_Format
        a = ((a + 1) & 3)
        a = a ^ ((a >> 1) ^ 1)

        # Bit pack e0, e1, and a into 8 byte blocks
        blocks = np.zeros_like(e0, dtype='<u8')
        blocks |= e_max
        blocks |= e_min << 16
        offset = 32
        for x in range(4):
            for y in range(4):
                # Bitwise OR the a values into the blocks
                blocks |= (((a[x::4,y::4] & 3) ^ lt) * n_eq) << offset
                offset += 2

        out += blocks.tobytes()
    return out
