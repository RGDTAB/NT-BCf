import common;
import neural_texture;
import mlp_sw;

extern static const int MLP_WIDTH;
extern static const int CHANNELS;

uniform NeuralTexture<MLP_WIDTH, CHANNELS> texture;
uniform uint mip_level;
uniform RWTexture2DArray<float> dst_texture;

// Kernel to decode every pixel in a mip level of a neural texture
// Used when calculating PSNR values
[shader("compute")]
[numthreads(8, 8, 1)]
void main(int2 tid: SV_DispatchThreadID)
{
    uint2 pos = tid.xy;
    uint w, h, elements, levels;
    dst_texture.GetDimensions(mip_level, w, h, elements, levels);
    if (pos.x >= w || pos.y >= h) {
        return;
    }
    // 0.5 offset to place it at the pixels' centers
    float2 uv = (float2(pos) + 0.5) / float2(w, h);

    let output = texture.Eval(NFloat2(uv), NFloat(mip_level));

    [ForceUnroll]
    for (int i = 0; i < CHANNELS; i++)
    {
        dst_texture[uint3(pos, i)] = output.data[i];
    }
}
