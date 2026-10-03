module target_set;

import common;
import mlp_sw;

// Represents the target texture array that we are trying to reconstruct
public struct TargetSet<int channels>
{
    public Texture2DArray<NFloat> texture; // Hardware texture handle
    public SamplerState sampler; // Bilinear texture sampler

    // Bilinear sampling for each texture in the texture array
    public MLVec<channels> SampleBilinear(float2 uv, uint mip_level)
    {
        MLVec<channels> output;
        for (int i = 0; i < channels; i++) {
            NFloat sample = texture.SampleLevel(sampler, float3(uv, float(i)), mip_level);
            output.data[i] = sample;
        }
        return output;
    }

    // Fast software cubic interpolation filter - Catmull-Rom
    // Note - PyTorch uses another form of cubic interpolation
    // The MIT License
    // Copyright 2019 MJP
    // https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
    public MLVec<channels> SampleBicubic(NFloat2 uv, uint mip_level, NFloat2 texSize)
    {
        // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
        // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
        // location [1, 1] in the grid, where [0, 0] is the top left corner.
        float2 samplePos = uv * texSize;
        float2 texPos1 = floor(samplePos - 0.5f) + 0.5f;

        // Compute the fractional offset from our starting texel to our original sample location, which we'll
        // feed into the Catmull-Rom spline function to get our filter weights.
        float2 f = samplePos - texPos1;

        // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
        // These equations are pre-expanded based on our knowledge of where the texels will be located,
        // which lets us avoid having to evaluate a piece-wise function.
        float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
        float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
        float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
        float2 w3 = f * f * (-0.5f + 0.5f * f);

        // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
        // simultaneously evaluate the middle 2 samples from the 4x4 grid.
        float2 w12 = w1 + w2;
        float2 offset12 = w2 / (w1 + w2);

        // Compute the final UV coordinates we'll use for sampling the texture
        float2 texPos0 = texPos1 - 1;
        float2 texPos3 = texPos1 + 2;
        float2 texPos12 = texPos1 + offset12;

        texPos0 /= texSize;
        texPos3 /= texSize;
        texPos12 /= texSize;

        MLVec<channels> result;
        return SampleBilinear(float2(texPos0.x,  texPos0.y), mip_level) * NFloat(w0.x * w0.y)
        + SampleBilinear(float2(texPos12.x, texPos0.y), mip_level) * NFloat(w12.x * w0.y)
        + SampleBilinear(float2(texPos3.x,  texPos0.y), mip_level) * NFloat(w3.x * w0.y)

        + SampleBilinear(float2(texPos0.x,  texPos12.y), mip_level) * NFloat(w0.x * w12.y)
        + SampleBilinear(float2(texPos12.x, texPos12.y), mip_level) * NFloat(w12.x * w12.y)
        + SampleBilinear(float2(texPos3.x,  texPos12.y), mip_level) * NFloat(w3.x * w12.y)

        + SampleBilinear(float2(texPos0.x,  texPos3.y), mip_level) * NFloat(w0.x * w3.y)
        + SampleBilinear(float2(texPos12.x, texPos3.y), mip_level) * NFloat(w12.x * w3.y)
        + SampleBilinear(float2(texPos3.x,  texPos3.y), mip_level) * NFloat(w3.x * w3.y);
    }

    // Linearly interpolates between bicubic samples
    public MLVec<channels> Sample(NFloat2 uv, NFloat LOD)
    {
        uint w;
        uint h;
        uint elements;
        uint levels;
        texture.GetDimensions(0, w, h, elements, levels);

        NFloat maxLOD = NFloat(levels - 1);
        NFloat clampedLOD = max(0, (min(maxLOD, LOD)));

        NFloat lodFrac = clampedLOD - floor(clampedLOD);
        uint lod0 = (uint)floor(clampedLOD);
        uint lod1 = min(levels - 1, lod0 + 1);
        NFloat weightLod0 = NFloat(1.0f) - lodFrac;
        NFloat weightLod1 = lodFrac;

        let v0 = SampleBicubic(uv, lod0, NFloat2(NFloat(w >> lod0), NFloat(h >> lod0))) * weightLod0;
        let v1 = SampleBicubic(uv, lod1, NFloat2(NFloat(w >> lod1), NFloat(h >> lod1))) * weightLod1;
        return v0 + v1;
    }
}
