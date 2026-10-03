module latent_texture;

import common;

// Software BC1 texture for auto diff
// Much of this code comes from slang's autodiff-texture example
// https://github.com/shader-slang/slang/blob/master/examples/autodiff-texture/train.slang
public struct LatentTexture
{
    RWBuffer<float> e0AccumulateBuffer;
    RWBuffer<float> e1AccumulateBuffer;
    RWBuffer<float> aAccumulateBuffer;
    Texture2D<NFloat4> e0Texture, e1Texture;     // Color endpoint Hardware texture handles.
    public Texture2D<NFloat> aTexture;                  // Color indices Hardware texture handles.
    Buffer<uint> mipOffsets;

    // Quantize color endpoint
    NFloat4 QuantizeE(NFloat4 e)
    {
        e = round(e * NFloat4(NFloat(31.0f), NFloat(63.0f), NFloat(31.0f), NFloat(1.0f)));
        e /= NFloat4(NFloat(31.0f), NFloat(63.0f), NFloat(31.0f), NFloat(1.0f));
        return e;
    }

    // Quantize color interpolator
    NFloat QuantizeA(NFloat a)
    {
        return round(a * NFloat(3.0f)) / NFloat(3.0f);
    }

    // Decodes the 'e0', 'e1', and 'a' textures into rgba format (alpha value is not used)
    [BackwardDerivative(bwd_LoadTexel)]
    public NFloat4 LoadTexel(int3 aLoc, uint dLayerW, uint dMipOffset)
    {
        int3 eLoc = int3(aLoc.xy / 4, aLoc.z);
        NFloat4 e0 = e0Texture.Load(eLoc);
        e0 = no_diff QuantizeE(sigmoid(e0));
        NFloat4 e1 = e1Texture.Load(eLoc);
        e1 = no_diff QuantizeE(sigmoid(e1));
        NFloat  a  = aTexture.Load(aLoc);
        a = no_diff QuantizeA(sigmoid(a));
        NFloat4 c = lerp(e0, e1, a);
        return c;
    }

    // Calculates gradients and atomically adds them to the accumulate buffers
    public void bwd_LoadTexel(int3 aLoc, uint dLayerW, uint dMipOffset, NFloat4 val)
    {
        int3 eLoc = int3(aLoc.xy / 4, aLoc.z);
        uint eLayerW = dLayerW / 4;

        NFloat4 e0 = e0Texture.Load(eLoc);
        NFloat4 e1 = e1Texture.Load(eLoc);
        NFloat a  = aTexture.Load(aLoc);
        NFloat4 sig_e0 = sigmoid(e0);
        NFloat4 sig_e1 = sigmoid(e1);
        NFloat sig_a = sigmoid(a);

        float4 dE0 = float4(bwd_sigmoid(sig_e0) * (NFloat(1.0) - sig_a) * val);
        float4 dE1 = float4(bwd_sigmoid(sig_e1) * sig_a * val);
        float4 vdA = float4(bwd_sigmoid(sig_a) * (sig_e1 - sig_e0) * val);
        float  dA  = dot(vdA.rgb, float3(1));

        // dMipOffset is divided by 4 because the e textures have 1/16th the number of pixels, but 4 channels instead of 1
        InterlockedAdd<float>(e0AccumulateBuffer[dMipOffset / 4 + ((uint)eLoc.y * eLayerW + (uint)eLoc.x) * 4 + 0], dE0.x);
        InterlockedAdd<float>(e0AccumulateBuffer[dMipOffset / 4 + ((uint)eLoc.y * eLayerW + (uint)eLoc.x) * 4 + 1], dE0.y);
        InterlockedAdd<float>(e0AccumulateBuffer[dMipOffset / 4 + ((uint)eLoc.y * eLayerW + (uint)eLoc.x) * 4 + 2], dE0.z);

        InterlockedAdd<float>(e1AccumulateBuffer[dMipOffset / 4 + ((uint)eLoc.y * eLayerW + (uint)eLoc.x) * 4 + 0], dE1.x);
        InterlockedAdd<float>(e1AccumulateBuffer[dMipOffset / 4 + ((uint)eLoc.y * eLayerW + (uint)eLoc.x) * 4 + 1], dE1.y);
        InterlockedAdd<float>(e1AccumulateBuffer[dMipOffset / 4 + ((uint)eLoc.y * eLayerW + (uint)eLoc.x) * 4 + 2], dE1.z);

        InterlockedAdd<float>(aAccumulateBuffer[dMipOffset + (uint)aLoc.y * dLayerW + (uint)aLoc.x], dA);
    }

    // Software reference implementation of bilinear filtering.
    [BackwardDifferentiable]
    public NFloat4 sampleTexture_linear(uint lod, NFloat2 uv, uint w, uint h)
    {
        w >>= lod;
        h >>= lod;
        uv = frac(uv);
        float2 loc = float2(uv) * float2(float(w), float(h)) - 0.5;
        float x0 = no_diff(floor(loc.x));
        float y0 = no_diff(floor(loc.y));
        float fracX = frac(loc.x);
        float fracY = frac(loc.y);
        float x1 = x0 + 1;
        float y1 = y0 + 1;
        if (x0 < 0) x0 += float(w);
        if (y0 < 0) y0 += float(h);
        if (x1 >= float(w)) x1 -= float(w);
        if (y1 >= float(h)) y1 -= float(h);
        NFloat weight0 = NFloat(1.0f - fracY);
        NFloat weight1 = NFloat(fracY);
        NFloat weight00 = weight0 * NFloat(1.0f - fracX);
        NFloat weight01 = weight0 * NFloat(fracX);
        NFloat weight10 = weight1 * NFloat(1.0f - fracX);
        NFloat weight11 = weight1 * NFloat(fracX);

        // autodiff-texture example bit-shifted w again here... I don't think that's correct
        uint dLayerW = w;
        var offset = mipOffsets[lod];
        return LoadTexel(int3(int(x0), int(y0), int(lod)), dLayerW, offset) * weight00 +
               LoadTexel(int3(int(x1), int(y0), int(lod)), dLayerW, offset) * weight01 +
               LoadTexel(int3(int(x0), int(y1), int(lod)), dLayerW, offset) * weight10 +
               LoadTexel(int3(int(x1), int(y1), int(lod)), dLayerW, offset) * weight11;
    }

    // Software reference implementation of trilinear filtering.
    [BackwardDifferentiable]
    public NFloat4 sampleLevelTexture_trilinear(uint w, uint h, uint levels, NFloat2 uv, NFloat LOD)
    {
        // Isotropic filter.
        NFloat maxLOD = NFloat(levels - 1);
        NFloat clampedLOD = max(0, (min(maxLOD, LOD)));

        NFloat lodFrac = frac(clampedLOD);
        uint lod0 = (uint)floor(clampedLOD);
        uint lod1 = min(levels - 1, lod0 + 1);

        return lerp(sampleTexture_linear(lod0, uv, w, h), sampleTexture_linear(lod1, uv, w, h), lodFrac);
    }

    // Wrapper around sampleLevelTexture_trilinear
    [BackwardDifferentiable]
    public NFloat4 Sample(NFloat2 uv, NFloat LOD)
    {
        uint w;
        uint h;
        uint levels;
        aTexture.GetDimensions(0, w, h, levels);
        return sampleLevelTexture_trilinear(w, h, levels, uv, LOD);
    }
}
