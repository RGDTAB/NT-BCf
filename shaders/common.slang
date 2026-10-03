module common;

// I originally had plans for an option to use half precision floats...
// Those plans fell through when all my attempts refused to converge,
// and the performance was actually worse than single precision.
// I left this in just in case anyone else wanted to take a crack at it
#ifndef HALF_MATH
public typealias NFloat  = float;
public typealias NFloat2 = float2;
public typealias NFloat3 = float3;
public typealias NFloat4 = float4;
#else
public typealias NFloat  = half;
public typealias NFloat2 = half2;
public typealias NFloat3 = half3;
public typealias NFloat4 = half4;
#endif

public NFloat sigmoid(NFloat input)
{
    return rcp(NFloat(1.0f) + exp(-input));
}

public NFloat4 sigmoid(NFloat4 input)
{
    return rcp(NFloat(1.0f) + exp(-input));
}

// Not configured for the autodiff, as it's only ever called manually
public NFloat bwd_sigmoid(NFloat input)
{
    return input * (NFloat(1.0) - input);
}

public NFloat4 bwd_sigmoid(NFloat4 input)
{
    return input * (NFloat(1.0) - input);
}


// Checkerboard pattern used by the encoder window
public float4 checkerboard(uint2 pos)
{
    float shades[2] = {0.2, 0.6};
    uint id = (pos.x / 10 + pos.y / 10) % 2;
    float shade = shades[id];
    return float4(shade, shade, shade, 1);
}

