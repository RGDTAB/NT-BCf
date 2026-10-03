module encoder_window;

import common;

public uniform RWTexture2D<float4> dst_texture;

// Encoder window kernel for before encoding has started
[shader("compute")]
[numthreads(4, 8, 1)]
void main(int2 tid: SV_DispatchThreadID)
{
    uint2 pos = tid.xy;
    uint w, h;
    dst_texture.GetDimensions(w, h);
    if (pos.x >= w || pos.y >= h) {
        return;
    }

    dst_texture[pos] = checkerboard(pos);
}
