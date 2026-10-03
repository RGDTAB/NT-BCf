import common;
import mlp_sw;
import adam;

uniform uint mip_offset;
uniform uint iteration;
uniform NFloat learning_rate;
uniform Buffer<float> grad_buffer;
uniform RWBuffer<NFloat2> adam_state_buffer;
uniform RWTexture2D<NFloat> dst_texture;

// Steps a single-channel interpolation texture using the ADAM optimizer
[shader("compute")]
[numthreads(8, 8, 1)]
void GradientStepTextureA(int3 tid: SV_DispatchThreadID)
{
    // Range check
    uint2 pos = tid.xy;
    uint w, h;
    dst_texture.GetDimensions(w, h);
    if (pos.x >= w || pos.y >= h) {
        return;
    }

    // Load the current value
    NFloat val = dst_texture[pos];

    // Load the accumulated gradient
    NFloat grad = NFloat(grad_buffer[mip_offset + pos.y * w + pos.x]);

    if (isnan(grad)) {
        return;
    }

    // Load the mean and variance of the ADAM state
    NFloat2 adam_state = adam_state_buffer[mip_offset + pos.y * w + pos.x];

    // Step
    AdamOptimizer.step(iteration, adam_state, val, grad, learning_rate);

    // Store updated values
    dst_texture[pos] = val;
    adam_state_buffer[mip_offset + pos.y * w + pos.x] = adam_state;
}

