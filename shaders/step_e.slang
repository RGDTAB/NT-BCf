import common;
import mlp_sw;
import adam;

uniform uint mip_offset;
uniform uint iteration;
uniform NFloat learning_rate;
uniform Buffer<float4> grad_buffer;
uniform RWBuffer<NFloat4> adam_state_buffer;
uniform RWTexture2D<NFloat4> dst_texture;

// Steps a color endpoint texture using the ADAM optimizer
[shader("compute")]
[numthreads(8, 8, 1)]
void GradientStepTextureE(int3 tid: SV_DispatchThreadID)
{
    // Range check
    uint2 pos = tid.xy;
    uint w, h;
    dst_texture.GetDimensions(w, h);
    if (pos.x >= w || pos.y >= h) {
        return;
    }

    // Load the current value
    NFloat4 val = dst_texture[pos];

    // Load the accumulated gradient
    NFloat4 grad = NFloat4(grad_buffer[mip_offset / 16 + pos.y * w + pos.x]);

    if (isnan(grad.x) || isnan(grad.y) || isnan(grad.z)) {
        return;
    }

    // Load the mean and variance of the ADAM state
    NFloat4 adam0 = adam_state_buffer[mip_offset / 8 + (pos.y * w + pos.x) * 2 + 0];
    NFloat4 adam1 = adam_state_buffer[mip_offset / 8 + (pos.y * w + pos.x) * 2 + 1];

    // Step
    AdamOptimizer.step4(iteration, adam0, adam1, val, grad, learning_rate);

    // Store updated values
    dst_texture[pos] = val;
    adam_state_buffer[mip_offset / 8 + (pos.y * w + pos.x) * 2 + 0] = adam0;
    adam_state_buffer[mip_offset / 8 + (pos.y * w + pos.x) * 2 + 1] = adam1;
}

