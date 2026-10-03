import common;
import mlp_sw;
import adam;

uniform uint count;
uniform uint iteration;
uniform NFloat learning_rate;
uniform Buffer<float> grad_buffer;
uniform RWBuffer<NFloat2> adam_state_buffer;
uniform RWBuffer<NFloat> dst_buffer;

// Steps a single-channel interpolation texture using the ADAM optimizer
[shader("compute")]
[numthreads(64, 1, 1)]
void GradientStepBuffer(int tid: SV_DispatchThreadID)
{
    // Range check
    if (tid.x >= count) {
        return;
    }

    // Load the current value
    NFloat val = dst_buffer[tid.x];

    // Load the accumulated gradient
    NFloat grad = NFloat(grad_buffer[tid.x]);

    if (isnan(grad)) {
        return;
    }

    // Load the mean and variance of the ADAM state
    NFloat2 adam_state = adam_state_buffer[tid.x];

    // Step
    AdamOptimizer.step(iteration, adam_state, val, grad, learning_rate);

    // Store updated values
    dst_buffer[tid.x] = val;
    adam_state_buffer[tid.x] = adam_state;
}

