import common;
import neural_texture;
import target_set;
import mlp_sw;

extern static const int MLP_WIDTH;
extern static const int CHANNELS;

uniform NeuralTexture<MLP_WIDTH, CHANNELS> texture;
uniform TargetSet<CHANNELS> target;
uniform float2 sample_stride;
uniform float2 sample_offset;
uniform float mip_level;
uniform RWBuffer<float> loss_buffer;
uniform bool l1;
uniform bool report_loss;
uniform bool no_grad; 

// Kernel to accumulate gradients for a single epoch
[shader("compute")]
[numthreads(8, 8, 1)]
void LearnGradient(int2 tid: SV_DispatchThreadID)
{
    // Calculate this thread's location
    NFloat2 local_offset = NFloat2(sample_stride * float2(tid) + sample_offset);
    local_offset = local_offset - floor(local_offset);
    NFloat mip = NFloat(mip_level);

    MLVec<CHANNELS> target_sample = target.Sample(local_offset, mip);
    // no_grad is currently unused
    if (!no_grad)
    {
        // Compute gradients - L2 loss
        if (!l1)
        {
            bwd_diff(L2Loss<MLP_WIDTH, CHANNELS>)(texture, local_offset, mip, target_sample, NFloat(1.0f));
        }
        // Compute gradients - L1 loss
        else
        {
            bwd_diff(L1Loss<MLP_WIDTH, CHANNELS>)(texture, local_offset, mip, target_sample, NFloat(1.0f));
        }
    }

    // We need to do an extra forward pass to be able to report loss
    // The performance cost is negligible
    if (report_loss)
    {
        float loss = 0.0;
        // L2 loss
        if (!l1)
        {
            loss = (float) L2Loss<MLP_WIDTH, CHANNELS>(texture, local_offset, mip, target_sample);
        }
        else
        {
            loss = (float) L1Loss<MLP_WIDTH, CHANNELS>(texture, local_offset, mip, target_sample);
        }

        // Wave intrinsics are only a minor speedup here
        loss = WaveActiveSum(loss);
        if (WaveIsFirstLane())
        {
            InterlockedAdd<float>(loss_buffer[0], loss);
        }
    }
}

