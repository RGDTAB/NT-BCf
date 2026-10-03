module adam;

import common;

// Much of this code is modified from SlangPy's mlp-training example
// https://github.com/shader-slang/slang/blob/master/examples/mlp-training/adam.slang
public struct AdamOptimizer
{
    public static const NFloat beta1 = NFloat(0.9f);
    public static const NFloat beta2 = NFloat(0.999f);

    // Single precision floats can have a lower epsilon value without risk of NaNs
    #ifdef HALF_MATH
    public static const NFloat epsilon = NFloat(1e-7f);
    #else
    public static const NFloat epsilon = NFloat(1e-8f);
    #endif

    public static void step(uint iteration, inout NFloat2 state, inout NFloat param, inout NFloat grad, NFloat learningRate)
    {
        // Gradient clipping is done just as an anti-inf measure
        grad = clamp(grad, NFloat(-10000.0f), NFloat(10000.0f));

        state[0] = beta1 * state[0] + (NFloat(1.0f) - beta1) * grad;
        state[1] = beta2 * state[1] + (NFloat(1.0f) - beta2) * grad * grad;
        NFloat meanHat = state[0] / (NFloat(1.0f) - pow(beta1, NFloat(iteration)));
        NFloat varianceHat = state[1] / (NFloat(1.0f) - pow(beta2, NFloat(iteration)));
        param -= learningRate * meanHat / sqrt(max(NFloat(0.0f), varianceHat) + epsilon);
    }

    // Step function for 4-component vectors - helpful with updating color endpoint textures
    public static void step4(uint iteration, inout NFloat4 mean, inout NFloat4 variance, inout NFloat4 param, inout NFloat4 grad, NFloat learningRate)
    {
        // Gradient clipping is done just as an anti-inf measure
        grad = clamp(grad, NFloat(-10000.0f), NFloat(10000.0f));

        mean = beta1 * mean + (NFloat(1.0f) - beta1) * grad;
        variance = beta2 * variance + (NFloat(1.0f) - beta2) * grad * grad;
        NFloat4 meanHat = mean / (NFloat(1.0f) - pow(beta1, NFloat(iteration)));
        NFloat4 varianceHat = variance / (NFloat4(1.0f) - pow(beta2, NFloat(iteration)));
        param -= learningRate * meanHat / sqrt(max(NFloat(0.0f), varianceHat) + epsilon);
    }
}
