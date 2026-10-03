module neural_texture;

import common;
import latent_texture;
import mlp_sw;

// Encapsulates the 4 latent textures, and 2 feed-forward layers of a neural texture
public struct NeuralTexture<int width, int channels>
{
    public LatentTexture t0, t1, t2, t3;
    public FeedForwardLayer<12, width> layer1;
    public FeedForwardLayer<width, channels> layer2;
    public bool var_a;

    [Differentiable]
    public MLVec<channels> Eval(NFloat2 uv, NFloat mip)
    {
        // Per Ubisoft paper, smaller latents get a mip offset
        NFloat t0_s, t1_s, t2_s, t3_s;
        if (var_a) {
            t0_s = t1_s = mip;
            t2_s = t3_s = mip - NFloat(1.0);
        }
        else {
            t0_s = mip;
            t1_s = mip - NFloat(1.0);
            t2_s = mip - NFloat(2.0);
            t3_s = mip - NFloat(3.0);
        }
        // Sample the latent textures and pack them into a vector
        let t0_sample = t0.Sample(uv, t0_s);
        let t1_sample = t1.Sample(uv, t1_s);
        let t2_sample = t2.Sample(uv, t2_s);
        let t3_sample = t3.Sample(uv, t3_s);

        MLVec<12> input;
        input.data[0] = t0_sample[0];
        input.data[1] = t0_sample[1];
        input.data[2] = t0_sample[2];
        input.data[3] = t1_sample[0];
        input.data[4] = t1_sample[1];
        input.data[5] = t1_sample[2];
        input.data[6] = t2_sample[0];
        input.data[7] = t2_sample[1];
        input.data[8] = t2_sample[2];
        input.data[9] = t3_sample[0];
        input.data[10] = t3_sample[1];
        input.data[11] = t3_sample[2];

        // MLP decoder - first layer
        var layer1_output = layer1.eval(input);

        // ReLU activation
        [ForceUnroll]
        for (int i = 0; i < width; i++) {
            layer1_output.data[i] = max(layer1_output.data[i], NFloat(0.0));
        }

        // MLP decoder - second layer
        let layer2_output = layer2.eval(layer1_output);

        return layer2_output;
    }
}


// Returns the L2 norm of the loss between the target and neural textures
[Differentiable]
public NFloat L2Loss<int width, int channels>(NeuralTexture<width, channels> texture, no_diff NFloat2 uv, no_diff NFloat mip, no_diff MLVec<channels> target)
{
    let output = texture.Eval(uv, mip);
    NFloat loss = NFloat(0.0f);
    [ForceUnroll]
    for (int i = 0; i < channels; i++) {
        NFloat diff = target.data[i] - output.data[i];
        loss += diff * diff;
    }

    return loss;
}

// Returns the L1 norm of the loss between the target and neural textures
[Differentiable]
public NFloat L1Loss<int width, int channels>(NeuralTexture<width, channels> texture, no_diff NFloat2 uv, no_diff NFloat mip, no_diff MLVec<channels> target)
{
    let output = texture.Eval(uv, mip);
    NFloat loss = NFloat(0.0f);
    [ForceUnroll]
    for (int i = 0; i < channels; i++) {
        NFloat diff = target.data[i] - output.data[i];
        loss += abs(diff);
    }

    return loss;
}

