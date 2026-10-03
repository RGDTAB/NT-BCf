module mlp_sw;

import common;

__include mlvec_sw;

// Much of this code comes from slang's mlp-training example
// The only major change was moving the ReLU outside of FeedForward layer and into the NeuralTexture
// https://github.com/shader-slang/slang/blob/master/examples/mlp-training/mlp_sw.slang
public struct FeedForwardLayer<int InputSize, int OutputSize>
{
    public Buffer<NFloat> weights;
    public Buffer<NFloat> biases;
    public RWBuffer<float> weightsGrad;
    public RWBuffer<float> biasesGrad;

    [BackwardDerivative(evalBwd)]
    public MLVec<OutputSize> eval(MLVec<InputSize> input)
    {
        var output = matMulAdd<OutputSize>(
            input,
            weights,
            biases);
        return output; 
    }

    public void evalBwd(
        inout DifferentialPair<MLVec<InputSize>> input,
        MLVec<OutputSize> resultGrad)
    {
        let fwd = eval(input.p);

        // Back-prop gradients to the weights matrix.
        outerProductAccumulate(
            resultGrad,
            input.p,
            weightsGrad);

        // Back-prop gradients to the biases vector.
        for (int i = 0; i < OutputSize; i++)
        {
            float grad = float(resultGrad.data[i]);
            // Wave intrinsics for a slight speedup
            grad = WaveActiveSum(grad);
            if (WaveIsFirstLane()) {
                InterlockedAdd<float>(biasesGrad[i], grad);
            }
        }

        // Back-prop gradients to the input vector.
        let dInput = matMulTransposed<InputSize>(resultGrad, weights);

        input = {input.p, dInput};
    }
}

