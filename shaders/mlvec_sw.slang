implementing mlp_sw;

// Much of this code comes from slang's mlp-training example
// https://github.com/shader-slang/slang/blob/master/examples/mlp-training/mlvec_sw.slang
public struct MLVec<int N> : IDifferentiable
{
    public NFloat data[N];
    
    [Differentiable]
    public NFloat[N] toArray()
    {
        return data;
    }

    [Differentiable]
    public static MLVec<N> fromArray(NFloat[N] values)
    {
        MLVec<N> result;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result.data[i] = values[i];
        return result;
    }
}

MLVec<OutputSize> matMulAdd<int OutputSize, int InputSize>(MLVec<InputSize> input, Buffer<NFloat> matrix, Buffer<NFloat> bias)
{
    let getMatElem = (int row, int col) => matrix[row*InputSize + col];
    let getBias = (int idx) => bias[idx];
    MLVec<OutputSize> result = {};
    for (int i = 0; i < OutputSize; i++)
    {
        NFloat r = getBias(i);
        for (int j = 0; j < InputSize; j++)
            r += getMatElem(i, j) * input.data[j];
        result.data[i] = r;
    }
    return result;
}

MLVec<OutputSize> matMulTransposed<int OutputSize, int InputSize>(MLVec<InputSize> input, Buffer<NFloat> matrix)
{
    let getMatElem = (int row, int col) => matrix[col*OutputSize + row];
    MLVec<OutputSize> result = {};
    for (int i = 0; i < OutputSize; i++)
    {
        NFloat r = {};
        for (int j = 0; j < InputSize; j++)
            r += getMatElem(i, j) * input.data[j];
        result.data[i] = r;
    }
    return result;
}

void outerProductAccumulate<int M, int N>(MLVec<M> v0, MLVec<N> v1, RWBuffer<float> matrix)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float elem = float(v0.data[i] * v1.data[j]);
            // Wave intrinsics for a major speedup - very hot data
            elem = WaveActiveSum(elem);
            if (WaveIsFirstLane()) {
                InterlockedAdd<float>(matrix[i*N + j], elem);
            }
        }
    }
}

// These operator functions were added on afterwards to make interpolation easier
public MLVec<N> operator+<int N>(MLVec<N> v0, MLVec<N> v1)
{
    MLVec<N> result = {};
    [ForceUnroll]
    for (int i = 0; i < N; i++)
    {
        result.data[i] = v0.data[i] + v1.data[i];
    }
    return result;
}

public MLVec<N> operator*<int N>(MLVec<N> v0, MLVec<N> v1)
{
    MLVec<N> result = {};
    [ForceUnroll]
    for (int i = 0; i < N; i++)
    {
        result.data[i] = v0.data[i] * v1.data[i];
    }
    return result;
}

public MLVec<N> operator+<int N>(MLVec<N> v0, NFloat val)
{
    MLVec<N> result = {};
    [ForceUnroll]
    for (int i = 0; i < N; i++)
    {
        result.data[i] = v0.data[i] + val;
    }
    return result;
}

public MLVec<N> operator*<int N>(MLVec<N> v0, NFloat val)
{
    MLVec<N> result = {};
    [ForceUnroll]
    for (int i = 0; i < N; i++)
    {
        result.data[i] = v0.data[i] * val;
    }
    return result;
}

