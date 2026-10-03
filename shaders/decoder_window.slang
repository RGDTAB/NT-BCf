import common;
import neural_texture;
import target_set;
import mlp_sw;

extern static const int MLP_WIDTH;
extern static const int CHANNELS;

// BC1 textures
uniform Texture2D<float4> t0;
uniform Texture2D<float4> t1;
uniform Texture2D<float4> t2;
uniform Texture2D<float4> t3;
uniform SamplerState bilinear_sampler;
uniform SamplerState nearest_sampler;
bool var_a;

// MLP Decoder
uniform FeedForwardLayer<12, MLP_WIDTH> layer1;
uniform FeedForwardLayer<MLP_WIDTH, CHANNELS> layer2;

uniform RWTexture2D<float4> dst_texture;

// Viewing information
uniform Buffer<uint> texture_channels;
uniform uint mip_level;
uniform bool show_latents;
uniform float zoom;
uniform float2 view_offset;
uniform uint rows;
uniform uint columns;

// Draws requested texture (output, latents) to dst_texture
[shader("compute")]
[numthreads(8, 8, 1)]
void main(int2 tid: SV_DispatchThreadID)
{
    uint2 pos = tid.xy;
    uint w, h;
    dst_texture.GetDimensions(w, h);
    if (pos.x >= w || pos.y >= h) {
        return;
    }
    float2 dims = float2(float(w), float(h));
    float aspect_ratio = dims.x / dims.y;

    uint t_w, t_h;
    t0.GetDimensions(t_w, t_h);
    float2 t_dims = float2(float(t_w), float(t_h));
    float t_aspect_ratio = t_dims.x / t_dims.y;
    t_aspect_ratio *= float(columns) / float(rows);

    // We need to correct the aspect ratio on the uv so that textures don't get stretched out
    float2 uv = float2(pos) / dims;
    // Textures are scaled to fit the screen's width
    uv.y /= aspect_ratio;
    uv.y *= t_aspect_ratio;
    uv += view_offset * zoom;

    // The screen gets divied up to make room for each texture
    uv *= float2(columns, rows);
    uv /= float(zoom);
    uint zone = uint(floor(uv.x) + columns * floor(uv.y));
    // Offsets into the vectors are packed into the texture_channels
    uint zone_offset = texture_channels[zone] >> 3;
    uint zone_channels = min(texture_channels[zone] & 7, 4);
    if (int(floor(uv.x)) >= columns || int(float(uv.y)) >= rows)
    {
        zone_channels = 0;
    }
    uv = frac(uv);

    float4 color = float4(0);
    // Display reconstructed texture
    if (!show_latents)
    {
        // Per Ubisoft paper, smaller latents get a mip offset
        float t0_s, t1_s, t2_s, t3_s;
        if (var_a) {
            t0_s = t1_s = mip_level;
            t2_s = t3_s = mip_level - float(1.0);
        }
        else {
            t0_s = mip_level;
            t1_s = mip_level - float(1.0);
            t2_s = mip_level - float(2.0);
            t3_s = mip_level - float(3.0);
        }

        // Sample the latent textures and pack them into a vector
        let t0_sample = t0.SampleLevel(bilinear_sampler, uv, t0_s);
        let t1_sample = t1.SampleLevel(bilinear_sampler, uv, t1_s);
        let t2_sample = t2.SampleLevel(bilinear_sampler, uv, t2_s);
        let t3_sample = t3.SampleLevel(bilinear_sampler, uv, t3_s);

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

        // MLP decoder
        var layer1_output = layer1.eval(input);
        // ReLU activation
        [ForceUnroll]
        for (int i = 0; i < MLP_WIDTH; i++) {
            layer1_output.data[i] = max(layer1_output.data[i], NFloat(0.0));
        }
        var output = layer2.eval(layer1_output);

        // Fill transparency for textures without a defined alpha channel
        if (zone_channels != 0 && zone_channels < 4)
        {
            color.a = 1.0f;
        }

        for (int i = 0; i < zone_channels; i++)
        {
            color[i] = output.data[i + zone_offset];
        }
    }
    // Display latent textures
    else
    {
        var latent_output = MLVec<12>();
        // We don't perform bilinear filtering here, so you can't inspect each pixel up close
        let t0_sample = t0.SampleLevel(nearest_sampler, uv, mip_level);
        let t1_sample = t1.SampleLevel(nearest_sampler, uv, mip_level);
        let t2_sample = t2.SampleLevel(nearest_sampler, uv, mip_level);
        let t3_sample = t3.SampleLevel(nearest_sampler, uv, mip_level);

        latent_output.data[0] = t0_sample[0];
        latent_output.data[1] = t0_sample[1];
        latent_output.data[2] = t0_sample[2];
        latent_output.data[3] = t1_sample[0];
        latent_output.data[4] = t1_sample[1];
        latent_output.data[5] = t1_sample[2];
        latent_output.data[6] = t2_sample[0];
        latent_output.data[7] = t2_sample[1];
        latent_output.data[8] = t2_sample[2];
        latent_output.data[9] = t3_sample[0];
        latent_output.data[10] = t3_sample[1];
        latent_output.data[11] = t3_sample[2];
        if (zone < 4)
        {
            color.rgb = float3(
                latent_output.data[zone * 3 + 0],
                latent_output.data[zone * 3 + 1],
                latent_output.data[zone * 3 + 2],
            );
            color.a = 1.0f;
        }
    }
    // Alpha blending with the checkerboard
    float4 background = checkerboard(pos);
    color = lerp(background, color, color.a);

    dst_texture[pos] = float4(color.rgb, 1.0f);
}
