import common;
import neural_texture;
import target_set;
import mlp_sw;

extern static const int MLP_WIDTH;
extern static const int CHANNELS;

uniform NeuralTexture<MLP_WIDTH, CHANNELS> texture;
uniform TargetSet<CHANNELS> target;
uniform RWTexture2D<float4> dst_texture;
uniform Buffer<uint> texture_channels;
uniform int mip_level;
uniform uint show_loss;
uniform uint show_texture;
uniform float zoom;
uniform float2 view_offset;
uniform uint rows;
uniform uint columns;

// Draws requested texture (target, output, latents) to dst_texture
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
    NFloat2 dims = NFloat2(NFloat(w), NFloat(h));
    NFloat aspect_ratio = dims.x / dims.y;

    uint t_w, t_h;
    texture.t0.aTexture.GetDimensions(t_w, t_h);
    NFloat2 t_dims = NFloat2(NFloat(t_w), NFloat(t_h));
    NFloat t_aspect_ratio = t_dims.x / t_dims.y;
    t_aspect_ratio *= NFloat(columns) / NFloat(rows);

    // We need to correct the aspect ratio on the uv so that textures don't get stretched out
    NFloat2 uv = NFloat2(pos) / dims;
    // Textures are scaled to fit the screen's width
    uv.y /= aspect_ratio;
    uv.y *= t_aspect_ratio;
    uv += view_offset * zoom;

    // The screen gets divied up to make room for each texture
    uv *= NFloat2(columns, rows);
    uv /= NFloat(zoom);
    uint zone = uint(floor(uv.x) + columns * floor(uv.y));
    // Offsets into the vectors are packed into the texture_channels buffer
    uint zone_offset = texture_channels[zone] >> 3;
    uint zone_channels = min(texture_channels[zone] & 7, 4);
    if (int(floor(uv.x)) >= columns || int(float(uv.y)) >= rows)
    {
        zone_channels = 0;
    }
    uv = frac(uv);

    // Decode neural texture, and sample the target texture
    var output = texture.Eval(uv, NFloat(mip_level));
    let target_sample = target.Sample(uv, NFloat(mip_level));

    var latent_output = MLVec<12>();
    // Display some form of loss
    if (show_loss != 0)
    {
        for (int i = 0; i < CHANNELS; i++)
        {
            output.data[i] -= target_sample.data[i];
            // L2
            if (show_loss == 2)
            {
                output.data[i] *= output.data[i];
            }
            // L1
            else
            {
                output.data[i] = abs(output.data[i]);
            }
        }
    }
    // Display target texture
    else if (show_texture == 1)
    {
        for (int i = 0; i < CHANNELS; i++)
        {
            output.data[i] = target_sample.data[i];
        }
    }
    // Display latent texture
    else if (show_texture == 2)
    {
        int t0_s, t1_s, t2_s, t3_s;
        if (texture.var_a) {
            t0_s = t1_s = mip_level;
            t2_s = t3_s = max(mip_level - 1, 0);
        }
        else {
            t0_s = mip_level;
            t1_s = max(mip_level - 1, 0);
            t2_s = max(mip_level - 2, 0);
            t3_s = max(mip_level - 3, 0);
        }
        // We don't perform bilinear filtering here, so you can't inspect each pixel up close
        uint2 t_dims;
        texture.t0.aTexture.GetDimensions(t_dims.x, t_dims.y);
        let t0_sample = texture.t0.LoadTexel(int3(int2(uv * NFloat2(t_dims >> t0_s)), t0_s), 0, 0);
        texture.t1.aTexture.GetDimensions(t_dims.x, t_dims.y);
        let t1_sample = texture.t1.LoadTexel(int3(int2(uv * NFloat2(t_dims >> t1_s)), t1_s), 0, 0);
        texture.t2.aTexture.GetDimensions(t_dims.x, t_dims.y);
        let t2_sample = texture.t2.LoadTexel(int3(int2(uv * NFloat2(t_dims >> t2_s)), t2_s), 0, 0);
        texture.t3.aTexture.GetDimensions(t_dims.x, t_dims.y);
        let t3_sample = texture.t3.LoadTexel(int3(int2(uv * NFloat2(t_dims >> t3_s)), t3_s), 0, 0);
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

        if (zone >= 4)
        {
            zone_channels = 0;
        }
        else
        {
            zone_channels = 3;
            zone_offset = zone * 3;
        }
    }

    float4 color = float4(0);
    // Fill transparency for textures without a defined alpha channel
    if (zone_channels != 0 && zone_channels < 4)
    {
        color.a = 1.0f;
    }

    // Not showing latent textures
    if (show_texture != 2)
    {
        for (int i = 0; i < zone_channels; i++)
        {
            color[i] = output.data[i + zone_offset];
        }
    }
    // Showing latent textures
    else
    {
        for (int i = 0; i < zone_channels; i++)
        {
            color[i] = latent_output.data[i + zone_offset];
        }
    }

    // Alpha blending with the checkerboard
    float4 background = checkerboard(pos);
    color = lerp(background, color, color.a);

    dst_texture[pos] = float4(color.rgb, 1.0f);
}
