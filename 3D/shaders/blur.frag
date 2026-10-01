#version 330

uniform sampler2D u_depth;
uniform ivec2 u_direction; // vertical or horizontal pass
uniform int u_radius; // blur radius
uniform float u_sigma; // gauss blur constant

const float BACKGROUND = 1e5;

layout(location = 0) out float o_blur;

float gaussian_blur_weight(int offset, float sigma){
    float num = float(offset * offset);
    float c = 2 * sigma * sigma;
    return exp(-num / c);
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(u_depth, 0);

    float centre = texelFetch(u_depth, pixel, 0).r;
    if (centre >= BACKGROUND) {
        o_blur = centre;
        return;
    }

    float sum = 0.0;
    float weightsum = 0.0;

    for(int x = -u_radius; x<=u_radius; x++){
        ivec2 p = clamp(pixel + x * u_direction, ivec2(0), size - 1);
        float d = texelFetch(u_depth, p, 0).r;
        if (d >= BACKGROUND) {
            continue;
        }
        float weight = gaussian_blur_weight(x, u_sigma);
        sum += d * weight;
        weightsum += weight;
    }

    o_blur = sum / weightsum;
}


