#version 330

uniform sampler2D u_depth;
uniform ivec2 u_direction; // vertical or horizontal pass
uniform float u_radius; // blur radius
uniform float u_sigma; // gauss blur constant
uniform float u_depth_sigma; // bilateral
uniform mat4 u_proj;

const float BACKGROUND = 1e5;

layout(location = 0) out float o_blur;

float gaussian_blur_weight(int offset, float sigma){
    float num = float(offset * offset);
    float c = 2 * sigma * sigma;
    return exp(-num / c);
}

float gaussian_blur(ivec2 pixel, ivec2 size){
    float sum = 0.0;
    float weightsum = 0.0;
    for(int x = -int(u_radius); x<=u_radius; x++){
        ivec2 p = clamp(pixel + x * u_direction, ivec2(0), size - 1);
        float d = texelFetch(u_depth, p, 0).r;
        if (d >= BACKGROUND) {
            continue;
        }
        float weight = gaussian_blur_weight(x, u_sigma);
        sum += d * weight;
        weightsum += weight;
    }

    return sum / weightsum;
}

float bilateral(ivec2 pixel, ivec2 size, float centre){
    float sum = 0.0;
    float weightsum = 0.0;


    int r = int(min(u_radius * u_proj[1][1] * float(size.y) / (2.0 * centre), 30.0));
    float sigma = max(float(r) / 2.0, 0.5);
    for(int x = -r; x<=r; x++){
        ivec2 p = clamp(pixel + x * u_direction, ivec2(0), size - 1);
        float d = texelFetch(u_depth, p, 0).r;
        if (d >= BACKGROUND) {
            continue;
        }

        float w = gaussian_blur_weight(x, sigma);


        float r2 = (d - centre) / u_depth_sigma;
        float g = exp(-r2 * r2);
        sum += d * w * g;
        weightsum += w * g;
    }

    return sum / weightsum;
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(u_depth, 0);

    float centre = texelFetch(u_depth, pixel, 0).r;
    if (centre >= BACKGROUND) {
        o_blur = centre;
        return;
    }


    o_blur = bilateral(pixel, size, centre);
}


