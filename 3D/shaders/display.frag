#version 330

uniform sampler2D u_depth;
uniform mat4 u_proj;

out vec4 f_color;

vec3 view_pos(ivec2 tex_coord, ivec2 size){
    float d = texelFetch(u_depth, tex_coord, 0).r;
    vec2 ndc = (vec2(tex_coord) + 0.5) / vec2(size) * 2.0 - 1.0;
    return vec3(ndc.x * d / u_proj[0][0], ndc.y * d / u_proj[1][1], -d);
}

void main() {

    ivec2 tex_coord = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(u_depth, 0);
    float d = texelFetch(u_depth, tex_coord, 0).r;
    if (d >= 1e5) {
        discard;
    }


    vec3 p = view_pos(tex_coord, size);
    vec3 ddx = view_pos(tex_coord + ivec2(1, 0), size) - p;
    vec3 ddx2 = p - view_pos(tex_coord - ivec2(1, 0), size);
    if(abs(ddx.z) > abs(ddx2.z)){
        ddx = ddx2;
    }

    vec3 ddy = view_pos(tex_coord + ivec2(0, 1), size) - p;
    vec3 ddy2 = p - view_pos(tex_coord - ivec2(0, 1), size);
    if(abs(ddy.z) > abs(ddy2.z)){
        ddy = ddy2;
    }

    vec3 n = normalize(cross(ddx, ddy));
    float t = clamp((d - 10.0) / 40.0, 0.0, 1.0);
    f_color = vec4(vec3(t), 1.0);
    f_color = vec4(n * 0.5 + 0.5, 1.0);
}
