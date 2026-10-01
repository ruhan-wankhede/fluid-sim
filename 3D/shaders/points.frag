#version 330

in float v_scalar;
in vec3 v_center;

uniform mat4 u_proj;
uniform float u_radius;

layout(location = 0) out float o_depth;

void main() {

    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    coord.y *= -1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) {
        discard;
    }
    
    // drawing a circle ^
    vec3 n = vec3(coord, sqrt(1.0 - r2));
    vec3 light = normalize(vec3(0.3, 0.6, 1.0));



    //vec3 cold = vec3(0.10, 0.50, 0.85);
    //vec3 hot  = vec3(1.00, 0.35, 0.10);
    //vec3 base = mix(cold, hot, clamp(v_scalar, 0.0, 1.0));

    vec3 surface = v_center + n * u_radius;

    o_depth = -surface.z;
    vec4 clip = u_proj * vec4(surface, 1.0);
    gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;

    /**float dist = -surface.z;
    float t = clamp((dist - 10.0) / 40.0, 0.0, 1.0);
    f_color = vec4(vec3(t), 1.0);**/
    //f_color = vec4(n * 0.5 + 0.5, 1.0);

    //f_color = vec4(base * (0.3 + 0.7 * max(dot(n, light), 0.0)), 1.0);
}


