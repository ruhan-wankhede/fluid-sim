#version 330

in float v_scalar;
out vec4 f_color;

void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) {
        discard;
    }
    // drawing a circle ^
    vec3 cold = vec3(0.10, 0.50, 0.85);
    vec3 hot  = vec3(1.00, 0.35, 0.10);
    f_color = vec4(mix(cold, hot, clamp(v_scalar, 0.0, 1.0)), 1.0);
    // f_color = vec4(1.0, 0.0, 0.0, 1.0);
}
