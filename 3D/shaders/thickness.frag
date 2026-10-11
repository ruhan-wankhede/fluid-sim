#version 330

in vec3 v_center;


uniform float u_radius;
uniform float u_volume_scale;

layout(location = 0) out float o_thickness;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if(r2 > 1.0){
        discard;
    }

    o_thickness = 2.0 * u_radius * sqrt(1.0 - r2) * u_volume_scale;
}
