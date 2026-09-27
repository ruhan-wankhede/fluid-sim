#version 330

in vec3 in_pos; // position of vertex
in float in_scalar; // what value to colour based on ( density , velocity etc)

uniform vec2 u_domain; // (WIDTH, HEIGHT) of sim
uniform float u_diameter_px; // circle size (diameter)

out float v_scalar; // varying variable passed to the fragment shader

void main(){
    vec2 ndc = (in_pos.xy / u_domain) * 2.0 - 1.0;

    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = u_diameter_px;
    v_scalar = in_scalar;
}