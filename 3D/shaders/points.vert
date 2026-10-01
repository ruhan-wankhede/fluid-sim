#version 330

in vec3 in_pos; // position of vertex
in float in_scalar; // what value to colour based on ( density , velocity etc)

uniform mat4 u_view; // cam view matrix
uniform mat4 u_proj; // cam projection matrix
uniform float u_radius; // sphere radius
uniform float u_viewport_h;

out float v_scalar; // varying variable passed to the fragment shader
out vec3 v_center;

void main(){
    vec4 view_pos = u_view * vec4(in_pos, 1.0);
    gl_Position = u_proj * view_pos;
    gl_PointSize = u_viewport_h * u_proj[1][1] * u_radius / gl_Position.w;
    v_scalar = in_scalar;
    v_center = view_pos.xyz;
}