#version 330

in vec3 in_position;
out vec3 tex_coords;

uniform mat4 proj;
uniform mat4 view;

void main() {
    tex_coords = in_position;
    mat4 static_view = mat4(mat3(view));
    vec4 pos = proj * static_view * vec4(in_position, 1.0);
    gl_Position = pos.xyww;
}
