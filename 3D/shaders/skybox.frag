#version 330

in vec3 tex_coords;

out vec4 f_color;

uniform samplerCube skybox;

void main() {
    f_color = texture(skybox, tex_coords);
}
