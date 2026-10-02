#version 330

uniform sampler2D u_depth;
uniform samplerCube u_skybox;
uniform mat4 u_proj;
uniform mat4 u_view;

out vec4 f_color;

const float shininess = 100.0;
const vec3 spec_color = vec3(1.0, 1.0, 1.0);
const float light_power = 1.2;
const float R0 = 0.2;

const vec3 SUN_WORLD = normalize(vec3(0.3, 0.6, 1.0));

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
    //float t = clamp((d - 10.0) / 40.0, 0.0, 1.0);
    //f_color = vec4(vec3(t), 1.0);
    //f_color = vec4(n * 0.5 + 0.5, 1.0);

    vec3 light = normalize(mat3(u_view) * SUN_WORLD);
    vec3 cold = vec3(0.10, 0.5, 0.85);
    //vec3 hot  = vec3(1.00, 0.35, 0.10);
    vec3 base = cold;

    float lambertian = max(dot(n, light), 0.0);
    float specular = 0.0;
    vec3 sky;
    float fresnel;
    if(lambertian > 0.0){
        vec3 view_dir = normalize(-p);
        vec3 half_dir = normalize(light + view_dir);
        float spec_angle = max(dot(half_dir, n), 0.0);
        specular = pow(spec_angle, shininess);

        float cos_theta = max(dot(n, view_dir), 0.0); // n and view_dir are normalized so a.b = cos theta

        fresnel = R0 + (1 - R0) * pow((1 - cos_theta), 5);

        vec3 reflect_view = reflect(-view_dir, n);

        vec3 reflect_world = transpose(mat3(u_view)) * reflect_view;
        sky = texture(u_skybox, reflect_world).rgb;
    }

    vec3 water = base * (0.3 + 0.7 * lambertian);
    f_color = vec4(mix(water, sky, fresnel) * light_power + spec_color * specular * light_power, 1.0);
    //f_color = vec4(sky * light_power + spec_color * specular * light_power, 1.0);
}
