#version 330


in vec2 v_uv;

uniform sampler2D u_depth;
uniform samplerCube u_skybox;
uniform sampler2D thickness;
uniform mat4 u_proj;
uniform mat4 u_view;

out vec4 f_color;

const float shininess = 100.0;
const vec3 spec_color = vec3(1.0, 1.0, 1.0);
const float light_power = 1.2;
const float R0 = 0.02;
const float gamma = 0.01;

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


    float lambertian = max(dot(n, light), 0.0);
    float specular = 0.0;
    vec3 sky;
    float fresnel;

    vec3 view_dir = normalize(-p);
    float cos_theta = max(dot(n, view_dir), 0.0); // n and view_dir are normalized so a.b = cos theta
    fresnel = R0 + (1 - R0) * pow((1 - cos_theta), 5);
    vec3 reflect_view = reflect(-view_dir, n);
    vec3 reflect_world = transpose(mat3(u_view)) * reflect_view;
    sky = texture(u_skybox, reflect_world).rgb;

    if(lambertian > 0.0){
        vec3 half_dir = normalize(light + view_dir);
        float spec_angle = max(dot(half_dir, n), 0.0);
        specular = pow(spec_angle, shininess);
    }

    vec3 water_color = vec3(0.0, 0.0, 0.0);

    float thickness = texture(thickness, v_uv).r;
    float beta = thickness * gamma;

    vec2 distortedUV = v_uv + beta * min(thickness, 5.0) * n.xy;

    vec2 ndc = distortedUV * 2.0 - 1.0;
    vec3 ray_view = normalize(vec3(ndc.x / u_proj[0][0], ndc.y / u_proj[1][1], -1.0));
    vec3 background = texture(u_skybox, transpose(mat3(u_view)) * ray_view).rgb;
    //vec3 background = texture(u_skybox, v_uv + beta * n.xy).rgb;
    vec3 SIGMA = vec3(2.12, 0.43, 0.30) * 0.2; // straight from lague
    // sigma basically decides how much the specific colour of light gets absorbed while passing through the liquid
    // so a high value for R means less red light will pass through as it makes its way through the liquid

    SIGMA = vec3(2.0, 0.4, 0.3) * 0.2; // water
    //SIGMA = vec3(0.3, 1.0, 1.0) * 0.2; // blood
    //vec3 refracted = water_color + (background - water_color) * exp(-thickness * SIGMA);
    vec3 refracted = background * exp(-thickness * SIGMA);
    //vec3 water = refracted * (0.3 + 0.7 * lambertian);
    vec3 water = refracted;
    f_color = vec4(mix(water, sky, fresnel) + spec_color * specular * light_power, 1.0);
    //f_color = vec4(sky * light_power + spec_color * specular * light_power, 1.0);
}
