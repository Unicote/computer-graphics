#version 450

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    vec3 directional_light_direction;
    vec3 directional_light_color;
    vec3 ambient_light;
    uint point_light_count;
    uint spot_light_count;
} scene_uniforms;

layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec3 specular;
    float alpha;
    vec2 uv_scale;
} model_uniforms;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 0) out vec3 world_position;
layout(location = 1) out vec3 world_normal;
layout(location = 2) out vec2 uv;

void main() {
    const vec4 position = model_uniforms.model * vec4(in_position, 1.0);
    world_position = position.xyz;

    world_normal = transpose(inverse(mat3(model_uniforms.model))) * in_normal;
    uv = in_uv * model_uniforms.uv_scale;
    gl_Position = scene_uniforms.view_projection * position;
}
