#version 450

struct PointLight {
    vec3 position;
    vec3 color;
};

struct SpotLight {
    vec3 position;
    vec3 color;
    vec3 direction;
    float inner_angle;
    float outer_angle;
};

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    vec3 directional_light_direction;
    vec3 directional_light_color;
    vec3 ambient_light;
    uint point_light_count;
    uint spot_light_count;
} scene_uniforms;

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

layout(location = 0) in vec3 in_position;

layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 vertex_color;
layout(location = 1) out vec3 world_position;
layout(location = 2) out vec3 world_normal;
layout(location = 3) flat out uint light_index;

const float marker_size = 0.1;

void main() {
    uint index = uint(gl_InstanceIndex);
    light_index = index;
    vec3 position;
    vec3 color;
    if (index < scene_uniforms.point_light_count) {
        position = point_lights[index].position;
        color = point_lights[index].color;
    } else {
        index -= scene_uniforms.point_light_count;
        position = spot_lights[index].position;
        color = spot_lights[index].color;
    }

    world_position = position + in_position * marker_size;
    world_normal = in_normal;
    gl_Position = scene_uniforms.view_projection * vec4(world_position, 1.0);
    vertex_color = color / max(max(color.r, max(color.g, color.b)), 0.0001);
}
