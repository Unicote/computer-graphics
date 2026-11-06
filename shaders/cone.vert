#version 450

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

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

layout(location = 0) in vec3 in_position;

layout(location = 0) out vec3 vertex_color;

const float cone_length = 2.0;

void main() {
    const SpotLight light = spot_lights[gl_InstanceIndex];

    const vec3 forward = normalize(light.direction);
    const vec3 helper = abs(forward.y) > 0.99 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    const vec3 right = normalize(cross(forward, helper));
    const vec3 up = cross(right, forward);
    const float radius = tan(light.outer_angle) * cone_length;

    const vec3 position = light.position + forward * in_position.z * cone_length + (right * in_position.x + up * in_position.y) * radius;

    gl_Position = scene_uniforms.view_projection * vec4(position, 1.0);
    vertex_color = vec3(1.0);
}
