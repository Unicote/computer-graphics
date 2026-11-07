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

layout(location = 0) in vec3 vertex_color;
layout(location = 1) in vec3 world_position;
layout(location = 2) in vec3 world_normal;
layout(location = 3) flat in uint light_index;

layout(location = 0) out vec4 out_color;

const vec3 specular = vec3(1.0);
const float alpha = 32.0;

vec3 blinn_phong(const vec3 light_direction, const vec3 light_color, const vec3 normal, const vec3 view_direction) {
    const vec3 h = normalize(light_direction + view_direction);
    const float cos_theta = max(dot(normal, light_direction), 0.0);
    const float cos_phi = max(dot(normal, h), 0.0);
    const float highlight = cos_theta > 0.0 ? pow(cos_phi, alpha) : 0.0;
    return light_color * (cos_theta * vertex_color + specular * highlight);
}

void main() {
    const vec3 normal = normalize(world_normal);
    const vec3 view_direction = normalize(scene_uniforms.camera_position - world_position);

    vec3 color = scene_uniforms.ambient_light * vertex_color;

    color += blinn_phong(-normalize(scene_uniforms.directional_light_direction), scene_uniforms.directional_light_color, normal, view_direction);

    for (uint i = 0; i < scene_uniforms.point_light_count; ++i) {
        if (i == light_index) continue;
        const vec3 to_light = point_lights[i].position - world_position;
        const float distance_squared = max(dot(to_light, to_light), 0.0001);
        color += blinn_phong(to_light * inversesqrt(distance_squared), point_lights[i].color / distance_squared, normal, view_direction);
    }

    for (uint i = 0; i < scene_uniforms.spot_light_count; ++i) {
        if (i + scene_uniforms.point_light_count == light_index) continue;
        const SpotLight light = spot_lights[i];
        const vec3 to_light = light.position - world_position;
        const float distance_squared = max(dot(to_light, to_light), 0.0001);
        const vec3 light_direction = to_light * inversesqrt(distance_squared);

        const float cos_angle = dot(-light_direction, normalize(light.direction));
        const float cos_inner = cos(light.inner_angle);
        const float cos_outer = cos(light.outer_angle);
        const float edge = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);

        color += blinn_phong(light_direction, light.color * edge / distance_squared, normal, view_direction);
    }

    out_color = vec4(color, 1.0);
}
