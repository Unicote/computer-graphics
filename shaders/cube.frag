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
    mat4 sun_view_projection;
    mat4 spot_view_projection;
    uint sun_shadows_enabled;
    uint spot_shadows_enabled;
} scene_uniforms;

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

layout(set = 0, binding = 3) uniform sampler2DShadow sun_shadow_map;
layout(set = 0, binding = 4) uniform sampler2DShadow spot_shadow_map;

layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec3 specular;
    float alpha;
    vec2 uv_scale;
} model_uniforms;

layout(set = 1, binding = 1) uniform sampler2D albedo_texture;
layout(set = 1, binding = 2) uniform sampler2D specular_texture;
layout(set = 1, binding = 3) uniform sampler2D emissive_texture;

layout(push_constant) uniform PushConstants {
    float time;
    uint animated_sampling;
    uint show_sun_shadow_underneath;
} push_constants;

layout(location = 0) in vec3 world_position;
layout(location = 1) in vec3 world_normal;
layout(location = 2) in vec2 uv;

layout(location = 0) out vec4 out_color;

vec3 sample_animated(sampler2D image, const vec2 coordinates) {
    if (push_constants.animated_sampling == 0) return texture(image, coordinates).rgb;

    const float t = push_constants.time;
    const vec2 flow = coordinates + vec2(0.1 * t, 0.05 * sin(t));
    const float angle = 0.3 * t;
    const mat2 rotation = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
    const vec2 swirl = rotation * (coordinates - 0.5) * 1.5 + 0.5;

    const vec3 first = texture(image, flow).rgb;
    const vec3 second = texture(image, swirl).rgb;
    const vec3 modulation = 0.75 + 0.25 * vec3(sin(t), sin(t + 2.094), sin(t + 4.189));
    return mix(first, second, 0.5 + 0.5 * sin(0.7 * t)) * modulation;
}

vec3 blinn_phong(const vec3 light_direction, const vec3 light_color, const vec3 normal, const vec3 view_direction, const vec3 albedo, const vec3 specular) {
    const vec3 h = normalize(light_direction + view_direction);
    const float cos_theta = max(dot(normal, light_direction), 0.0);
    const float cos_phi = max(dot(normal, h), 0.0);
    const float highlight = cos_theta > 0.0 ? pow(cos_phi, model_uniforms.alpha) : 0.0;
    return light_color * (cos_theta * albedo + specular * highlight);
}

float shadow_visibility(sampler2DShadow shadow_map, const mat4 light_view_projection) {
    const vec4 light_clip = light_view_projection * vec4(world_position, 1.0);
    if (light_clip.w <= 0.0) return 1.0;
    const vec3 light_ndc = light_clip.xyz / light_clip.w;
    const vec2 shadow_uv = light_ndc.xy * 0.5 + 0.5;
    if (light_ndc.z <= 0.0 || light_ndc.z >= 1.0 || any(lessThan(shadow_uv, vec2(0.0))) || any(greaterThan(shadow_uv, vec2(1.0)))) return 1.0;

    const vec2 texel = 1.0 / vec2(textureSize(shadow_map, 0));
    float lit = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            lit += textureLod(shadow_map, vec3(shadow_uv + vec2(x, y) * texel, light_ndc.z), 0.0);
        }
    }
    return lit / 9.0;
}

void main() {
    const vec3 normal = normalize(world_normal);
    const vec3 view_direction = normalize(scene_uniforms.camera_position - world_position);

    const vec3 albedo = sample_animated(albedo_texture, uv);
    const vec3 specular = model_uniforms.specular * texture(specular_texture, uv).rgb;
    const vec3 emissive = sample_animated(emissive_texture, uv);

    vec3 color = scene_uniforms.ambient_light * albedo;

    const float sun_visibility = scene_uniforms.sun_shadows_enabled != 0 ? shadow_visibility(sun_shadow_map, scene_uniforms.sun_view_projection) : 1.0;
    color += sun_visibility * blinn_phong(-scene_uniforms.directional_light_direction, scene_uniforms.directional_light_color, normal, view_direction, albedo, specular);

    for (uint i = 0; i < scene_uniforms.point_light_count; ++i) {
        const vec3 to_light = point_lights[i].position - world_position;
        const float distance_squared = max(dot(to_light, to_light), 0.0001);
        color += blinn_phong(to_light * inversesqrt(distance_squared), point_lights[i].color / distance_squared, normal, view_direction, albedo, specular);
    }

    for (uint i = 0; i < scene_uniforms.spot_light_count; ++i) {
        const SpotLight light = spot_lights[i];
        const vec3 to_light = light.position - world_position;
        const float distance_squared = max(dot(to_light, to_light), 0.0001);
        const vec3 light_direction = to_light * inversesqrt(distance_squared);

        const float cos_angle = dot(-light_direction, normalize(light.direction));
        const float cos_inner = cos(light.inner_angle);
        const float cos_outer = cos(light.outer_angle);
        const float edge = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);

        if (edge <= 0.0) continue;
        const float spot_visibility = i == 0 && scene_uniforms.spot_shadows_enabled != 0 ? shadow_visibility(spot_shadow_map, scene_uniforms.spot_view_projection) : 1.0;
        const float floor_shadow = push_constants.show_sun_shadow_underneath != 0 && normal.y < -0.5 ? sun_visibility : 1.0;
        color += floor_shadow * spot_visibility * blinn_phong(light_direction, light.color * edge / distance_squared, normal, view_direction, albedo, specular);
    }

    color += emissive;

    out_color = vec4(color, 1.0);
}
