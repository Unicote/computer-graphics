#version 450

layout(push_constant) uniform ShadowPass {
    mat4 light_view_projection;
} shadow;

layout(std140, set = 0, binding = 0) uniform ModelUniform {
    mat4 model;
} model_uniforms;

layout(location = 0) in vec3 in_position;

void main() {
    gl_Position = shadow.light_view_projection * model_uniforms.model * vec4(in_position, 1.0);
}
