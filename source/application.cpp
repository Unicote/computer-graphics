#include "application.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <numbers>

namespace {

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
};

struct SceneUniforms {
	glm::mat4 view_projection;
	alignas(16) glm::vec3 camera_position;
	alignas(16) glm::vec3 directional_light_direction;
	alignas(16) glm::vec3 directional_light_color;
	alignas(16) glm::vec3 ambient_light;
	uint32_t point_light_count;
	uint32_t spot_light_count;
};

struct ModelUniforms {
	glm::mat4 model;
	alignas(16) glm::vec3 albedo;
	alignas(16) glm::vec3 specular;
	float alpha;
};

struct PointLight {
	alignas(16) glm::vec3 position;
	alignas(16) glm::vec3 color;
};

struct SpotLight {
	alignas(16) glm::vec3 position;
	alignas(16) glm::vec3 color;
	alignas(16) glm::vec3 direction;
	float inner_angle;
	float outer_angle;
};

const glm::vec3 cube_corners[] = {
	{ -1, -1, -1 },
	{ +1, -1, -1 },
	{ +1, +1, -1 },
	{ -1, +1, -1 },
	{ -1, -1, +1 },
	{ +1, -1, +1 },
	{ +1, +1, +1 },
	{ -1, +1, +1 },
};

const uint32_t cube_indices[] = {
	0, 2, 1,  0, 3, 2,
	4, 5, 6,  4, 6, 7,
	0, 4, 7,  0, 7, 3,
	1, 2, 6,  1, 6, 5,
	0, 1, 5,  0, 5, 4,
	3, 7, 6,  3, 6, 2,
};

constexpr uint32_t cone_segments = 32;
constexpr uint32_t cone_rays = 8;
constexpr uint32_t sphere_rings = 12;
constexpr uint32_t sphere_sectors = 24;

Vertex cube_vertices[std::size(cube_indices)];
Vertex cone_vertices[(cone_segments + cone_rays) * 2];
Vertex sphere_vertices[sphere_rings * sphere_sectors * 6];

constexpr uint32_t object_count = 3;
constexpr size_t max_lights = 16;

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
};

struct Scene {
	Buffer cube_buffer;
	Buffer cone_buffer;

	Buffer scene_uniform_buffer;
	Buffer point_light_buffer;
	Buffer spot_light_buffer;
	Buffer model_uniform_buffers[object_count];

	VkDescriptorSetLayout scene_descriptor_layout = VK_NULL_HANDLE;
	VkDescriptorSetLayout model_descriptor_layout = VK_NULL_HANDLE;
	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
	VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;
	VkDescriptorSet model_descriptor_sets[object_count]{};
	VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

	VkPipeline lit_pipeline = VK_NULL_HANDLE;
	VkPipeline light_pipeline = VK_NULL_HANDLE;
	VkPipeline cone_pipeline = VK_NULL_HANDLE;

	uint32_t point_light_count = 0;
	uint32_t spot_light_count = 0;
};

struct Camera {
	glm::vec3 position{0, 2, 7};
	float yaw = 0;
	float pitch = -8;
	float fov = 120;
};

struct Object {
	const char* name;
	glm::vec3 position;
	glm::vec3 rotation;
	glm::vec3 scale;
	glm::vec3 albedo;
	glm::vec3 specular;
	float alpha;
};

struct PointLightSettings {
	glm::vec3 position{0, 1, 3};
	glm::vec3 color{1, 1, 1};
	float intensity = 5;
};

struct SpotLightSettings {
	glm::vec3 position{0, 3, 0};
	glm::vec3 direction{0, -1, 0};
	glm::vec3 color{1, 1, 1};
	float intensity = 15;
	float inner_angle = 15;
	float outer_angle = 25;
};

struct Settings {
	Camera camera;
	bool perspective = true;

	bool animation_playing = true;
	float animation_speed = 1;
	float radius = 1;
	float height = 1;
	float depth = 1;
	float phase = 1;

	glm::vec3 ambient_light{0.08, 0.08, 0.1};
	glm::vec3 directional_light_direction{-0.5, -1, -0.3};
	glm::vec3 directional_light_color{1, 0.95, 0.9};
	float directional_light_intensity = 0.4;

	Object objects[object_count] = {
		{ .name = "Primary cube", .position = {}, .rotation = {}, .scale = glm::vec3(1), .albedo = {0.9, 0.4, 0.2}, .specular = glm::vec3(0.6), .alpha = 32 },
		{ .name = "Second cube", .position = {3.5, -2, 0}, .rotation = {}, .scale = glm::vec3(0.5), .albedo = {0.2, 0.5, 0.9}, .specular = glm::vec3(1), .alpha = 128 },
		{ .name = "Floor", .position = {0, -2.6, 0}, .rotation = {}, .scale = {10, 0.1, 10}, .albedo = glm::vec3(0.6), .specular = glm::vec3(0.2), .alpha = 8 },
	};

	std::vector<PointLightSettings> point_lights = {
		{ .position = {-2, 2, 0}, .color = {0.3, 0.6, 1}, .intensity = 6 },
	};
	std::vector<SpotLightSettings> spot_lights = {
		{ .position = {2, 3, 2}, .direction = {-0.3, -1, -0.4}, .color = {1, 0.85, 0.6}, .intensity = 20 },
	};
	bool show_lights = true;
};

Scene scene;
Settings controls;
double previous_time = 0.0;

// region helpers

std::vector<uint32_t> read_spirv(const char* path) {
	std::ifstream file(path, std::ios::ate | std::ios::binary);
	if (!file) {
		std::cerr << "Cannot open shader file: " << path << '\n';
		return {};
	}

	const std::streamsize bytes = file.tellg();
	std::vector<uint32_t> code(static_cast<std::size_t>(bytes) / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), bytes);
	return code;
}

VkShaderModule load_shader(const char* path) {
	const std::vector<uint32_t> code = read_spirv(path);
	if (code.empty()) return VK_NULL_HANDLE;

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size() * sizeof(uint32_t),
		.pCode = code.data(),
	};
	VkShaderModule shader = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &shader) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module: " << path << '\n';
		return VK_NULL_HANDLE;
	}
	return shader;
}

bool create_buffer(Buffer& buffer, const VkDeviceSize size, const VkBufferUsageFlags usage) {
	const VkBufferCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	constexpr VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	VmaAllocationInfo allocation_result{};
	if (vmaCreateBuffer(graphics::internal::context.allocator, &info, &allocation_info, &buffer.buffer, &buffer.allocation, &allocation_result) != VK_SUCCESS) {
		std::cerr << "Failed to create buffer\n";
		return false;
	}
	buffer.mapped = allocation_result.pMappedData;
	return true;
}

void destroy_buffer(Buffer& buffer) {
	if (buffer.buffer != VK_NULL_HANDLE) vmaDestroyBuffer(graphics::internal::context.allocator, buffer.buffer, buffer.allocation);
	buffer = {};
}

void upload(const Buffer& buffer, const void* data, const VkDeviceSize size) {
	std::memcpy(buffer.mapped, data, size);
	vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, 0, size);
}

void write_buffer_descriptor(VkDescriptorSet set, const uint32_t binding, const VkDescriptorType type, const Buffer& buffer) {
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = buffer.buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE,
	};
	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = binding,
		.descriptorCount = 1,
		.descriptorType = type,
		.pBufferInfo = &buffer_info,
	};
	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

glm::vec3 safe_direction(const glm::vec3& direction) {
	return glm::length(direction) > 0.0001f ? glm::normalize(direction) : glm::vec3(0, -1, 0);
}

	// endregion

glm::mat4 model_matrix(const glm::vec3& position, const glm::vec3& rotation, const glm::vec3& scale, const glm::vec3 offset = {}, const glm::vec3 animation_rotation = {}) {
	constexpr glm::mat4 identity(1.0);
	const glm::vec3 angles = rotation + animation_rotation;
	return glm::translate(identity, position + offset) * glm::rotate(identity, glm::radians(angles.z), glm::vec3(0.0, 0.0, 1.0)) * glm::rotate(identity, glm::radians(angles.y), glm::vec3(0.0, 1.0, 0.0)) * glm::rotate(identity, glm::radians(angles.x), glm::vec3(1.0, 0.0, 0.0)) * glm::scale(identity, scale);
}

glm::mat4 camera_rotation() {
	constexpr glm::mat4 identity(1.0);
	return glm::rotate(identity, glm::radians(controls.camera.yaw), glm::vec3(0.0, 1.0, 0.0)) * glm::rotate(identity, glm::radians(controls.camera.pitch), glm::vec3(1.0, 0.0, 0.0));
}

void generate_cube() {
	for (size_t i = 0; i < std::size(cube_indices); i += 3) {
		const glm::vec3 a = cube_corners[cube_indices[i]];
		const glm::vec3 b = cube_corners[cube_indices[i + 1]];
		const glm::vec3 c = cube_corners[cube_indices[i + 2]];
		const glm::vec3 normal = glm::normalize(glm::cross(b - a, c - a));
		cube_vertices[i] = { .position = a, .normal = normal };
		cube_vertices[i + 1] = { .position = b, .normal = normal };
		cube_vertices[i + 2] = { .position = c, .normal = normal };
	}
}

void generate_cone() {
	size_t v = 0;
	const auto circle = [](const uint32_t i) {
		const float angle = 2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / cone_segments;
		return glm::vec3(std::cos(angle), std::sin(angle), 1);
	};
	for (uint32_t i = 0; i < cone_segments; ++i) {
		cone_vertices[v++] = { .position = circle(i) };
		cone_vertices[v++] = { .position = circle(i + 1) };
	}
	for (uint32_t i = 0; i < cone_rays; ++i) {
		cone_vertices[v++] = { .position = {} };
		cone_vertices[v++] = { .position = circle(i * cone_segments / cone_rays) };
	}
}

void destroy_resources() {
	const auto& context = graphics::internal::context;
	if (scene.cone_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, scene.cone_pipeline, nullptr);
	if (scene.light_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, scene.light_pipeline, nullptr);
	if (scene.lit_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, scene.lit_pipeline, nullptr);
	if (scene.pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(context.device, scene.pipeline_layout, nullptr);
	if (scene.descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(context.device, scene.descriptor_pool, nullptr);
	if (scene.model_descriptor_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context.device, scene.model_descriptor_layout, nullptr);
	if (scene.scene_descriptor_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context.device, scene.scene_descriptor_layout, nullptr);
	for (Buffer& buffer : scene.model_uniform_buffers) destroy_buffer(buffer);
	destroy_buffer(scene.spot_light_buffer);
	destroy_buffer(scene.point_light_buffer);
	destroy_buffer(scene.scene_uniform_buffer);
	destroy_buffer(scene.cone_buffer);
	destroy_buffer(scene.cube_buffer);
	scene = {};
}

bool create_descriptors() {
	const auto& context = graphics::internal::context;
	constexpr VkShaderStageFlags stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	constexpr VkDescriptorSetLayoutBinding scene_bindings[] = {
		{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = stages },
		{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = stages },
		{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = stages },
	};
	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = std::size(scene_bindings),
		.pBindings = scene_bindings,
	};
	if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr, &scene.scene_descriptor_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create scene descriptor set layout\n";
		return false;
	}

	constexpr VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = stages,
	};
	const VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};
	if (vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr, &scene.model_descriptor_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create model descriptor set layout\n";
		return false;
	}

	constexpr VkDescriptorPoolSize pool_sizes[] = {
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 + object_count },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + object_count,
		.poolSizeCount = std::size(pool_sizes),
		.pPoolSizes = pool_sizes,
	};
	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &scene.descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetLayout layouts[] = {
		scene.scene_descriptor_layout,
		scene.model_descriptor_layout,
		scene.model_descriptor_layout,
		scene.model_descriptor_layout,
	};
	static_assert(std::size(layouts) == 1 + object_count);
	VkDescriptorSet descriptor_sets[1 + object_count]{};
	const VkDescriptorSetAllocateInfo allocation_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = scene.descriptor_pool,
		.descriptorSetCount = std::size(layouts),
		.pSetLayouts = layouts,
	};
	if (vkAllocateDescriptorSets(context.device, &allocation_info, descriptor_sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}

	scene.scene_descriptor_set = descriptor_sets[0];
	write_buffer_descriptor(scene.scene_descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, scene.scene_uniform_buffer);
	write_buffer_descriptor(scene.scene_descriptor_set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, scene.point_light_buffer);
	write_buffer_descriptor(scene.scene_descriptor_set, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, scene.spot_light_buffer);
	for (uint32_t i = 0; i < object_count; ++i) {
		scene.model_descriptor_sets[i] = descriptor_sets[1 + i];
		write_buffer_descriptor(scene.model_descriptor_sets[i], 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, scene.model_uniform_buffers[i]);
	}

	const VkDescriptorSetLayout pipeline_set_layouts[] = {
		scene.scene_descriptor_layout,
		scene.model_descriptor_layout,
	};
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = std::size(pipeline_set_layouts),
		.pSetLayouts = pipeline_set_layouts,
	};
	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &scene.pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}
	return true;
}

bool create_pipeline(VkPipeline& pipeline, const char* vertex_path, const char* fragment_path, const VkPrimitiveTopology topology) {
	const auto& context = graphics::internal::context;
	auto vertex_shader = load_shader(vertex_path);
	auto fragment_shader = load_shader(fragment_path);
	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		if (vertex_shader != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		if (fragment_shader != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertex_shader, .pName = "main" },
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragment_shader, .pName = "main" },
	};
	constexpr VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	constexpr VkVertexInputAttributeDescription vertex_attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = std::size(vertex_attributes),
		.pVertexAttributeDescriptions = vertex_attributes,
	};
	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = topology,
	};
	constexpr VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	constexpr VkPipelineRasterizationStateCreateInfo rasterizer = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0,
	};

	constexpr VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	constexpr VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};
	constexpr VkPipelineColorBlendAttachmentState color_attachment = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};
	const VkPipelineColorBlendStateCreateInfo color_blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_attachment,
	};

	constexpr VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = std::size(dynamic_states),
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = std::size(stages),
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blending,
		.pDynamicState = &dynamic_state,
		.layout = scene.pipeline_layout,
		.renderPass = context.render_pass,
	};
	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline: " << vertex_path << '\n';
		return false;
	}
	return true;
}

bool create_resources() {
	generate_cube();
	generate_cone();

	if (!create_buffer(scene.cube_buffer, sizeof(cube_vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) || !create_buffer(scene.cone_buffer, sizeof(cone_vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) return false;
	if (!create_buffer(scene.scene_uniform_buffer, sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) return false;
	if (!create_buffer(scene.point_light_buffer, max_lights * sizeof(PointLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) || !create_buffer(scene.spot_light_buffer, max_lights * sizeof(SpotLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) return false;
	for (Buffer& buffer : scene.model_uniform_buffers) if (!create_buffer(buffer, sizeof(ModelUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) return false;

	upload(scene.cube_buffer, cube_vertices, sizeof(cube_vertices));
	upload(scene.cone_buffer, cone_vertices, sizeof(cone_vertices));
	return create_descriptors()
		&& create_pipeline(scene.lit_pipeline, "shaders/cube.vert.spv", "shaders/cube.frag.spv", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
		&& create_pipeline(scene.light_pipeline, "shaders/light.vert.spv", "shaders/marker.frag.spv", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
		&& create_pipeline(scene.cone_pipeline, "shaders/cone.vert.spv", "shaders/light.frag.spv", VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
}

void update_uniforms() {
	const auto [width, height] = graphics::internal::context.swapchain_extent;
	if (!height) return;

	const float aspect = static_cast<float>(width) / static_cast<float>(height);
	glm::mat4 projection = controls.perspective ? glm::perspective(glm::radians(static_cast<double>(controls.camera.fov)), static_cast<double>(aspect), 0.1, 100.0) : glm::ortho(-3.0 * aspect, 3.0 * aspect, -3.0, 3.0, 0.1, 100.0);
	projection[1][1] *= -1.0;

	constexpr glm::mat4 identity(1.0);
	const glm::mat4 view = glm::transpose(camera_rotation()) * glm::translate(identity, -controls.camera.position);

	std::vector<PointLight> point_lights;
	point_lights.reserve(controls.point_lights.size());
	for (const auto&[position, color, intensity] : controls.point_lights) {
		point_lights.push_back({ .position = position, .color = color * intensity });
	}
	std::vector<SpotLight> spot_lights;
	spot_lights.reserve(controls.spot_lights.size());
	for (const auto&[position, direction, color, intensity, inner_angle, outer_angle] : controls.spot_lights) {
		spot_lights.push_back({
			.position = position,
			.color = color * intensity,
			.direction = safe_direction(direction),
			.inner_angle = glm::radians(inner_angle),
			.outer_angle = glm::radians(std::max(inner_angle, outer_angle)),
		});
	}

	scene.point_light_count = static_cast<uint32_t>(point_lights.size());
	scene.spot_light_count = static_cast<uint32_t>(spot_lights.size());
	if (scene.point_light_count) upload(scene.point_light_buffer, point_lights.data(), scene.point_light_count * sizeof(PointLight));
	if (scene.spot_light_count) upload(scene.spot_light_buffer, spot_lights.data(), scene.spot_light_count * sizeof(SpotLight));

	const SceneUniforms scene_uniforms{
		.view_projection = projection * view,
		.camera_position = controls.camera.position,
		.directional_light_direction = safe_direction(controls.directional_light_direction),
		.directional_light_color = controls.directional_light_color * controls.directional_light_intensity,
		.ambient_light = controls.ambient_light,
		.point_light_count = scene.point_light_count,
		.spot_light_count = scene.spot_light_count,
	};
	upload(scene.scene_uniform_buffer, &scene_uniforms, sizeof(scene_uniforms));

	const float phase = controls.phase;
	const glm::vec3 trajectory(controls.radius * std::cos(phase),controls.height * std::sin(phase),controls.depth * std::sin(phase));
	const glm::vec3 animated_rotation(30* std::sin(phase),glm::degrees(phase),std::cos(phase));

	for (uint32_t i = 0; i < object_count; ++i) {
		const auto object = controls.objects[i];
		const auto animated = i == 0;
		const ModelUniforms model_uniforms{
			.model = model_matrix(object.position, object.rotation, object.scale, animated ? trajectory : glm::vec3(), animated ? animated_rotation : glm::vec3()),
			.albedo = object.albedo,
			.specular = object.specular,
			.alpha = object.alpha,
		};
		upload(scene.model_uniform_buffers[i], &model_uniforms, sizeof(model_uniforms));
	}
}

}

namespace application {

bool initialize() {
	if (!create_resources()) {
		destroy_resources();
		return false;
	}
	update_uniforms();
	return true;
}

void shutdown() {
	vkQueueWaitIdle(graphics::internal::context.graphics_queue);
	destroy_resources();
}

void update(const double time) {
	const float delta_time = previous_time == 0.0 ? 0.0 : static_cast<float>(time - previous_time);
	previous_time = time;
	if (controls.animation_playing) {
		controls.phase = std::fmod(controls.phase + delta_time * controls.animation_speed,2.0f * std::numbers::pi_v<float>);
	}

	ImGui::Begin("Lab 2");

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::RadioButton("Orthographic", !controls.perspective)) controls.perspective = false;
		ImGui::SameLine();
		if (ImGui::RadioButton("Perspective", controls.perspective)) controls.perspective = true;
		ImGui::DragFloat3("Position", glm::value_ptr(controls.camera.position), 0.05);
		ImGui::SliderFloat("Yaw", &controls.camera.yaw, -180.0, 180.0);
		ImGui::SliderFloat("Pitch", &controls.camera.pitch, -89.0, 89.0);
		ImGui::SliderFloat("FOV", &controls.camera.fov, 20.0, 120.0);
	}

	if (ImGui::CollapsingHeader("Ambient and directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Ambient", glm::value_ptr(controls.ambient_light));
		ImGui::DragFloat3("Direction", glm::value_ptr(controls.directional_light_direction), 0.01);
		ImGui::ColorEdit3("Color", glm::value_ptr(controls.directional_light_color));
		ImGui::SliderFloat("Intensity", &controls.directional_light_intensity, 0.0, 3.0);
	}

	for (auto& object : controls.objects) {
		ImGui::PushID(object.name);
		if (ImGui::CollapsingHeader(object.name)) {
			ImGui::DragFloat3("Position", glm::value_ptr(object.position), 0.02);
			ImGui::DragFloat3("Rotation", glm::value_ptr(object.rotation), 0.5);
			ImGui::DragFloat3("Scale", glm::value_ptr(object.scale), 0.01, 0.1, 20.0);
			ImGui::ColorEdit3("Albedo", glm::value_ptr(object.albedo));
			ImGui::ColorEdit3("Specular", glm::value_ptr(object.specular));
			ImGui::SliderFloat("Shininess", &object.alpha, 1.0, 512.0, "%.0f", ImGuiSliderFlags_Logarithmic);
		}
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Animation")) {
		ImGui::Checkbox("Play animation", &controls.animation_playing);
		ImGui::SameLine();
		if (ImGui::Button("Restart animation")) controls.phase = 0.0;
		ImGui::SliderFloat("Animation speed", &controls.animation_speed, 0.0, 4.0, "%.2fx");
		ImGui::SliderFloat("Trajectory radius", &controls.radius, 0.0, 2.0);
		ImGui::SliderFloat("Trajectory height", &controls.height, 0.0, 2.0);
		ImGui::SliderFloat("Trajectory depth", &controls.depth, 0.0, 2.0);
	}

	ImGui::Separator();
	ImGui::Checkbox("Show lights", &controls.show_lights);

	if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("Point lights");
		for (size_t i = 0; i < controls.point_lights.size(); ++i) {
			PointLightSettings& light = controls.point_lights[i];
			ImGui::PushID(static_cast<int>(i));
			ImGui::Text("Point light %zu", i);
			ImGui::SameLine();
			const bool remove = ImGui::SmallButton("Remove");
			ImGui::DragFloat3("Position", glm::value_ptr(light.position), 0.02);
			ImGui::ColorEdit3("Color", glm::value_ptr(light.color));
			ImGui::SliderFloat("Intensity", &light.intensity, 0.0, 50.0);
			ImGui::PopID();
			if (remove) controls.point_lights.erase(controls.point_lights.begin() + static_cast<std::ptrdiff_t>(i--));
		}
		if (controls.point_lights.size() < max_lights && ImGui::Button("Add point light")) controls.point_lights.push_back({});
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("Spot lights");
		for (size_t i = 0; i < controls.spot_lights.size(); ++i) {
			auto&[position, direction, color, intensity, inner_angle, outer_angle] = controls.spot_lights[i];
			ImGui::PushID(static_cast<int>(i));
			ImGui::Text("Spot light %zu", i);
			ImGui::SameLine();
			const auto remove = ImGui::SmallButton("Remove");
			ImGui::DragFloat3("Position", glm::value_ptr(position), 0.02);
			ImGui::DragFloat3("Direction", glm::value_ptr(direction), 0.01);
			ImGui::ColorEdit3("Color", glm::value_ptr(color));
			ImGui::SliderFloat("Intensity", &intensity, 0.0, 100.0);
			ImGui::SliderFloat("Inner angle", &inner_angle, 1.0, 80.0);
			ImGui::SliderFloat("Outer angle", &outer_angle, inner_angle, 80.0);
			ImGui::PopID();
			if (remove) controls.spot_lights.erase(controls.spot_lights.begin() + static_cast<std::ptrdiff_t>(i--));
		}
		if (controls.spot_lights.size() < max_lights && ImGui::Button("Add spot light")) controls.spot_lights.push_back({});
		ImGui::PopID();
	}

	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	const auto& context = graphics::internal::context;
	update_uniforms();
	vkResetCommandBuffer(fd.command_buffer, 0);

	constexpr VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};

	vkBeginCommandBuffer(fd.command_buffer, &begin_info);

	constexpr VkClearValue clear_values[] = {
		{ .color = {{0.02, 0.02, 0.03, 1.0}} },
		{ .depthStencil = {.depth = 1.0, .stencil = 0} },
	};
	const VkRenderPassBeginInfo render_pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = std::size(clear_values),
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(fd.command_buffer, &render_pass, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.x = 0,.y =0,
		.width = static_cast<float>(context.swapchain_extent.width),
		.height = static_cast<float>(context.swapchain_extent.height),
		.minDepth = 0 ,.maxDepth = 1,
	};
	const VkRect2D scissor = { .extent = context.swapchain_extent };
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
	constexpr VkDeviceSize offset = 0;

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, scene.lit_pipeline);
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &scene.cube_buffer.buffer, &offset);
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, scene.pipeline_layout, 0, 1, &scene.scene_descriptor_set, 0, nullptr);
	for (auto model_set: scene.model_descriptor_sets) {
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, scene.pipeline_layout, 1, 1, &model_set, 0, nullptr);
		vkCmdDraw(fd.command_buffer, std::size(cube_vertices), 1, 0, 0);
	}

	if (controls.show_lights) {
		if (const auto light_count = scene.point_light_count + scene.spot_light_count) {
			vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, scene.light_pipeline);
			vkCmdDraw(fd.command_buffer, std::size(cube_vertices), light_count, 0, 0);
		}
		if (scene.spot_light_count) {
			vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, scene.cone_pipeline);
			vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &scene.cone_buffer.buffer, &offset);
			vkCmdDraw(fd.command_buffer, std::size(cone_vertices), scene.spot_light_count, 0, 0);
		}
	}

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

}
