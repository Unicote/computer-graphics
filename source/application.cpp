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
#include <lodepng.h>
#include <numbers>

namespace {

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 uv;
};

struct SceneUniforms {
	glm::mat4 view_projection;
	alignas(16) glm::vec3 camera_position;
	alignas(16) glm::vec3 directional_light_direction;
	alignas(16) glm::vec3 directional_light_color;
	alignas(16) glm::vec3 ambient_light;
	uint32_t point_light_count;
	uint32_t spot_light_count;
	alignas(16) glm::mat4 sun_view_projection;
	alignas(16) glm::mat4 spot_view_projection;
	uint32_t sun_shadows_enabled;
	uint32_t spot_shadows_enabled;
};

struct ModelUniforms {
	glm::mat4 model;
	alignas(16) glm::vec3 specular;
	float alpha;
	glm::vec2 uv_scale;
};

struct PushConstants {
	float time;
	uint32_t animated_sampling;
	uint32_t show_sun_shadow_underneath;
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

struct Material {
	const char* name;
	const char* albedo;
	const char* specular;
	const char* emissive;
};

constexpr Material materials[] = {
	{ .name = "Dirt", .albedo = "textures/dirt.png", .specular = nullptr, .emissive = nullptr },
	{ .name = "Stone bricks", .albedo = "textures/stone_bricks.png", .specular = "textures/stone_bricks.png", .emissive = nullptr },
	{ .name = "Lava", .albedo = "textures/lava_still.png", .specular = nullptr, .emissive = "textures/lava_still.png" },
	{ .name = "Oak planks", .albedo = "textures/oak_planks.png", .specular = nullptr, .emissive = nullptr },
};

constexpr uint32_t material_count = std::size(materials);

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

const uint32_t cube_faces[][4] = {
	{ 0, 3, 2, 1 },
	{ 4, 5, 6, 7 },
	{ 0, 4, 7, 3 },
	{ 1, 2, 6, 5 },
	{ 0, 1, 5, 4 },
	{ 3, 7, 6, 2 },
};

constexpr uint32_t cone_segments = 32;
constexpr uint32_t cone_rays = 8;

Vertex cube_vertices[std::size(cube_faces) * 4];
uint32_t cube_indices[std::size(cube_faces) * 6];
Vertex cone_vertices[(cone_segments + cone_rays) * 2];

constexpr uint32_t object_count = 4;
constexpr uint32_t max_models = 64;
constexpr size_t max_lights = 16;
constexpr VkDeviceSize staging_size = 64 * 1024 * 1024;
constexpr VkExtent2D shadow_extent{2048, 2048};
constexpr VkImageSubresourceRange shadow_depth_range{
	.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
	.levelCount = 1,
	.layerCount = 1,
};

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
};

struct Texture {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VkImageView view = VK_NULL_HANDLE;
};

struct MaterialTextures {
	Texture albedo;
	Texture specular;
	Texture emissive;
};

struct Staging {
	Buffer buffer;
	VkDeviceSize offset = 0;
	VkCommandPool command_pool = VK_NULL_HANDLE;
	VkCommandBuffer command_buffer = VK_NULL_HANDLE;
};

struct Scene {
	Buffer cube_vertex_buffer;
	Buffer cube_index_buffer;
	Buffer cone_buffer;

	Buffer scene_uniform_buffer;
	Buffer point_light_buffer;
	Buffer spot_light_buffer;
	Buffer model_uniform_buffer;
	VkDeviceSize model_uniform_stride = 0;

	MaterialTextures material_textures[material_count];
	Texture black_texture;
	VkSampler sampler = VK_NULL_HANDLE;
	Texture sun_shadow_map;
	Texture spot_shadow_map;
	VkFormat shadow_format = VK_FORMAT_UNDEFINED;
	VkSampler shadow_sampler = VK_NULL_HANDLE;
	bool shadow_maps_initialized = false;
	glm::mat4 sun_view_projection{1.0f};
	glm::mat4 spot_view_projection{1.0f};

	VkDescriptorSetLayout scene_descriptor_layout = VK_NULL_HANDLE;
	VkDescriptorSetLayout material_descriptor_layout = VK_NULL_HANDLE;
	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
	VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;
	VkDescriptorSet material_descriptor_sets[material_count]{};
	VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
	VkPipelineLayout shadow_pipeline_layout = VK_NULL_HANDLE;

	VkPipeline lit_pipeline = VK_NULL_HANDLE;
	VkPipeline light_pipeline = VK_NULL_HANDLE;
	VkPipeline cone_pipeline = VK_NULL_HANDLE;
	VkPipeline shadow_pipeline = VK_NULL_HANDLE;

	uint32_t point_light_count = 0;
	uint32_t spot_light_count = 0;
};

struct Camera {
	glm::vec3 position{0, 3, 8};
	float yaw = 0;
	float pitch = -22;
	float fov = 60;
};

struct Object {
	const char* name;
	glm::vec3 position;
	glm::vec3 rotation;
	glm::vec3 scale;
	glm::vec3 specular;
	float alpha;
	glm::vec2 uv_scale;
	int material;
	bool animated_sampling;
	bool casts_shadow = true;
	bool show_sun_shadow_underneath = false;
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
	float sampling_time = 0;
	float sampling_speed = 1;

	glm::vec3 ambient_light{0.08, 0.08, 0.1};
	glm::vec3 directional_light_direction{-0.5, -1, -0.3};
	glm::vec3 directional_light_color{1, 0.95, 0.9};
	float directional_light_intensity = 0.85f;
	bool sun_shadows = true;
	bool spot_shadows = true;
	float sun_coverage = 16.0f;
	float depth_bias = 1.25f;
	float slope_bias = 1.75f;

	Object objects[object_count] = {
		{ .name = "Primary cube", .position = {}, .rotation = {}, .scale = glm::vec3(1), .specular = glm::vec3(0.6), .alpha = 32, .uv_scale = glm::vec2(1), .material = 2, .animated_sampling = true },
		{ .name = "Second cube", .position = {3.5, -2, 0}, .rotation = {}, .scale = glm::vec3(0.5), .specular = glm::vec3(1), .alpha = 128, .uv_scale = glm::vec2(1), .material = 1, .animated_sampling = false },
		{ .name = "Floor", .position = {0, -2.6, 0}, .rotation = {}, .scale = {10, 0.1, 10}, .specular = glm::vec3(0.2), .alpha = 8, .uv_scale = glm::vec2(5), .material = 3, .animated_sampling = false, .casts_shadow = false, .show_sun_shadow_underneath = true },
		{ .name = "Wall", .position = {0, 1.5, -6}, .rotation = {}, .scale = {10, 4, 0.1}, .specular = glm::vec3(0.5), .alpha = 16, .uv_scale = {5, 2}, .material = 0, .animated_sampling = false },
	};

	std::vector<PointLightSettings> point_lights = {
		{ .position = {-2, 2, 0}, .color = {0.3, 0.6, 1}, .intensity = 6 },
	};
	std::vector<SpotLightSettings> spot_lights = {
		{ .position = {-3, 4, 3}, .direction = {0.5, -1, -0.6}, .color = {1, 0.85, 0.6}, .intensity = 30 },
		{ .position = {0, -6, 0}, .direction = {0, 1, 0}, .color = {1, 1, 1}, .intensity = 24, .inner_angle = 40, .outer_angle = 55 },
	};
	bool show_lights = false;
};

Scene scene;
Staging staging;
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

bool create_gpu_buffer(Buffer& buffer, const VkDeviceSize size, const VkBufferUsageFlags usage) {
	const VkBufferCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	constexpr VmaAllocationCreateInfo allocation_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};
	if (vmaCreateBuffer(graphics::internal::context.allocator, &info, &allocation_info, &buffer.buffer, &buffer.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create GPU buffer\n";
		return false;
	}
	return true;
}

void destroy_buffer(Buffer& buffer) {
	if (buffer.buffer != VK_NULL_HANDLE) vmaDestroyBuffer(graphics::internal::context.allocator, buffer.buffer, buffer.allocation);
	buffer = {};
}

void destroy_texture(Texture& texture) {
	const auto& context = graphics::internal::context;
	if (texture.view != VK_NULL_HANDLE) vkDestroyImageView(context.device, texture.view, nullptr);
	if (texture.image != VK_NULL_HANDLE) vmaDestroyImage(context.allocator, texture.image, texture.allocation);
	texture = {};
}

void upload(const Buffer& buffer, const void* data, const VkDeviceSize size) {
	std::memcpy(buffer.mapped, data, size);
	vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, 0, size);
}

void write_buffer_descriptor(VkDescriptorSet set, const uint32_t binding, const VkDescriptorType type, const Buffer& buffer, const VkDeviceSize range = VK_WHOLE_SIZE) {
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = buffer.buffer,
		.offset = 0,
		.range = range,
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

void write_image_descriptor(VkDescriptorSet set, const uint32_t binding, const Texture& texture, VkSampler sampler,
	const VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
	const VkDescriptorImageInfo image_info = {
		.sampler = sampler,
		.imageView = texture.view,
		.imageLayout = layout,
	};
	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = binding,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &image_info,
	};
	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

glm::vec3 safe_direction(const glm::vec3& direction) {
	return glm::length(direction) > 0.0001f ? glm::normalize(direction) : glm::vec3(0, -1, 0);
}
//endregion

// region staging

void destroy_staging() {
	if (staging.command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(graphics::internal::context.device, staging.command_pool, nullptr);
	destroy_buffer(staging.buffer);
	staging = {};
}

bool begin_staging() {
	const auto& context = graphics::internal::context;
	if (!create_buffer(staging.buffer, staging_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;

	const VkCommandPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
		.queueFamilyIndex = context.graphics_queue_index,
	};
	if (vkCreateCommandPool(context.device, &pool_info, nullptr, &staging.command_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create staging command pool\n";
		return false;
	}

	const VkCommandBufferAllocateInfo allocation_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = staging.command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	if (vkAllocateCommandBuffers(context.device, &allocation_info, &staging.command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to allocate staging command buffer\n";
		return false;
	}

	constexpr VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	return vkBeginCommandBuffer(staging.command_buffer, &begin_info) == VK_SUCCESS;
}

bool stage(const void* data, const VkDeviceSize size, VkDeviceSize& offset) {
	offset = staging.offset + 15 & ~static_cast<VkDeviceSize>(15);
	if (offset + size > staging_size) {
		std::cerr << "Staging buffer is too small\n";
		return false;
	}
	std::memcpy(static_cast<std::byte*>(staging.buffer.mapped) + offset, data, size);
	staging.offset = offset + size;
	return true;
}

bool upload_to_gpu(Buffer& buffer, const void* data, const VkDeviceSize size, const VkBufferUsageFlags usage) {
	VkDeviceSize offset = 0;
	if (!create_gpu_buffer(buffer, size, usage) || !stage(data, size, offset)) return false;

	const VkBufferCopy region = {
		.srcOffset = offset,
		.dstOffset = 0,
		.size = size,
	};
	vkCmdCopyBuffer(staging.command_buffer, staging.buffer.buffer, buffer.buffer, 1, &region);
	return true;
}

bool create_image(Texture& texture, const VkExtent2D extent, const VkFormat format, const VkImageUsageFlags usage,
	const VkImageSubresourceRange subresource) {
	const auto& context = graphics::internal::context;

	const VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = { .width = extent.width, .height = extent.height, .depth = 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	constexpr VmaAllocationCreateInfo allocation_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};
	if (vmaCreateImage(context.allocator, &image_info, &allocation_info, &texture.image, &texture.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create image\n";
		return false;
	}

	const VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = texture.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = subresource,
	};
	if (vkCreateImageView(context.device, &view_info, nullptr, &texture.view) != VK_SUCCESS) {
		std::cerr << "Failed to create image view\n";
		return false;
	}
	return true;
}

bool create_texture(Texture& texture, const void* pixels, const uint32_t width, const uint32_t height, const VkFormat format) {
	constexpr VkImageSubresourceRange subresource = {
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.levelCount = 1,
		.layerCount = 1,
	};
	if (!create_image(texture, {width, height}, format,
		VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, subresource)) return false;

	VkDeviceSize offset = 0;
	if (!stage(pixels, static_cast<VkDeviceSize>(width) * height * 4, offset)) return false;

	const VkImageMemoryBarrier to_transfer = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = 0,
		.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = texture.image,
		.subresourceRange = subresource,
	};
	vkCmdPipelineBarrier(staging.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_transfer);

	const VkBufferImageCopy region = {
		.bufferOffset = offset,
		.bufferRowLength = 0,
		.bufferImageHeight = 0,
		.imageSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
		.imageOffset = {},
		.imageExtent = { .width = width, .height = height, .depth = 1 },
	};
	vkCmdCopyBufferToImage(staging.command_buffer, staging.buffer.buffer, texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	const VkImageMemoryBarrier to_shader = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = texture.image,
		.subresourceRange = subresource,
	};
	vkCmdPipelineBarrier(staging.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_shader);
	return true;
}

bool load_texture(Texture& texture, const char* path, const VkFormat format) {
	std::vector<unsigned char> pixels;
	unsigned width = 0, height = 0;
	if (const unsigned error = lodepng::decode(pixels, width, height, path)) {
		std::cerr << "Failed to load texture " << path << ": " << lodepng_error_text(error) << '\n';
		return false;
	}
	return create_texture(texture, pixels.data(), width, height, format);
}

bool end_staging() {
	const auto& context = graphics::internal::context;

	constexpr VkMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT,
	};
	vkCmdPipelineBarrier(staging.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
	if (vkEndCommandBuffer(staging.command_buffer) != VK_SUCCESS) return false;

	constexpr VkFenceCreateInfo fence_info = {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
	};
	VkFence fence = VK_NULL_HANDLE;
	if (vkCreateFence(context.device, &fence_info, nullptr, &fence) != VK_SUCCESS) return false;

	constexpr VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &staging.command_buffer,
	};
	const bool submitted = vkQueueSubmit(context.graphics_queue, 1, &submit, fence) == VK_SUCCESS;
	if (submitted) vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX);
	vkDestroyFence(context.device, fence, nullptr);
	destroy_staging();
	if (!submitted) std::cerr << "Failed to submit staging commands\n";
	return submitted;
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
	for (uint32_t face = 0; face < std::size(cube_faces); ++face) {
		const glm::vec3 a = cube_corners[cube_faces[face][0]];
		const glm::vec3 b = cube_corners[cube_faces[face][1]];
		const glm::vec3 c = cube_corners[cube_faces[face][2]];
		const glm::vec3 normal = glm::normalize(glm::cross(b - a, c - a));
		const glm::vec3 up = std::abs(normal.y) > 0.5f ? glm::vec3(0, 0, -normal.y) : glm::vec3(0, 1, 0);
		const glm::vec3 right = glm::cross(up, normal);

		for (uint32_t corner = 0; corner < 4; ++corner) {
			const glm::vec3 position = cube_corners[cube_faces[face][corner]];
			const glm::vec2 uv(glm::dot(position, right) * 0.5f + 0.5f, -glm::dot(position, up) * 0.5f + 0.5f);
			cube_vertices[face * 4 + corner] = { .position = position, .normal = normal, .uv = uv };
		}

		constexpr uint32_t quad[] = { 0, 1, 2, 0, 2, 3 };
		for (uint32_t i = 0; i < 6; ++i) cube_indices[face * 6 + i] = face * 4 + quad[i];
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
	destroy_staging();
	if (scene.shadow_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, scene.shadow_pipeline, nullptr);
	if (scene.cone_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, scene.cone_pipeline, nullptr);
	if (scene.light_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, scene.light_pipeline, nullptr);
	if (scene.lit_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, scene.lit_pipeline, nullptr);
	if (scene.shadow_pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(context.device, scene.shadow_pipeline_layout, nullptr);
	if (scene.pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(context.device, scene.pipeline_layout, nullptr);
	if (scene.descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(context.device, scene.descriptor_pool, nullptr);
	if (scene.material_descriptor_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context.device, scene.material_descriptor_layout, nullptr);
	if (scene.scene_descriptor_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context.device, scene.scene_descriptor_layout, nullptr);
	if (scene.sampler != VK_NULL_HANDLE) vkDestroySampler(context.device, scene.sampler, nullptr);
	if (scene.shadow_sampler != VK_NULL_HANDLE) vkDestroySampler(context.device, scene.shadow_sampler, nullptr);
	destroy_texture(scene.spot_shadow_map);
	destroy_texture(scene.sun_shadow_map);
	for (auto&[albedo, specular, emissive] : scene.material_textures) {
		destroy_texture(emissive);
		destroy_texture(specular);
		destroy_texture(albedo);
	}
	destroy_texture(scene.black_texture);
	destroy_buffer(scene.model_uniform_buffer);
	destroy_buffer(scene.spot_light_buffer);
	destroy_buffer(scene.point_light_buffer);
	destroy_buffer(scene.scene_uniform_buffer);
	destroy_buffer(scene.cone_buffer);
	destroy_buffer(scene.cube_index_buffer);
	destroy_buffer(scene.cube_vertex_buffer);
	scene = {};
}

bool create_textures() {
	constexpr uint8_t black_pixel[] = { 0, 0, 0, 255 };
	if (!create_texture(scene.black_texture, black_pixel, 1, 1, VK_FORMAT_R8G8B8A8_UNORM)) return false;

	for (uint32_t i = 0; i < material_count; ++i) {
		const auto&[name, albedo, specular, emissive] = materials[i];
		auto& textures = scene.material_textures[i];
		if (!load_texture(textures.albedo, albedo, VK_FORMAT_R8G8B8A8_SRGB)) return false;
		if (specular && !load_texture(textures.specular, specular, VK_FORMAT_R8G8B8A8_UNORM)) return false;
		if (emissive && !load_texture(textures.emissive, emissive, VK_FORMAT_R8G8B8A8_SRGB)) return false;
	}

	constexpr VkSamplerCreateInfo sampler_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_NEAREST,
		.minFilter = VK_FILTER_NEAREST,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.maxLod = 0,
	};
	if (vkCreateSampler(graphics::internal::context.device, &sampler_info, nullptr, &scene.sampler) != VK_SUCCESS) {
		std::cerr << "Failed to create sampler\n";
		return false;
	}
	return true;
}

bool create_shadow_maps() {
	const auto& context = graphics::internal::context;
	constexpr VkFormat formats[] = {VK_FORMAT_D16_UNORM, VK_FORMAT_D32_SFLOAT};
	constexpr auto required = VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT
		| VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
	for (const auto format : formats) {
		VkFormatProperties3 features{.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
		VkFormatProperties2 properties{.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &features};
		vkGetPhysicalDeviceFormatProperties2(context.physical_device, format, &properties);
		if ((features.optimalTilingFeatures & required) != required) continue;
		scene.shadow_format = format;
		break;
	}
	if (scene.shadow_format == VK_FORMAT_UNDEFINED) {
		std::cerr << "No depth format supports both shadow rendering and comparison sampling\n";
		return false;
	}

	constexpr auto usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	if (!create_image(scene.sun_shadow_map, shadow_extent, scene.shadow_format, usage, shadow_depth_range)
		|| !create_image(scene.spot_shadow_map, shadow_extent, scene.shadow_format, usage, shadow_depth_range)) return false;

	constexpr VkSamplerCreateInfo sampler_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_NEAREST,
		.minFilter = VK_FILTER_NEAREST,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.compareEnable = VK_TRUE,
		.compareOp = VK_COMPARE_OP_LESS,
		.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
	};
	if (vkCreateSampler(context.device, &sampler_info, nullptr, &scene.shadow_sampler) != VK_SUCCESS) {
		std::cerr << "Failed to create shadow comparison sampler\n";
		return false;
	}
	return true;
}

bool create_descriptors() {
	const auto& context = graphics::internal::context;
	constexpr VkShaderStageFlags stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = stages },
		{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = stages },
		{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = stages },
		{ .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .pImmutableSamplers = &scene.shadow_sampler },
		{ .binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .pImmutableSamplers = &scene.shadow_sampler },
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

	constexpr VkDescriptorSetLayoutBinding material_bindings[] = {
		{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1, .stageFlags = stages },
		{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
		{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
		{ .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
	};
	const VkDescriptorSetLayoutCreateInfo material_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = std::size(material_bindings),
		.pBindings = material_bindings,
	};
	if (vkCreateDescriptorSetLayout(context.device, &material_layout_info, nullptr, &scene.material_descriptor_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create material descriptor set layout\n";
		return false;
	}

	constexpr VkDescriptorPoolSize pool_sizes[] = {
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = material_count },
		{ .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = material_count * 3 + 2 },
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + material_count,
		.poolSizeCount = std::size(pool_sizes),
		.pPoolSizes = pool_sizes,
	};
	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &scene.descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	VkDescriptorSetLayout layouts[1 + material_count];
	std::ranges::fill(layouts, scene.material_descriptor_layout);
	layouts[0] = scene.scene_descriptor_layout;
	VkDescriptorSet descriptor_sets[1 + material_count]{};
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
	write_image_descriptor(scene.scene_descriptor_set, 3, scene.sun_shadow_map, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
	write_image_descriptor(scene.scene_descriptor_set, 4, scene.spot_shadow_map, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
	for (uint32_t i = 0; i < material_count; ++i) {
		const auto&[albedo, specular, emissive] = scene.material_textures[i];
		scene.material_descriptor_sets[i] = descriptor_sets[1 + i];
		write_buffer_descriptor(scene.material_descriptor_sets[i], 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, scene.model_uniform_buffer, sizeof(ModelUniforms));
		write_image_descriptor(scene.material_descriptor_sets[i], 1, albedo, scene.sampler);
		write_image_descriptor(scene.material_descriptor_sets[i], 2, specular.view != VK_NULL_HANDLE ? specular : scene.black_texture, scene.sampler);
		write_image_descriptor(scene.material_descriptor_sets[i], 3, emissive.view != VK_NULL_HANDLE ? emissive : scene.black_texture, scene.sampler);
	}

	const VkDescriptorSetLayout pipeline_set_layouts[] = {
		scene.scene_descriptor_layout,
		scene.material_descriptor_layout,
	};
	constexpr VkPushConstantRange push_constant_range = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(PushConstants),
	};
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = std::size(pipeline_set_layouts),
		.pSetLayouts = pipeline_set_layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constant_range,
	};
	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &scene.pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}
	constexpr VkPushConstantRange shadow_matrix_range = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.size = sizeof(glm::mat4),
	};
	const VkPipelineLayoutCreateInfo shadow_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &scene.material_descriptor_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &shadow_matrix_range,
	};
	if (vkCreatePipelineLayout(context.device, &shadow_layout_info, nullptr, &scene.shadow_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create shadow pipeline layout\n";
		return false;
	}
	return true;
}

bool create_pipeline(VkPipeline& pipeline, const char* vertex_path, const char* fragment_path, const VkPrimitiveTopology topology) {
	const auto& context = graphics::internal::context;
	const bool shadow_pass = fragment_path == nullptr;
	auto vertex_shader = load_shader(vertex_path);
	auto fragment_shader = shadow_pass ? VK_NULL_HANDLE : load_shader(fragment_path);
	if (vertex_shader == VK_NULL_HANDLE || (!shadow_pass && fragment_shader == VK_NULL_HANDLE)) {
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
		{ .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, uv) },
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = shadow_pass ? 1u : static_cast<uint32_t>(std::size(vertex_attributes)),
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

	const VkPipelineRasterizationStateCreateInfo rasterizer = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = shadow_pass ? VK_TRUE : VK_FALSE,
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
		VK_DYNAMIC_STATE_DEPTH_BIAS,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = shadow_pass ? 3u : 2u,
		.pDynamicStates = dynamic_states,
	};

	const VkPipelineRenderingCreateInfo shadow_rendering = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.depthAttachmentFormat = scene.shadow_format,
	};
	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = shadow_pass ? &shadow_rendering : nullptr,
		.stageCount = shadow_pass ? 1u : static_cast<uint32_t>(std::size(stages)),
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = shadow_pass ? nullptr : &color_blending,
		.pDynamicState = &dynamic_state,
		.layout = shadow_pass ? scene.shadow_pipeline_layout : scene.pipeline_layout,
		.renderPass = shadow_pass ? VK_NULL_HANDLE : context.render_pass,
	};
	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);
	if (fragment_shader != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, fragment_shader, nullptr);
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline: " << vertex_path << " (VkResult " << result << ")\n";
		return false;
	}
	return true;
}

bool create_resources() {
	generate_cube();
	generate_cone();

	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device, &properties);
	const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
	scene.model_uniform_stride = (sizeof(ModelUniforms) + alignment - 1) & ~(alignment - 1);

	if (!create_buffer(scene.scene_uniform_buffer, sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) return false;
	if (!create_buffer(scene.point_light_buffer, max_lights * sizeof(PointLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) || !create_buffer(scene.spot_light_buffer, max_lights * sizeof(SpotLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) return false;
	if (!create_buffer(scene.model_uniform_buffer, max_models * scene.model_uniform_stride, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) return false;

	if (!begin_staging()) return false;
	if (!upload_to_gpu(scene.cube_vertex_buffer, cube_vertices, sizeof(cube_vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) return false;
	if (!upload_to_gpu(scene.cube_index_buffer, cube_indices, sizeof(cube_indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT)) return false;
	if (!upload_to_gpu(scene.cone_buffer, cone_vertices, sizeof(cone_vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) return false;
	if (!create_textures() || !end_staging()) return false;

	return create_shadow_maps() && create_descriptors()
		&& create_pipeline(scene.shadow_pipeline, "shaders/shadow.vert.spv", nullptr, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
		&& create_pipeline(scene.lit_pipeline, "shaders/cube.vert.spv", "shaders/cube.frag.spv", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
		&& create_pipeline(scene.light_pipeline, "shaders/light.vert.spv", "shaders/marker.frag.spv", VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
		&& create_pipeline(scene.cone_pipeline, "shaders/cone.vert.spv", "shaders/light.frag.spv", VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
}

glm::mat4 light_view(const glm::vec3& position, const glm::vec3& direction) {
	const glm::vec3 up = std::abs(direction.y) > 0.99f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
	return glm::lookAt(position, position + direction, up);
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

	const auto sun_direction = safe_direction(controls.directional_light_direction);
	constexpr glm::vec3 sun_target(0.0f, -0.5f, -1.5f);
	const float half_size = controls.sun_coverage * 0.5f;
	auto sun_projection = glm::ortho(-half_size, half_size, -half_size, half_size, 0.1f, 50.0f);
	sun_projection[1][1] *= -1.0f;
	scene.sun_view_projection = sun_projection * light_view(sun_target - 25.0f * sun_direction, sun_direction);
	if (!spot_lights.empty()) {
		const SpotLight& spot = spot_lights.front();
		auto spot_projection = glm::perspective(2.0f * spot.outer_angle, 1.0f, 0.1f, 50.0f);
		spot_projection[1][1] *= -1.0f;
		scene.spot_view_projection = spot_projection * light_view(spot.position, spot.direction);
	}

	const SceneUniforms scene_uniforms{
		.view_projection = projection * view,
		.camera_position = controls.camera.position,
		.directional_light_direction = sun_direction,
		.directional_light_color = controls.directional_light_color * controls.directional_light_intensity,
		.ambient_light = controls.ambient_light,
		.point_light_count = scene.point_light_count,
		.spot_light_count = scene.spot_light_count,
		.sun_view_projection = scene.sun_view_projection,
		.spot_view_projection = scene.spot_view_projection,
		.sun_shadows_enabled = controls.sun_shadows,
		.spot_shadows_enabled = controls.spot_shadows && scene.spot_light_count > 0,
	};
	upload(scene.scene_uniform_buffer, &scene_uniforms, sizeof(scene_uniforms));

	const float phase = controls.phase;
	const glm::vec3 trajectory(controls.radius * std::cos(phase),controls.height * std::sin(phase),controls.depth * std::sin(phase));
	const glm::vec3 animated_rotation(30* std::sin(phase),glm::degrees(phase),std::cos(phase));

	auto* models = static_cast<std::byte*>(scene.model_uniform_buffer.mapped);
	for (uint32_t i = 0; i < object_count; ++i) {
		const auto object = controls.objects[i];
		const auto animated = i == 0;
		const ModelUniforms model_uniforms{
			.model = model_matrix(object.position, object.rotation, object.scale, animated ? trajectory : glm::vec3(), animated ? animated_rotation : glm::vec3()),
			.specular = object.specular,
			.alpha = object.alpha,
			.uv_scale = object.uv_scale,
		};
		std::memcpy(models + i * scene.model_uniform_stride, &model_uniforms, sizeof(model_uniforms));
	}
	vmaFlushAllocation(graphics::internal::context.allocator, scene.model_uniform_buffer.allocation, 0, object_count * scene.model_uniform_stride);
}

void render_shadow_map(VkCommandBuffer commands, const Texture& map, const glm::mat4& light_view_projection, const bool draw_geometry) {
	constexpr VkPipelineStageFlags depth_tests = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	constexpr VkImageLayout depth_attachment = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	constexpr VkImageLayout depth_read_only = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
	const VkImageMemoryBarrier to_attachment = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = scene.shadow_maps_initialized ? VK_ACCESS_SHADER_READ_BIT : VkAccessFlags{0},
		.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		.oldLayout = scene.shadow_maps_initialized ? depth_read_only : VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = depth_attachment,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = map.image,
		.subresourceRange = shadow_depth_range,
	};
	vkCmdPipelineBarrier(commands, scene.shadow_maps_initialized ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,depth_tests, 0, 0, nullptr, 0, nullptr, 1, &to_attachment);

	const VkRenderingAttachmentInfo depth = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = map.view,
		.imageLayout = depth_attachment,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.clearValue = {.depthStencil = {.depth = 1.0f}},
	};
	const VkRenderingInfo rendering = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = {.extent = shadow_extent},
		.layerCount = 1,
		.pDepthAttachment = &depth,
	};
	vkCmdBeginRendering(commands, &rendering);
	if (draw_geometry) {
		constexpr VkViewport viewport = {
			.width = static_cast<float>(shadow_extent.width),
			.height = static_cast<float>(shadow_extent.height),
			.maxDepth = 1.0f,
		};
		constexpr VkRect2D scissor = {.extent = shadow_extent};
		constexpr VkDeviceSize offset = 0;
		vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, scene.shadow_pipeline);
		vkCmdSetViewport(commands, 0, 1, &viewport);
		vkCmdSetScissor(commands, 0, 1, &scissor);
		vkCmdSetDepthBias(commands, controls.depth_bias, 0.0f, controls.slope_bias);
		vkCmdBindVertexBuffers(commands, 0, 1, &scene.cube_vertex_buffer.buffer, &offset);
		vkCmdBindIndexBuffer(commands, scene.cube_index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdPushConstants(commands, scene.shadow_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(light_view_projection), glm::value_ptr(light_view_projection));
		for (uint32_t i = 0; i < object_count; ++i) {
			if (!controls.objects[i].casts_shadow) continue;
			const auto dynamic_offset = static_cast<uint32_t>(i * scene.model_uniform_stride);
			vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, scene.shadow_pipeline_layout,
				0, 1, &scene.material_descriptor_sets[controls.objects[i].material], 1, &dynamic_offset);
			vkCmdDrawIndexed(commands, std::size(cube_indices), 1, 0, 0, 0);
		}
	}
	vkCmdEndRendering(commands);

	const VkImageMemoryBarrier to_sampled = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		.oldLayout = depth_attachment,
		.newLayout = depth_read_only,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = map.image,
		.subresourceRange = shadow_depth_range,
	};
	vkCmdPipelineBarrier(commands, depth_tests, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_sampled);
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
	controls.sampling_time = std::fmod(controls.sampling_time + delta_time * controls.sampling_speed, 1000.0f);

	ImGui::Begin("Lab 4");
	if (ImGui::CollapsingHeader("Shadows", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("Sun shadows", &controls.sun_shadows);
		ImGui::Checkbox("First spotlight shadows", &controls.spot_shadows);
		ImGui::SliderFloat("Sun span", &controls.sun_coverage, 2.0f, 40.0f, "%.1f");
		ImGui::Text("Orthographic area: %.1f world units squared", controls.sun_coverage * controls.sun_coverage);
		ImGui::SliderFloat("Depth bias", &controls.depth_bias, 0.0f, 12.0f);
		ImGui::SliderFloat("Slope bias", &controls.slope_bias, 0.0f, 8.0f);
	}

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
			ImGui::Combo("Texture", &object.material, [](void*, const int i) { return materials[i].name; }, nullptr, material_count);
			ImGui::DragFloat2("UV scale", glm::value_ptr(object.uv_scale), 0.05, 0.1, 20.0);
			ImGui::Checkbox("Animated sampling", &object.animated_sampling);
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
		ImGui::SliderFloat("Sampling speed", &controls.sampling_speed, 0.0, 4.0, "%.2fx");
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
	render_shadow_map(fd.command_buffer, scene.sun_shadow_map, scene.sun_view_projection, true);
	render_shadow_map(fd.command_buffer, scene.spot_shadow_map, scene.spot_view_projection, scene.spot_light_count > 0);
	scene.shadow_maps_initialized = true;

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
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &scene.cube_vertex_buffer.buffer, &offset);
	vkCmdBindIndexBuffer(fd.command_buffer, scene.cube_index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, scene.pipeline_layout, 0, 1, &scene.scene_descriptor_set, 0, nullptr);
	for (uint32_t i = 0; i < object_count; ++i) {
		const Object& object = controls.objects[i];
		const auto dynamic_offset = static_cast<uint32_t>(i * scene.model_uniform_stride);
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, scene.pipeline_layout, 1, 1, &scene.material_descriptor_sets[object.material], 1, &dynamic_offset);
		const PushConstants push_constants{
			.time = controls.sampling_time,
			.animated_sampling = object.animated_sampling,
			.show_sun_shadow_underneath = object.show_sun_shadow_underneath,
		};
		vkCmdPushConstants(fd.command_buffer, scene.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push_constants), &push_constants);
		vkCmdDrawIndexed(fd.command_buffer, std::size(cube_indices), 1, 0, 0, 0);
	}

	if (controls.show_lights) {
		if (const auto light_count = scene.point_light_count + scene.spot_light_count) {
			vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, scene.light_pipeline);
			vkCmdDrawIndexed(fd.command_buffer, std::size(cube_indices), light_count, 0, 0, 0);
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
