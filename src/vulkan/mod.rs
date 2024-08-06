use std::{sync::Arc, time::Instant};

use bytemuck::AnyBitPattern;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
    },
    pipeline::{
        compute::ComputePipelineCreateInfo,
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::VertexInputState,
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{RenderPass, Subpass},
    shader::ShaderModule,
    sync::{self, GpuFuture},
    Validated, VulkanError, VulkanLibrary,
};
use wayland_client::backend::smallvec::SmallVec;

pub struct VulkanProcessor {
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: StandardCommandBufferAllocator,
}

impl VulkanProcessor {
    pub fn new() -> Self {
        println!("Creating new vulkan processor.");
        let creation_start = Instant::now();

        let library = VulkanLibrary::new().expect("No local Vulkan library/DLL.");
        let instance = Instance::new(library, InstanceCreateInfo::default())
            .expect("Failed to create instance.");

        let physical_device = instance
            .enumerate_physical_devices()
            .expect("Could not enumerate devices.")
            .min_by_key(|device| match device.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("No devices available.");

        println!(
            "Chose physical device: {:?}.",
            physical_device.properties().device_name
        );

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_queue_family_index, queue_family_properties)| {
                queue_family_properties
                    .queue_flags
                    .contains(QueueFlags::GRAPHICS)
            })
            .expect("Couldn't find a graphical queue family.")
            as u32;

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("Failed to create device.");

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        );

        let graphics_queue = queues.next().unwrap();

        println!(
            "Vulkan processor creation completed in {} milliseconds. \n",
            creation_start.elapsed().as_millis()
        );

        return VulkanProcessor {
            device,
            graphics_queue,
            memory_allocator,
            command_buffer_allocator,
        };
    }

    pub fn create_data_buffer<T: AnyBitPattern + BufferContents>(
        &self,
        data: T,
        buffer_usage: BufferUsage,
        memory_type_filters: MemoryTypeFilter,
    ) -> Subbuffer<T> {
        let buffer = Buffer::from_data(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: buffer_usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: memory_type_filters,
                ..Default::default()
            },
            data,
        )
        .expect("Failed to create buffer.");

        return buffer;
    }

    pub fn create_iter_buffer<T: AnyBitPattern + BufferContents>(
        &self,
        iter: Vec<T>,
        buffer_usage: BufferUsage,
        memory_type_filters: MemoryTypeFilter,
    ) -> Subbuffer<[T]> {
        let buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: buffer_usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: memory_type_filters,
                ..Default::default()
            },
            iter,
        )
        .expect("Failed to create buffer.");
        return buffer;
    }

    pub fn create_image(
        &self,
        image_type: ImageType,
        format: Format,
        extent: [u32; 3],
        usage: ImageUsage,
        memory_type_filters: MemoryTypeFilter,
    ) -> Arc<Image> {
        Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: image_type,
                format: format,
                extent: extent,
                usage: usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: memory_type_filters,
                ..Default::default()
            },
        )
        .unwrap()
    }

    pub fn create_render_pass(&self, format: Format) -> Arc<RenderPass> {
        vulkano::single_pass_renderpass!(
            self.device.clone(),
            attachments: {
                color: {
                    format: format,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap()
    }

    pub fn create_pipeline_stages_layout<T>(
        &self,
        load_functions: Vec<T>,
    ) -> (Vec<PipelineShaderStageCreateInfo>, Arc<PipelineLayout>)
    where
        T: Fn(Arc<Device>) -> Result<Arc<ShaderModule>, Validated<VulkanError>>,
    {
        let mut stages: Vec<PipelineShaderStageCreateInfo> = vec![];

        for load_function in load_functions {
            stages.push(PipelineShaderStageCreateInfo::new(
                load_function(self.device.clone())
                    .expect("Failed to create shader module.")
                    .entry_point("main")
                    .unwrap(),
            ));
        }

        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(
                stages
                    .iter()
                    .collect::<Vec<&PipelineShaderStageCreateInfo>>(),
            )
            .into_pipeline_layout_create_info(self.device.clone())
            .unwrap(),
        )
        .expect("Failed to create pipeline layout.");

        return (stages, layout);
    }

    pub fn create_compute_pipeline(
        &self,
        stage: PipelineShaderStageCreateInfo,
        layout: Arc<PipelineLayout>,
    ) -> Arc<ComputePipeline> {
        return ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("Failed to create compute pipeline.");
    }

    pub fn create_graphics_pipeline(
        &self,
        (stages, layout): (Vec<PipelineShaderStageCreateInfo>, Arc<PipelineLayout>),
        vertex_input_state: VertexInputState,
        viewport: Viewport,
        subpass: Subpass,
    ) -> Arc<GraphicsPipeline> {
        GraphicsPipeline::new(
            self.device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: SmallVec::from_vec(stages),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [viewport].into_iter().collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    }

    pub fn create_compute_descriptor_set(
        &self,
        compute_pipeline: Arc<ComputePipeline>,
        write_descriptor_sets: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> Arc<PersistentDescriptorSet> {
        return PersistentDescriptorSet::new(
            &StandardDescriptorSetAllocator::new(self.device.clone(), Default::default()),
            compute_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            write_descriptor_sets,
            [],
        )
        .expect("Failed to create descriptor set.");
    }

    pub fn create_graphics_descriptor_set(
        &self,
        graphics_pipeline: Arc<GraphicsPipeline>,
        write_descriptor_sets: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> Arc<PersistentDescriptorSet> {
        return PersistentDescriptorSet::new(
            &StandardDescriptorSetAllocator::new(self.device.clone(), Default::default()),
            graphics_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            write_descriptor_sets,
            [],
        )
        .expect("Failed to create descriptor set.");
    }

    pub fn create_command_buffer<T>(
        &self,
        builder_fn: T,
        usage: CommandBufferUsage,
    ) -> Arc<PrimaryAutoCommandBuffer>
    where
        T: FnOnce(&mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>),
    {
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.graphics_queue.queue_family_index(),
            usage,
        )
        .expect("Failed to create command buffer builder.");

        builder_fn(&mut builder);

        return builder.build().expect("Failed to create command buffer.");
    }

    pub fn execute_then_wait(&self, command_buffer: Arc<PrimaryAutoCommandBuffer>) {
        sync::now(self.device.clone())
            .then_execute(self.graphics_queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    #[cfg(debug_assertions)]
    pub fn _print_physical_devices(&self) {
        println!(
            "Found {} physical devices:",
            self.device
                .instance()
                .enumerate_physical_devices()
                .unwrap()
                .len()
        );

        for physical_device in self.device.instance().enumerate_physical_devices().unwrap() {
            println!(
                "  Found a physical device with name: {:?},",
                physical_device.properties().device_name
            )
        }

        println!()
    }

    #[cfg(debug_assertions)]
    pub fn _print_queue_families(&self) {
        println!(
            "Found {} queue families:",
            self.device
                .physical_device()
                .queue_family_properties()
                .len()
        );

        for family in self.device.physical_device().queue_family_properties() {
            println!(
                "  Found a queue family with flags '{:?}' containing {:?} queue(s),",
                family.queue_flags, family.queue_count
            );
        }

        println!()
    }
}
