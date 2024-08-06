use std::{
    fs::File,
    io::{BufWriter, Write},
    sync::Arc,
};

use bytemuck::AnyBitPattern;
use vulkan::VulkanProcessor;
use vulkano::{
    buffer::{BufferUsage, Subbuffer},
    command_buffer::{
        CommandBufferUsage, CopyImageToBufferInfo, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
        SubpassBeginInfo, SubpassContents, SubpassEndInfo,
    },
    format::Format,
    image::{view::ImageView, ImageType, ImageUsage},
    memory::allocator::MemoryTypeFilter,
    pipeline::graphics::{
        vertex_input::{Vertex, VertexDefinition},
        viewport::Viewport,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, Subpass},
};
use wayland::{settings::SIZE, WaylandClient};

mod vulkan;
mod wayland;

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0);
            }
        ",
    }
}

struct GraphicsProcessor<'a> {
    processor: &'a VulkanProcessor,
    size: (u32, u32),
    data_buffer: Subbuffer<[u8]>,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
}

#[derive(Vertex, AnyBitPattern, Clone, Copy)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

impl<'a> GraphicsProcessor<'a> {
    fn new(processor: &'a VulkanProcessor, size: (u32, u32)) -> Self {
        const FORMAT: Format = Format::R8G8B8A8_UNORM;

        let vertex_buffer = processor.create_iter_buffer(
            vec![
                MyVertex {
                    position: [-0.1, 0.1],
                },
                MyVertex {
                    position: [0.1, 0.1],
                },
                MyVertex {
                    position: [0.0, -0.141421356],
                },
            ],
            BufferUsage::VERTEX_BUFFER,
            MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        );

        let render_pass = processor.create_render_pass(FORMAT);

        let image = processor.create_image(
            ImageType::Dim2d,
            FORMAT,
            [size.0, size.1, 1],
            ImageUsage::TRANSFER_SRC | ImageUsage::COLOR_ATTACHMENT,
            MemoryTypeFilter::PREFER_DEVICE,
        );

        let view = ImageView::new_default(image.clone()).expect("Failed to create image view.");

        let framebuffer = Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![view],
                ..Default::default()
            },
        )
        .unwrap();

        let data_buffer = processor.create_iter_buffer(
            (0..size.0 * size.1 * 4).map(|_| 0u8).collect(),
            BufferUsage::TRANSFER_DST,
            MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
        );

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [size.0 as f32, size.1 as f32],
            depth_range: 0.0..=1.0,
        };

        let pipeline = {
            let stages_layout = processor.create_pipeline_stages_layout(vec![vs::load, fs::load]);

            let vertex_input_state = MyVertex::per_vertex()
                .definition(&stages_layout.0[0].entry_point.info().input_interface)
                .unwrap();

            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            processor.create_graphics_pipeline(stages_layout, vertex_input_state, viewport, subpass)
        };

        let command_buffer = processor.create_command_buffer(
            |builder| {
                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.2, 0.2, 0.2, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .unwrap()
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .unwrap()
                    .draw(3, 1, 0, 0)
                    .unwrap()
                    .end_render_pass(SubpassEndInfo::default())
                    .unwrap()
                    .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                        image,
                        data_buffer.clone(),
                    ))
                    .unwrap();
            },
            CommandBufferUsage::MultipleSubmit,
        );

        return GraphicsProcessor {
            processor,
            size,
            data_buffer,
            command_buffer,
        };
    }

    fn execute(&self, buffer_file: &File) {
        self.processor
            .execute_then_wait(self.command_buffer.clone());

        let result = self.data_buffer.read().unwrap();
        let mut writer = BufWriter::new(buffer_file);
        let mut index = 0;
        for _x in 0..self.size.0 {
            for _y in 0..self.size.1 {
                let r: u8 = result[index];
                let g: u8 = result[index + 1];
                let b: u8 = result[index + 2];
                let a: u8 = result[index + 3];
                index += 4;

                writer.write_all(&[b, g, r, a]).unwrap();
            }
        }
        writer.flush().unwrap();
    }
}

fn main() {
    let processor = VulkanProcessor::new();
    let graphics_processor = GraphicsProcessor::new(&processor, SIZE);

    let mut wayland_client =
        WaylandClient::new(|buffer_file| graphics_processor.execute(buffer_file));

    wayland_client.run();
}
