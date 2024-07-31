use std::{cmp::min, fs::File, io::Write, os::fd::AsFd};

use settings::{NAME, SIZE};
use wayland_client::{
    delegate_noop,
    protocol::{
        wl_buffer::WlBuffer,
        wl_compositor::WlCompositor,
        wl_keyboard,
        wl_registry::{Event, WlRegistry},
        wl_seat::{self, Capability, WlSeat},
        wl_shm::{Format, WlShm},
        wl_shm_pool::WlShmPool,
        wl_surface::WlSurface,
    },
    Connection, Dispatch, EventQueue, QueueHandle, WEnum,
};
use wayland_protocols::xdg::shell::client::{
    xdg_surface::{self, XdgSurface},
    xdg_toplevel::{self, XdgToplevel},
    xdg_wm_base::{self, XdgWmBase},
};

mod settings;

struct State {
    running: bool,
    base_surface: Option<WlSurface>,
    buffer: Option<WlBuffer>,
    wm_base: Option<XdgWmBase>,
    xdg_surface: Option<(XdgSurface, XdgToplevel)>,
    configured: bool,
}

impl Dispatch<WlRegistry, ()> for State {
    fn event(
        state: &mut Self,
        proxy: &WlRegistry,
        event: <WlRegistry as wayland_client::Proxy>::Event,
        _data: &(),
        _connection: &Connection,
        queue_handle: &QueueHandle<Self>,
    ) {
        match event {
            Event::Global {
                name,
                interface,
                version,
            } => {
                match interface.as_str() {
                    "wl_compositor" => {
                        let wl_compositor =
                            proxy.bind::<WlCompositor, _, _>(name, version, queue_handle, ());

                        let surface = wl_compositor.create_surface(queue_handle, ());
                        state.base_surface = Some(surface);

                        if state.wm_base.is_some() && state.xdg_surface.is_none() {
                            state.init_xdg_surface(queue_handle);
                        }
                    }
                    "wl_shm" => {
                        let wl_shm = proxy.bind::<WlShm, _, _>(name, version, queue_handle, ());

                        let mut file = tempfile::tempfile().unwrap();
                        draw(&mut file, SIZE);

                        let pool = wl_shm.create_pool(
                            file.as_fd(),
                            (SIZE.0 * SIZE.1 * 4) as i32,
                            queue_handle,
                            (),
                        );
                        let buffer = pool.create_buffer(
                            0,
                            SIZE.0 as i32,
                            SIZE.1 as i32,
                            (SIZE.0 * 4) as i32,
                            Format::Argb8888,
                            queue_handle,
                            (),
                        );
                        state.buffer = Some(buffer.clone());

                        if state.configured {
                            let surface = state.base_surface.as_ref().unwrap();
                            surface.attach(Some(&buffer), 0, 0);
                            surface.commit();
                        }
                    }
                    "wl_seat" => {
                        proxy.bind::<WlSeat, _, _>(name, version, queue_handle, ());
                    }
                    "xdg_wm_base" => {
                        let xdg_wm_base =
                            proxy.bind::<XdgWmBase, _, _>(name, version, queue_handle, ());
                        state.wm_base = Some(xdg_wm_base);

                        if state.base_surface.is_some() && state.xdg_surface.is_none() {
                            state.init_xdg_surface(queue_handle);
                        }
                    }
                    _ => (),
                };
            }
            Event::GlobalRemove { name: _ } => (),
            _ => (),
        }
    }
}

delegate_noop!(State: ignore WlCompositor);
delegate_noop!(State: ignore WlSurface);
delegate_noop!(State: ignore WlShm);
delegate_noop!(State: ignore WlShmPool);
delegate_noop!(State: ignore WlBuffer);

impl Dispatch<XdgSurface, ()> for State {
    fn event(
        state: &mut Self,
        proxy: &XdgSurface,
        event: <XdgSurface as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial, .. } = event {
            proxy.ack_configure(serial);
            state.configured = true;
            let surface = state.base_surface.as_ref().unwrap();
            if let Some(ref buffer) = state.buffer {
                surface.attach(Some(buffer), 0, 0);
                surface.commit();
            }
        }
    }
}

impl Dispatch<XdgToplevel, ()> for State {
    fn event(
        state: &mut Self,
        _: &XdgToplevel,
        event: <XdgToplevel as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_toplevel::Event::Close {} = event {
            state.running = false;
        }
    }
}

impl Dispatch<XdgWmBase, ()> for State {
    fn event(
        _: &mut Self,
        proxy: &XdgWmBase,
        event: <XdgWmBase as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            proxy.pong(serial);
        }
    }
}

impl Dispatch<WlSeat, ()> for State {
    fn event(
        _: &mut Self,
        proxy: &WlSeat,
        event: <WlSeat as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        queue_handle: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities {
            capabilities: WEnum::Value(capabilities),
        } = event
        {
            if capabilities.contains(Capability::Keyboard) {
                proxy.get_keyboard(queue_handle, ());
            }
        }
    }
}

impl Dispatch<wl_keyboard::WlKeyboard, ()> for State {
    fn event(
        state: &mut Self,
        _: &wl_keyboard::WlKeyboard,
        event: wl_keyboard::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_keyboard::Event::Key { key, .. } = event {
            match key {
                1 => state.running = false,
                _ => (),
            }
        }
    }
}

impl State {
    fn init_xdg_surface(&mut self, queue_handle: &QueueHandle<State>) {
        let wm_base = self.wm_base.as_ref().unwrap();
        let base_surface = self.base_surface.as_ref().unwrap();

        let xdg_surface = wm_base.get_xdg_surface(base_surface, queue_handle, ());
        let toplevel = xdg_surface.get_toplevel(queue_handle, ());
        toplevel.set_title(NAME.into());

        base_surface.commit();

        self.xdg_surface = Some((xdg_surface, toplevel));
    }
}

fn draw(tmp: &mut File, (buf_x, buf_y): (u32, u32)) {
    let mut buf = std::io::BufWriter::new(tmp);
    for y in 0..buf_y {
        for x in 0..buf_x {
            let a = 0xFF;
            let r = min(((buf_x - x) * 0xFF) / buf_x, ((buf_y - y) * 0xFF) / buf_y);
            let g = min((x * 0xFF) / buf_x, ((buf_y - y) * 0xFF) / buf_y);
            let b = min(((buf_x - x) * 0xFF) / buf_x, (y * 0xFF) / buf_y);
            buf.write_all(&[b as u8, g as u8, r as u8, a as u8])
                .unwrap();
        }
    }
    buf.flush().unwrap();
}

pub struct WaylandClient {
    event_queue: EventQueue<State>,
    state: State,
}

impl WaylandClient {
    pub fn new() -> Self {
        let connection = Connection::connect_to_env().expect("Couldn't connect to wayland server.");

        let event_queue = connection.new_event_queue();

        connection.display().get_registry(&event_queue.handle(), ());

        let state = State {
            running: true,
            base_surface: None,
            buffer: None,
            wm_base: None,
            xdg_surface: None,
            configured: false,
        };

        return WaylandClient { event_queue, state };
    }

    pub fn run(&mut self) {
        println!("Start:");
        while self.state.running {
            let _ = self.event_queue.blocking_dispatch(&mut self.state);
        }
        println!("End. \n");
    }
}
