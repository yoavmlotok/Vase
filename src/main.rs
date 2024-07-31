use wayland_client::WaylandClient;

mod processor;
mod wayland_client;

fn main() {
    let mut wayland_client = WaylandClient::new();

    wayland_client.run();
}
