"""Isaac-inspired visual defaults, independent of simulation and controls."""

import viser


def configure_scene(server: viser.ViserServer) -> None:
    # Neutral studio lighting and a metric ground grid, inspired by
    # Isaac Sim's editor viewport. This affects visualization only.
    server.gui.configure_theme(
        dark_mode=True,
        control_width="medium",
        brand_color=(118, 185, 0),
        show_logo=False,
        show_share_button=False,
    )
    server.gui.main_panel.dock_right()
    server.scene.configure_environment_map(None)
    server.scene.configure_default_lights(enabled=False)
    server.scene.add_light_ambient("/lighting/ambient", intensity=1.0)
    server.scene.add_light_directional(
        "/lighting/key",
        intensity=2.4,
        color=(255, 248, 237),
        wxyz=(0.8805, 0.2798, 0.3647, 0.1159),
        cast_shadow=True,
    )
    server.scene.add_light_directional(
        "/lighting/fill",
        intensity=1.2,
        color=(205, 221, 255),
        wxyz=(0.8536, -0.3536, -0.3536, 0.1464),
    )
