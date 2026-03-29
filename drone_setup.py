import genesis as gs
from buildings_scene import scene

drone = scene.add_entity(
    morph=gs.morphs.Drone(
        file="urdf/drones/cf2x.urdf",
        pos=(0.0, 0, 1),
        euler=(0, 0, 0)
    ),
)

drone_lidar = scene.add_sensor(
    gs.sensors.Lidar(
        pattern=gs.sensors.SphericalPattern(angles=([-5.0, 0.0, 5.0, 45.0, -45.0], [0.0])),
        entity_idx=drone.idx,
        pos_offset=(0.0, 0.0, 0.05),
        min_range=0.3,
        max_range=5.0,
        draw_debug=True,
    )
)

camera_bottom = scene.add_sensor(
    gs.sensors.RasterizerCameraOptions(
        res=(640, 480),
        pos=(0.1, 0.0, 0.05),
        lookat=(0.0, 0.5, 0.0),
        entity_idx=drone.idx,
    )
)

front_camera_mount = scene.add_entity(
    morph=gs.morphs.Box(
        size=(0.01, 0.01, 0.01),
        pos=(0.0, 0, 1),
        euler=(90, 0, -90),
    ),
)

camera_front = scene.add_sensor(
    gs.sensors.RasterizerCameraOptions(
        res=(640, 480),
        pos=(0.1, 0.0, 0.05),
        entity_idx=front_camera_mount.idx,
    )
)
