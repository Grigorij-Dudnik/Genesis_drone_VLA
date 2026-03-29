import genesis as gs

gs.init(backend=gs.gpu, logging_level="warning")

scene = gs.Scene(
    show_viewer    = True,
    viewer_options = gs.options.ViewerOptions(
        res           = (3200, 1800),
        camera_pos    = (-3.5, 0.0, 2.0),
        camera_lookat = (0.0, 0.0, 1.0),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True, # visualize the coordinate frame of `world` at its origin
        world_frame_size = 1.0, # length of the world frame in meter
        show_link_frame  = False, # do not visualize coordinate frames of entity links
        show_cameras     = False, # do not visualize mesh and frustum of the cameras added
        plane_reflection = True, # turn on plane reflection
        ambient_light    = (0.6, 0.6, 0.6), # ambient light setting

    ),
    renderer = gs.renderers.Rasterizer(), # using rasterizer for camera rendering
)

# Ground
scene.add_entity(
    gs.morphs.Box(size=(60, 60, 0.2), pos=(0, 0, -0.1), fixed=True),
    surface=gs.surfaces.Rough(color=(0.3, 0.5, 0.3)),
)

# Buildings
buildings = [
    (5,  5, 1, 2, 8,  (0.8, 0.2, 0.2)),
    (7,  3, 2, 1.5, 4, (0.2, 0.4, 0.8)),
    (4, -4, 1.5, 1, 6,  (0.9, 0.7, 0.1)),
    (10, 0, 1, 2, 5, (0.3, 0.8, 0.4)),
    (8,  -3, 1, 2, 3, (0.8, 0.4, 0.8)),
]
for x, y, w, d, h, col in buildings:
    scene.add_entity(
        gs.morphs.Box(size=(w, d, h), pos=(x, y, h/2), fixed=True),
        surface=gs.surfaces.Rough(color=col),
    )

# helipad
scene.add_entity(
    gs.morphs.Cylinder(radius=0.3, height=0.4, pos=(12, 0, 0.2), fixed=True),
    surface=gs.surfaces.Rough(color=(0.6, 0.1, 0.1)),
)
scene.add_entity(
    gs.morphs.Cylinder(radius=0.3, height=0.003, pos=(12, 0, 0.402), fixed=True),
    surface=gs.surfaces.Rough(color=(0.9, 0.85, 0.1)),
)
_H = 0.403 
for morph in [
    gs.morphs.Box(size=(0.07, 0.34, 0.004), pos=(11.90, 0, _H), fixed=True),  # left bar
    gs.morphs.Box(size=(0.07, 0.34, 0.004), pos=(12.10, 0, _H), fixed=True),  # right bar
    gs.morphs.Box(size=(0.27, 0.07, 0.004), pos=(12.00, 0, _H), fixed=True),  # crossbar
]:
    scene.add_entity(morph, surface=gs.surfaces.Rough(color=(1.0, 1.0, 1.0)))


# scene.build()
# for _ in range(250):
#     scene.step()
