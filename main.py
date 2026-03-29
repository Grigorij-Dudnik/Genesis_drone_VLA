import os
import time
import math
import numpy as np
import genesis as gs
from buildings_scene import scene
from drone_setup import drone, drone_lidar, camera_front, camera_bottom, front_camera_mount
from init_dataset import dataset

#scene.viewer.follow_entity(drone)

timestamp = time.strftime("%Y%m%d-%H%M%S")
scene.start_recording(
    data_func=lambda: camera_front.read().rgb,
    rec_options=gs.recorders.VideoFile(
        filename=os.path.expanduser(f'~/Gregs_tech/Drone_sim/drone_flight_{timestamp}.mp4'),
        hz=50,
    )
)

scene.build()

scene.sim.rigid_solver.add_weld_constraint(
    drone.links[0].idx,
    front_camera_mount.links[0].idx
)


def autoflight_policy(sensor_data):   
    helipad_distance_x, helipad_distance_y, helipad_distance_z, yaw_diff = sensor_data
    move_x = (helipad_distance_x) * 0.3
    move_y = (helipad_distance_y) * 0.3
    move_x = np.clip(move_x, -1, 1)
    move_y = np.clip(move_y, -1, 1)
    move_z = 0
    move_yaw = np.clip(yaw_diff * 5, -1, 1)

    ray_distances = drone_lidar.read().distances

    # Indices 0, 1, 2 are the front corridor (-5°, 0°, 5°)
    front_obstacle_distance = ray_distances[0:3, 0].min().item()
    left_obstacle_distance = ray_distances[3, 0].item()
    right_obstacle_distance = ray_distances[4, 0].item()

    if move_x < 0.1 and move_y < 0.1:
        move_z = (helipad_distance_z) * 0.3
        move_z = np.clip(move_z, -1, 1)
    elif front_obstacle_distance < 3.0:
        move_y = 0.5 * -math.copysign(1, move_y)
    elif right_obstacle_distance < 0.9:
        move_y = -2 * right_obstacle_distance * (1 - 0.9 / right_obstacle_distance)
        move_y = np.clip(move_y, -1, 1)
    elif left_obstacle_distance < 0.9:
        move_y = 2 * left_obstacle_distance * (1 - 0.9 / left_obstacle_distance)
        move_y = np.clip(move_y, -1, 1)

    return np.array([move_x, move_y, move_z, move_yaw], dtype=np.float32)


def get_sensor_data(helipad_pos=(12, 0, 0.4)):
    pos = drone.get_pos().cpu()
    helipad_bearing = math.atan2(helipad_pos[1] - pos[1].item(), helipad_pos[0] - pos[0].item())
    w, x, y, z = drone.get_quat().cpu()
    drone_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw_diff = helipad_bearing - drone_yaw

    helipad_distance_x = helipad_pos[0] - pos[0].item()
    helipad_distance_y = helipad_pos[1] - pos[1].item()
    helipad_distance_z = helipad_pos[2] - pos[2].item()

    return np.array([helipad_distance_x, helipad_distance_y, helipad_distance_z, yaw_diff], dtype=np.float32)


while True:
    try:
        front_frame_np = (camera_front.read().rgb.clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
        bottom_frame_np = (camera_bottom.read().rgb.clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
        obs_state = get_sensor_data()

        action = autoflight_policy(obs_state)
        print(f"Action: move_x={action[0]:.2f}, move_y={action[1]:.2f}, move_z={action[2]:.2f}, yaw={action[3]:.2f}")
        target_velocity = np.array([action[0] * 5, action[1] * 5, action[2] * 5, 0, 0, action[3]])
        drone.set_propellels_rpm([10000, 10000, 10000, 10000])
        drone.set_dofs_velocity(velocity=target_velocity)

        dataset.add_frame({
            "observation.images.camera_front":  front_frame_np,
            "observation.images.camera_bottom": bottom_frame_np,
            "observation.state":                obs_state,
            "action":                           action,
        })

        scene.step()
        time.sleep(0.03)
    except KeyboardInterrupt:
        dataset.save_episode()
        print("Autoflight stopped by user.")
        break

scene.stop_recording()
