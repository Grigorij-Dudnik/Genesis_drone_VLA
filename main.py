import os
import time
import math
import numpy as np
import genesis as gs
import random
from buildings_scene import scene
from drone_setup import drone, drone_lidar, camera_front, camera_bottom, front_camera_mount
from autoflight_policy import autoflight_policy
from init_dataset import dataset
from real_policy import PolicyInference



def body_to_world_vel(move_x, move_y, move_z):
    w, x, y, z = drone.get_quat().cpu().numpy()
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x**2 + z**2),  2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),         1 - 2*(x**2 + y**2)],
    ])
    return R @ np.array([move_x, move_y, move_z])

INFERENCE = True

#scene.viewer.follow_entity(drone)

# timestamp = time.strftime("%Y%m%d-%H%M%S")
# scene.start_recording(
#     data_func=lambda: camera_front.read().rgb,
#     rec_options=gs.recorders.VideoFile(
#         filename=os.path.expanduser(f'~/Gregs_tech/Drone_sim/drone_flight_{timestamp}.mp4'),
#         hz=50,
#     )
# )
scene.build()


def get_sensor_data(helipad_pos=(12, 0, 0.4)):
    pos = drone.get_pos().cpu()
    helipad_bearing = math.atan2(helipad_pos[1] - pos[1].item(), helipad_pos[0] - pos[0].item())
    w, x, y, z = drone.get_quat().cpu()
    drone_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw_diff = helipad_bearing - drone_yaw

    helipad_distance_x = helipad_pos[0] - pos[0].item()
    helipad_distance_y = helipad_pos[1] - pos[1].item()
    helipad_distance_z = helipad_pos[2] - pos[2].item()

    ray_distances = drone_lidar.read().distances
    front_obstacle_distance = ray_distances[0:3, 0].min().item()
    left_obstacle_distance = ray_distances[3, 0].item()
    right_obstacle_distance = ray_distances[4, 0].item()

    return np.array([helipad_distance_x, helipad_distance_y, helipad_distance_z, yaw_diff, front_obstacle_distance], dtype=np.float32), right_obstacle_distance, left_obstacle_distance


def check_contact(helipad_dx, helipad_dy, helipad_dz):
    if abs(helipad_dx) < 0.3 and abs(helipad_dy) < 0.3 and abs(helipad_dz) < 0.02:
        return True
    return False

scene.sim.rigid_solver.add_weld_constraint(
    drone.links[0].idx,
    front_camera_mount.links[0].idx
)

nr_episodes = 2
for _ in range(nr_episodes):
    if INFERENCE:
        break

    while True:
        front_frame_np = camera_front.read().rgb.cpu().numpy()
        bottom_frame_np = camera_bottom.read().rgb.cpu().numpy()
        obs_state, r_obstacle_d, l_obstacle_d = get_sensor_data()
        helipad_dx, helipad_dy, helipad_dz, yaw_diff, f_obstacle_d = obs_state

        action = autoflight_policy(obs_state, r_obstacle_d, l_obstacle_d)
        move_x, move_z, yaw = action
        wv = body_to_world_vel(move_x * 5, 0, move_z)
        target_velocity = np.array([wv[0], wv[1], wv[2], 0, 0, yaw])
        drone.set_propellels_rpm([10000, 10000, 10000, 10000])  # just decoration
        drone.set_dofs_velocity(velocity=target_velocity)

        dataset.add_frame({
            "observation.images.camera_front":  front_frame_np,
            "observation.images.camera_bottom": bottom_frame_np,
            "observation.state":                np.array([helipad_dz], dtype=np.float32),  
            "action":                           action,
            "task":                             "fly to helipad and land on it",
        })
        
        scene.step()

        if check_contact(helipad_dx, helipad_dy, helipad_dz):
            print("Helipad reached!")
            dataset.save_episode()
            scene.reset()
            drone_y_pos = random.uniform(-4, 4)
            drone_x_pos = random.uniform(5, 7)
            drone.set_pos((drone_x_pos, drone_y_pos, 1))
            front_camera_mount.set_pos((drone_x_pos, drone_y_pos, 1))
            drone.set_quat((1, 0, 0, 0))
            front_camera_mount.set_quat((0.5, 0.5, -0.5, -0.5))
            #front_camera_mount.set_quat((0.612, 0.354, -0.354, -0.612))
            scene.sim.rigid_solver.add_weld_constraint(
                drone.links[0].idx,
                front_camera_mount.links[0].idx
            )
            break
        #time.sleep(0.03)

if not INFERENCE:
    dataset.finalize()
    dataset.push_to_hub()

if INFERENCE:
    policy = PolicyInference(policy_name="Grigorij/xvla_drone_flight_no_buildings6", dataset_name="Grigorij/drone_flight_no_buildings2", task="fly to helipad and land on it")
    while True:
        front_frame_np = camera_front.read().rgb.cpu().numpy()
        bottom_frame_np = camera_bottom.read().rgb.cpu().numpy()
        obs_state, r_obstacle_d, l_obstacle_d = get_sensor_data()
        helipad_dx, helipad_dy, helipad_dz, yaw_diff, f_obstacle_d = obs_state

        action = policy.calculate_drone_actions(np.array([helipad_dz], dtype=np.float32), front_frame_np, bottom_frame_np)
        move_x, move_z, yaw = action[0].tolist()
        yaw = np.clip(yaw, -1, 1)
        move_z = np.clip(move_z, -1, 1)
        # restrict vertical move if is far from helipad
        if abs(helipad_dx) > 0.5:
            move_z = 0
        move_x = np.clip(move_x, 0.15, 1)
        print(f"Action: move_x={move_x:.2f}, move_z={move_z:.2f}, yaw={yaw:.2f}")
        wv = body_to_world_vel(move_x * 5, 0, move_z)
        target_velocity = np.array([wv[0], wv[1], wv[2], 0, 0, yaw])
        drone.set_propellels_rpm([10000, 10000, 10000, 10000])  # just decoration
        drone.set_dofs_velocity(velocity=target_velocity)
        scene.step()

        # if check_contact(helipad_dx, helipad_dy, helipad_dz):
        #     print("Helipad reached!")          
        #     break
    
    scene.viewer.stop()


