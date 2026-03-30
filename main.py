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


INFERENCE = True

scene.viewer.follow_entity(drone)

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
    if abs(helipad_dx) < 0.5 and abs(helipad_dy) < 0.5 and abs(helipad_dz) < 0.02:
        return True
    return False

nr_episodes = 50
for _ in range(nr_episodes):
    
    scene.sim.rigid_solver.add_weld_constraint(
        drone.links[0].idx,
        front_camera_mount.links[0].idx
    )

    while not INFERENCE:
        front_frame_np = camera_front.read().rgb.cpu().numpy()
        bottom_frame_np = camera_bottom.read().rgb.cpu().numpy()
        obs_state, r_obstacle_d, l_obstacle_d = get_sensor_data()
        helipad_dx, helipad_dy, helipad_dz, yaw_diff, f_obstacle_d = obs_state

        action = autoflight_policy(obs_state, r_obstacle_d, l_obstacle_d)
        move_x, move_y, move_z, yaw = action
        print(f"Action: move_x={action[0]:.2f}, move_y={action[1]:.2f}, move_z={action[2]:.2f}, yaw={action[3]:.2f}")
        target_velocity = np.array([move_x * 5, move_y * 5, move_z * 5, 0, 0, yaw])
        drone.set_propellels_rpm([10000, 10000, 10000, 10000])  # just decoration
        drone.set_dofs_velocity(velocity=target_velocity)

        dataset.add_frame({
            "observation.images.camera_front":  front_frame_np,
            "observation.images.camera_bottom": bottom_frame_np,
            "observation.state":                obs_state,
            "action":                           action,
            "task":                             "fly to helipad avoiding obstacles",
        })
        
        scene.step()

        if check_contact(helipad_dx, helipad_dy, helipad_dz):
            print("Helipad reached!")
            dataset.save_episode()
            #scene.stop_recording()
            scene.reset()
            drone_pos_y = random.uniform(-5, 5)
            drone.set_pos((0, drone_pos_y, 1))
            front_camera_mount.set_pos((0, drone_pos_y, 1))
            drone.set_quat((1, 0, 0, 0))
            front_camera_mount.set_quat((0.5, 0.5, -0.5, -0.5))
            break
        #time.sleep(0.03)

if INFERENCE:
    policy = PolicyInference(policy_name="Grigorij/drone_flight_policy", task="fly to helipad avoiding obstacles")
    scene.sim.rigid_solver.add_weld_constraint(
        drone.links[0].idx,
        front_camera_mount.links[0].idx
    )
while INFERENCE:
    front_frame_np = camera_front.read().rgb.cpu().numpy()
    bottom_frame_np = camera_bottom.read().rgb.cpu().numpy()
    obs_state, r_obstacle_d, l_obstacle_d = get_sensor_data()
    helipad_dx, helipad_dy, helipad_dz, yaw_diff, f_obstacle_d = obs_state

    
    move_x, move_y, move_z, yaw = policy.calculate_drone_actions(obs_state, front_frame_np, bottom_frame_np)
    print(f"Action: move_x={action[0]:.2f}, move_y={action[1]:.2f}, move_z={action[2]:.2f}, yaw={action[3]:.2f}")
    target_velocity = np.array([move_x * 5, move_y * 5, move_z * 5, 0, 0, yaw])
    drone.set_propellels_rpm([10000, 10000, 10000, 10000])  # just decoration
    drone.set_dofs_velocity(velocity=target_velocity)

    #time.sleep(0.03)

dataset.finalize()
dataset.push_to_hub()

