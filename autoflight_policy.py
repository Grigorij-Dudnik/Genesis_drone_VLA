import numpy as np
import math

def autoflight_policy(sensor_data, r_obstacle_d, l_obstacle_d):   
    helipad_distance_x, helipad_distance_y, helipad_distance_z, yaw_diff, f_obstacle_d = sensor_data
    helipad_distance_x_y = math.hypot(helipad_distance_x, helipad_distance_y)
    #move_x = helipad_distance_x * helipad_distance_x * 0.7
    move_x = 1.0 - math.exp(-5 * helipad_distance_x)
    move_x = np.clip(move_x, -1, 1)
    move_z = 0
    move_yaw = np.clip(yaw_diff * 5, -1, 1)

    

    if helipad_distance_x_y < 0.6:
        #move_z = helipad_distance_z
        #move_z = np.clip(move_z, -1, 1)
        move_z = -1
    # elif f_obstacle_d < 3.0:
    #     move_y = 0.5 * -math.copysign(1, move_y)
    # elif r_obstacle_d < 0.6:
    #     move_y = -(1 - 0.6 / r_obstacle_d)
    #     move_y = np.clip(move_y, -1, 1)
    # elif l_obstacle_d < 0.6:
    #     move_y = (1 - 0.6 / l_obstacle_d)
    #     move_y = np.clip(move_y, -1, 1)


    return np.array([move_x, move_z, move_yaw], dtype=np.float32)
