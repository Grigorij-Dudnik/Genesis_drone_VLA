import numpy as np
# import math
# from pynput import keyboard

# ---------------------------------------------------------------------------
# Manual flight policy — keyboard-driven
# ---------------------------------------------------------------------------

# _pressed = set()

# def _on_press(key):
#     try:
#         _pressed.add(key.char.lower())
#     except AttributeError:
#         _pressed.add(key)

# def _on_release(key):
#     try:
#         _pressed.discard(key.char.lower())
#     except AttributeError:
#         _pressed.discard(key)

# _listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
# _listener.start()


# def manual_flight_policy(speed: float = 1.0) -> np.ndarray:
#     """
#     Arrow keys — horizontal movement (Up=forward, Down=back, Left=left, Right=right)
#     Q / E      — yaw left / right
#     Z / X      — descend / ascend
#     Returns np.array([move_x, move_y, move_z, move_yaw], dtype=float32)
#     """
#     move_x   = float(keyboard.Key.up    in _pressed) - float(keyboard.Key.down  in _pressed)
#     move_y   = float(keyboard.Key.left  in _pressed) - float(keyboard.Key.right in _pressed)
#     move_z   = float('x' in _pressed) - float('z' in _pressed)
#     move_yaw = float('q' in _pressed) - float('e' in _pressed)

#     return np.array([move_x, move_y, move_z, move_yaw], dtype=np.float32) * speed


# ---------------------------------------------------------------------------
# Autoflight policy — sensor-driven
# ---------------------------------------------------------------------------

def autoflight_policy(sensor_data, r_obstacle_d, l_obstacle_d):   
    helipad_distance_x, helipad_distance_y, helipad_distance_z, yaw_diff, f_obstacle_d = sensor_data
    move_x = (helipad_distance_x) * 0.7
    move_x = np.clip(move_x, -1, 1)
    move_z = 0
    move_yaw = np.clip(yaw_diff * 5, -1, 1)

    

    if helipad_distance_x < 0.8 and helipad_distance_y < 0.8:
        move_z = helipad_distance_z
        move_z = np.clip(move_z, -1, 1)
    # elif f_obstacle_d < 3.0:
    #     move_y = 0.5 * -math.copysign(1, move_y)
    # elif r_obstacle_d < 0.6:
    #     move_y = -(1 - 0.6 / r_obstacle_d)
    #     move_y = np.clip(move_y, -1, 1)
    # elif l_obstacle_d < 0.6:
    #     move_y = (1 - 0.6 / l_obstacle_d)
    #     move_y = np.clip(move_y, -1, 1)

    if f_obstacle_d < 0.4:
        move_x = 0

    return np.array([move_x, move_z, move_yaw], dtype=np.float32)
