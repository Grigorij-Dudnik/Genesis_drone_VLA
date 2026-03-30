from lerobot.utils.control_utils import predict_action


class PolicyInference:
    def __init__(self, policy_name, task):
        self.policy_name = policy_name
        self.task = task

    def calculate_drone_actions(self, sensor_data, front_frame_np, bottom_frame_np):
        obs_frame = {
            "camera1": front_frame_np,
            "camera2": bottom_frame_np,
            "observation.state": sensor_data["observation.state"],
        }
        action = predict_action(
                observation=observation_frame,
                policy=self.policy_name,
                device="cuda",
                task=self.task,
            )
        return action