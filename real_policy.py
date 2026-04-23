from lerobot.utils.control_utils import predict_action
from lerobot.policies.factory import make_pre_post_processors, get_policy_class
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.device_utils import get_safe_torch_device
import torch
import numpy as np


class PolicyInference:
    def __init__(self, policy_name, dataset_name, task):
        self.policy_name = policy_name
        #self.policy = PreTrainedPolicy.from_pretrained(policy_name).to(device).eval()
        self.policy_cfg = PreTrainedConfig.from_pretrained(policy_name)
        
        # xvla only
        self.policy_cfg.tokenizer_max_length = 50
        self.policy_cfg.num_image_views = 2

        policy_class = get_policy_class(self.policy_cfg.type)
        #torch.set_default_device("cpu") # pi05 lerobot 0.5.1 bug walkaround
        self.policy = policy_class.from_pretrained(policy_name, config=self.policy_cfg).to("cuda").eval()
        self.task = task
        self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=self.policy_cfg,
        )


    def calculate_drone_actions(self, sensor_data, front_frame_np, bottom_frame_np):
        # check if policy name after "/" starts with "smolvla"
        if self.policy_name.split("/")[-1].startswith("smolvla"):
            # Use the smolvla policy
            observation_frame = {
            "observation.images.camera1": front_frame_np,
            "observation.images.camera2": bottom_frame_np,
            "observation.state": sensor_data,
        }
        elif self.policy_name.split("/")[-1].startswith("xvla"):
            observation_frame = {
                "observation.images.image": front_frame_np,
                "observation.images.image2": bottom_frame_np,
                "observation.state": sensor_data,
            }
        else:
            observation_frame = {
                "observation.images.camera_front": front_frame_np,
                "observation.images.camera_bottom": bottom_frame_np,
                "observation.state": sensor_data,
            }
        # save frame for debug as png
        # from PIL import Image
        # Image.fromarray(front_frame_np).save("front_frame.png")
        # Image.fromarray(bottom_frame_np).save("bottom_frame.png")


        action = predict_action(
                observation=observation_frame,
                policy=self.policy,
                device=get_safe_torch_device("cuda"),
                task=self.task,
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                use_amp=self.policy_cfg.use_amp,
            )
        return action