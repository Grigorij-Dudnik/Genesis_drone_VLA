from lerobot.utils.control_utils import predict_action
from lerobot.policies.factory import make_pre_post_processors, get_policy_class
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.factory import make_policy
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata


class PolicyInference:
    def __init__(self, policy_name, dataset_name, task):
        self.policy_name = policy_name
        #self.policy = PreTrainedPolicy.from_pretrained(policy_name).to(device).eval()
        self.policy_cfg = PreTrainedConfig.from_pretrained(policy_name)
        dataset_meta = LeRobotDatasetMetadata(dataset_name)
        policy_class = get_policy_class(self.policy_cfg.type)
        self.policy = policy_class.from_pretrained(policy_name, config=self.policy_cfg).to("cuda").eval()
        # rename_map = {
        #     "observation.images.camera_front": "observation.images.camera1",
        #     "observation.images.camera_bottom": "observation.images.camera2",
        # }
        # self.policy = make_policy(self.policy_cfg, dataset_meta, rename_map=rename_map)
        self.task = task
        self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=self.policy_cfg,
        )


    def calculate_drone_actions(self, sensor_data, front_frame_np, bottom_frame_np):
        observation_frame = {
            "observation.images.camera1": front_frame_np,
            "observation.images.camera2": bottom_frame_np,
            "observation.state": sensor_data,
        }
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