from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os
import shutil

dataset_repo = "Grigorij/drone_flight"
root = "dataset/"
# remove root directory if it already exists
if os.path.exists(root):
    shutil.rmtree(root)

dataset_features = {
    "observation.images.camera_front": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.camera_bottom": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (5,),               
        "names": ["dx","dy","dz","dyaw","d_obst_f"],
    },
    "action": {
        "dtype": "float32",
        "shape": (4,),               
        "names": ["vx", "vy", "vz", "yaw"],
    },
}

dataset = LeRobotDataset.create(
                dataset_repo,
                fps=30,
                features=dataset_features,
                root=root,
                #encoder_threads=cfg.dataset.encoder_threads,
            )

