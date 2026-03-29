from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset_repo = "Grigorij/drone_flight"

dataset_features = {
    "observation.images.camera_front": {
        "dtype": "image",
        "shape": (480, 640, 3),       # H x W x C
    },
    "observation.images.camera_bottom": {
        "dtype": "image",
        "shape": (480, 640, 3),
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (4,),               
        "names": ["dx","dy","dz","dyaw"],
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
                root="dataset/",
                #encoder_threads=cfg.dataset.encoder_threads,
            )

