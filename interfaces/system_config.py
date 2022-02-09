class SystemConfig:
    inference_root_path: str
    log_file_root_path: str
    frame_interval: int
    margin: int
    ear_threshold: float
    gpu: int

    def __init__(self,
                 inference_root_path: str,
                 frame_interval: int = 1,
                 margin: int = 10,
                 ear_threshold: float = 0.5,
                 gpu: int = -1
                 ):
        self.inference_root_path = inference_root_path
        self.frame_interval = frame_interval
        self.margin = margin
        self.ear_threshold = ear_threshold
        self.gpu = gpu
