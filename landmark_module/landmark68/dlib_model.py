import dlib
import numpy as np

from interfaces.face import Face
from interfaces.system_config import SystemConfig
from landmark_module.base import LandmarkModel


class LandmarkDlib(LandmarkModel):
    name = 'DlibLandmark'

    def _load_model(self, config: SystemConfig) -> None:
        self.model = dlib.shape_predictor(self.model_path)

    def detect(self, face_info: Face, frame: np.ndarray) -> np.ndarray:
        x_min, y_min, x_max, y_max = face_info.bbox.to_rect().astype(dtype=np.int32)
        facial_landmarks = self.model(frame, dlib.rectangle(x_min, y_min, x_max, y_max))
        return np.array([(point.x, point.y) for point in facial_landmarks.parts()])
