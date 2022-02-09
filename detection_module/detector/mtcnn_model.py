from typing import List

import numpy as np
from mtcnn import MTCNN

from detection_module.detector.base import Detector
from interfaces.bounding_box import BoundingBox
from interfaces.face import Face


class MtcnnModel(Detector):
    name = 'MTCNN'

    def _load_model(self) -> None:
        self.model = MTCNN()

    def detect_face(self, frame: np.ndarray) -> List[Face]:
        list_result = self.model.detect_faces(frame)
        list_faces = []
        for info in list_result:
            conf_score = info['confidence']
            if conf_score > self.conf_threshold:
                x_left, y_left, w, h = info['box']
                bbox = BoundingBox(x=x_left + w / 2, y=y_left + h / 2, w=w * 1.0, h=h * 1.0)
                face_info = Face(bbox=bbox, conf_score=conf_score)
                list_faces.append(face_info)
        return list_faces
