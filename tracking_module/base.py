from abc import ABCMeta, abstractmethod
from typing import List, Any

from interfaces.face import Face


class TrackingModel(metaclass=ABCMeta):
    name: str

    @abstractmethod
    def update(self, list_faces: List[Face], img_size: Any, predict_num: int) -> List[Face]:
        pass

    @abstractmethod
    def reset(self):
        pass
