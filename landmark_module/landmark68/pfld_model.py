import numpy as np
import torch
import torchvision.transforms

from interfaces.face import Face
from interfaces.system_config import SystemConfig
from landmark_module.base import LandmarkModel
from landmark_module.landmark68.pfld_pytorch.models.pfld import PFLDInference


class LandmarkPFLD(LandmarkModel):
    name = 'PFLDLandmark'

    _transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )

    def _load_model(self, config: SystemConfig) -> None:
        self._device = torch.device('cuda' if config.gpu >= 0 and torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(self.model_path, map_location=self._device)
        model = PFLDInference().to(self._device)
        model.load_state_dict(checkpoint['pfld_backbone'])
        model.eval()
        model = model.to(self._device)
        self.model = model

    def detect(self, face_info: Face, frame: np.ndarray) -> np.ndarray:
        pass
