import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from smile_module.base import SmileModel

IMG_SIZE = (224, 224)

TRANS_COMP = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class Enet(SmileModel):
    name = 'Enet'

    def _load_model(self) -> None:
        self.model = torch.load(self.model_path, map_location=torch.device(self.device))

    @staticmethod
    def _preprocess(cropped_face: np.ndarray):
        img = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        img_tensor = TRANS_COMP(Image.fromarray(img))
        img_tensor.unsqueeze_(0)
        return img_tensor

    def _get_output(self, inp) -> float:
        scores = self.model(inp)
        scores = scores[0].data.cpu().numpy()
        if np.argmax(scores) != 4:
            return 0.0
        else:
            norm = softmax(scores)
            return norm[4]

    def compute_score(self, cropped_face: np.ndarray) -> float:
        inp = self._preprocess(cropped_face)
        return self._get_output(inp.to(self.device))
