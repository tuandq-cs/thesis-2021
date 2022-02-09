import os

from appearance_module.hyperface import Hyperface
from detection_module.detector.mtcnn_model import MtcnnModel
from interfaces.system_config import SystemConfig
from landmark_module.landmark68.dlib_model import LandmarkDlib
from reenactment_module.landmark_based.swapping68 import SwappingBy68Landmarks
from smile_module.enet import Enet
from system import SystemWrapper
from tracking_module.sort import Sort

if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.abspath(__file__))
    inference_root_path = os.path.join(project_dir, 'inferences')

    # CONFIG
    config = SystemConfig(inference_root_path=inference_root_path,
                          ear_threshold=0.3,
                          margin=10,
                          gpu=-1
                          )
    # DEFINE MODELS
    # 1. DetectionModel
    detector = MtcnnModel(model_path='', conf_threshold=0.85, margin=config.margin)
    # 2. TrackingModel
    tracking_model = Sort()
    # 3. AppearanceModel
    appearance_model = Hyperface(model_path='models/model_epoch_190', vis_threshold=0.5, config=config)
    # appearance_model = TestAppearanceModel(model_path='', config=config)
    # 4. SmileModel
    smile_model = Enet(model_path='models/enet_b2_8.pt', device='cpu')
    # smile_model = LenetSmile(model_path='models/lenet_smiles.hdf5', device='cpu')
    # 5. LandmarkModel
    landmark_model = LandmarkDlib(model_path='models/shape_predictor_68_face_landmarks.dat', config=config)
    # 6. ReenactmentModel
    reenactment_model = SwappingBy68Landmarks()

    system = SystemWrapper(config=config,
                           detection_model=detector,
                           tracking_model=tracking_model,
                           appearance_model=appearance_model,
                           smile_model=smile_model,
                           landmark_model=landmark_model,
                           reenactment_model=reenactment_model)
    result = system.get_result(
        "/storageStudents/tuanld/tuan-quynh/tuan-dong/Thesis/videos/group_selfie_3/group_selfie_3.mp4")
