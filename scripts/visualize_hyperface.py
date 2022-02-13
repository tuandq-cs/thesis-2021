import cv2

from appearance_module.hyperface import Hyperface
from interfaces.system_config import SystemConfig

if __name__ == '__main__':
    # CONFIG
    config = SystemConfig(inference_root_path='',
                          ear_threshold=0.3,
                          margin=0,
                          gpu=-1
                          )
    img = cv2.imread('../test_imgs/test_hyperface.png')
    model = Hyperface(model_path='../models/model_epoch_190', vis_threshold=0.5, config=config)
    model.visualize(cropped_face=img)
    print('Done')
