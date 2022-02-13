from typing import Any

import chainer
import chainer.functions as F
import chainer.links as L
import cv2
import numpy as np

from appearance_module.base import AppearanceModel
from interfaces.system_config import SystemConfig
# Constant variables
from utils.face import FACIAL_LANDMARKS_21_IDXS

N_LANDMARK = 21
IMG_SIZE = (227, 227)


class HyperFaceModel(chainer.Chain):
    def __init__(self, loss_weights=(1.0, 100.0, 20.0, 5.0, 0.3)):
        super(HyperFaceModel, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 96, 11, stride=4, pad=0)
            self.conv1a = L.Convolution2D(96, 256, 4, stride=4, pad=0)
            self.conv2 = L.Convolution2D(96, 256, 5, stride=1, pad=2)
            self.conv3 = L.Convolution2D(256, 384, 3, stride=1, pad=1)
            self.conv3a = L.Convolution2D(384, 256, 2, stride=2, pad=0)
            self.conv4 = L.Convolution2D(384, 384, 3, stride=1, pad=1)
            self.conv5 = L.Convolution2D(384, 256, 3, stride=1, pad=1)
            self.conv_all = L.Convolution2D(768, 192, 1, stride=1, pad=0)
            self.fc_full = L.Linear(6 * 6 * 192, 3072)
            self.fc_detection1 = L.Linear(3072, 512)
            self.fc_detection2 = L.Linear(512, 2)
            self.fc_landmarks1 = L.Linear(3072, 512)
            self.fc_landmarks2 = L.Linear(512, 42)
            self.fc_visibility1 = L.Linear(3072, 512)
            self.fc_visibility2 = L.Linear(512, 21)
            self.fc_pose1 = L.Linear(3072, 512)
            self.fc_pose2 = L.Linear(512, 3)
            self.fc_gender1 = L.Linear(3072, 512)
            self.fc_gender2 = L.Linear(512, 2)
        self.train = True
        self.report = True
        self.backward = True
        assert (len(loss_weights) == 5)
        self.loss_weights = loss_weights

    @staticmethod
    def _disconnect(x):
        return chainer.Variable(x.data)

    def __call__(self, x_img, t_detection=None, t_landmark=None,
                 t_visibility=None, t_pose=None, t_gender=None,
                 m_landmark=None, m_visibility=None, m_pose=None):
        # Alexnet
        h = F.relu(self.conv1(x_img))  # conv1
        h = F.max_pooling_2d(h, 3, stride=2, pad=0)  # max1
        h = F.local_response_normalization(h)  # norm1
        h1 = F.relu(self.conv1a(h))  # conv1a
        h = F.relu(self.conv2(h))  # conv2
        h = F.max_pooling_2d(h, 3, stride=2, pad=0)  # max2
        h = F.local_response_normalization(h)  # norm2
        h = F.relu(self.conv3(h))  # conv3
        h2 = F.relu(self.conv3a(h))  # conv3a
        h = F.relu(self.conv4(h))  # conv4
        h = F.relu(self.conv5(h))  # conv5
        h = F.max_pooling_2d(h, 3, stride=2, pad=0)  # pool5

        h = F.concat((h1, h2, h))

        # Fusion CNN
        h = F.relu(self.conv_all(h))  # conv_all
        h = F.relu(self.fc_full(h))  # fc_full
        h = F.dropout(h)

        h_detection = F.relu(self.fc_detection1(h))
        h_detection = F.dropout(h_detection)
        h_detection = self.fc_detection2(h_detection)
        h_landmark = F.relu(self.fc_landmarks1(h))
        h_landmark = F.dropout(h_landmark)
        h_landmark = self.fc_landmarks2(h_landmark)
        h_visibility = F.relu(self.fc_visibility1(h))
        h_visibility = F.dropout(h_visibility)
        h_visibility = self.fc_visibility2(h_visibility)
        h_pose = F.relu(self.fc_pose1(h))
        h_pose = F.dropout(h_pose)
        h_pose = self.fc_pose2(h_pose)
        h_gender = F.relu(self.fc_gender1(h))
        h_gender = F.dropout(h_gender)
        h_gender = self.fc_gender2(h_gender)

        # Mask and Loss
        if self.backward:
            # Landmark masking with visibility
            m_landmark_ew = F.stack((t_visibility, t_visibility), axis=2)
            m_landmark_ew = F.reshape(m_landmark_ew, (-1, N_LANDMARK * 2))

            # Masking
            h_landmark *= self._disconnect(m_landmark)
            t_landmark *= self._disconnect(m_landmark)
            h_landmark *= self._disconnect(m_landmark_ew)
            t_landmark *= self._disconnect(m_landmark_ew)
            h_visibility *= self._disconnect(m_visibility)
            t_visibility *= self._disconnect(m_visibility)
            h_pose *= self._disconnect(m_pose)
            t_pose *= self._disconnect(m_pose)

            # Loss
            loss_detection = F.softmax_cross_entropy(h_detection, t_detection)
            loss_landmark = F.mean_squared_error(h_landmark, t_landmark)
            loss_visibility = F.mean_squared_error(h_visibility, t_visibility)
            loss_pose = F.mean_squared_error(h_pose, t_pose)
            loss_gender = F.softmax_cross_entropy(h_gender, t_gender)

            # Loss scaling
            loss_detection *= self.loss_weights[0]
            loss_landmark *= self.loss_weights[1]
            loss_visibility *= self.loss_weights[2]
            loss_pose *= self.loss_weights[3]
            loss_gender *= self.loss_weights[4]

            loss = (loss_detection + loss_landmark + loss_visibility +
                    loss_pose + loss_gender)

        # Prediction (the same shape as t_**, and [0:1])
        h_detection = F.softmax(h_detection)[:, 1]  # ([[y, n]] -> [d])
        h_gender = F.softmax(h_gender)[:, 1]  # ([[m, f]] -> [g])

        if self.report:
            if self.backward:
                # Report losses
                chainer.report({'loss': loss,
                                'loss_detection': loss_detection,
                                'loss_landmark': loss_landmark,
                                'loss_visibility': loss_visibility,
                                'loss_pose': loss_pose,
                                'loss_gender': loss_gender}, self)

            # Report results
            predict_data = {'img': x_img, 'detection': h_detection,
                            'landmark': h_landmark, 'visibility': h_visibility,
                            'pose': h_pose, 'gender': h_gender}
            teacher_data = {'img': x_img, 'detection': t_detection,
                            'landmark': t_landmark, 'visibility': t_visibility,
                            'pose': t_pose, 'gender': t_gender}
            chainer.report({'predict': predict_data}, self)
            chainer.report({'teacher': teacher_data}, self)

            # Report layer weights
            chainer.report({'conv1_w': {'weights': self.conv1.W},
                            'conv2_w': {'weights': self.conv2.W},
                            'conv3_w': {'weights': self.conv3.W},
                            'conv4_w': {'weights': self.conv4.W},
                            'conv5_w': {'weights': self.conv5.W}}, self)

        if self.backward:
            return loss
        else:
            return {'img': x_img, 'detection': h_detection,
                    'landmark': h_landmark, 'visibility': h_visibility,
                    'pose': h_pose, 'gender': h_gender}


class Hyperface(AppearanceModel):
    name = 'Hyperface'

    def _load_model(self, config: SystemConfig) -> None:
        model = HyperFaceModel()
        model.train = False
        model.report = False
        model.backward = False
        chainer.serializers.load_npz(self.model_path, model)
        self.model = model
        # Setup GPU
        self._use_cpu()

    def _use_cpu(self):
        self._xp = np

    # def _use_gpu(self, gpu: int):
    #     chainer.cuda.check_cuda_available()
    #     chainer.cuda.get_device_from_id(gpu).use()
    #     self.model.to_gpu()
    #     self._xp = cp

    @staticmethod
    def _cvt_variable(v: Any) -> np.ndarray:
        # Convert from chainer variable
        if isinstance(v, chainer.variable.Variable):
            v = v.data
            if hasattr(v, 'get'):
                v = v.get()
        return v

    def _preprocess(self, img: np.ndarray):
        img = img.astype(np.float32) / 255.0  # [0:1]
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.normalize(img, None, -0.5, 0.5, cv2.NORM_MINMAX)
        img = np.transpose(img, (2, 0, 1))
        # Create single batch
        imgs = self._xp.asarray([img])
        x = chainer.Variable(imgs)
        return x

    def _predict(self, inp):
        with chainer.no_backprop_mode():
            y = self.model(inp)
            img = self._cvt_variable(y['img'])[0]
            visibilities = self._cvt_variable(y['visibility'])[0]
            landmarks = self._cvt_variable(y['landmark'])[0]
        return img, landmarks, visibilities

    def _format_output(self, vis_landmark_estimated):
        visible_pts = [1] * 21
        # insert all landmarks are visible into "visible" dictionary
        visible = {}
        for key in FACIAL_LANDMARKS_21_IDXS:
            visible[key] = 1

        for i, point in enumerate(vis_landmark_estimated):
            if visible_pts[i] == 0:
                continue
            if point < self.vis_threshold:
                for key in FACIAL_LANDMARKS_21_IDXS:
                    if visible[key] == 0:
                        continue
                    i_start, i_end = FACIAL_LANDMARKS_21_IDXS[key]
                    i_range = range(i_start, i_end)
                    if i in i_range:
                        for j in i_range:
                            visible_pts[j] = 0
                        visible[key] = 0
        return visible

    def compute_score(self, cropped_face: np.ndarray) -> (bool, float):
        inp = self._preprocess(cropped_face)
        _, _, vis_landmark_estimated = self._predict(inp)
        vis_facial_parts = self._format_output(vis_landmark_estimated)
        total_score = 0.0
        if not vis_facial_parts['left_eye'] or not vis_facial_parts['right_eye']:
            return False, total_score
        for key in vis_facial_parts:
            if vis_facial_parts[key] == 1:
                total_score += 1
        return True, total_score / len(vis_facial_parts)

    def visualize(self, cropped_face: np.ndarray) -> None:
        inp = self._preprocess(cropped_face)
        img, landmarks, visibilities = self._predict(inp)
        img = np.transpose(img, (1, 2, 0))
        img = img.copy()
        img += 0.5
        draw_landmark(img, landmarks, visibilities)
        cv2.imwrite('test_6.png', img * 255)


def draw_landmark(img, landmarks, visibilities, vis_threshold=0.5, denormalize_scale=True):
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(int(landmarks.shape[-1] / 2), 2)
    assert (landmarks.shape[0] == 21 and visibilities.shape[0] == 21)

    if denormalize_scale:
        h, w = img.shape[0:2]
        size = np.array([[w, h]], dtype=np.float32)
        landmarks = landmarks * size + size / 2

    for pt, vis in zip(landmarks, visibilities):
        color = (0, 1, 0) if vis > vis_threshold else (0, 0, 1)
        pt = (int(pt[0]), int(pt[1]))
        # cv2.putText(img, '{:.2f}'.format(vis), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.1, color, 2)
        cv2.circle(img, pt, radius=4, color=color, thickness=-1)
