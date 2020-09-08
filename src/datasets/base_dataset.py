"""
Adapted from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
Original licence: Copyright (c) Microsoft, under the MIT License.
"""

import copy
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

from ..transforms import get_affine_transform
from ..transforms import affine_transform
from ..transforms import fliplr_joints


class BaseDataset(Dataset, ABC):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.flip_pairs = None

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.scale_factor = cfg.dataset.scale_factor
        self.rotation_factor = cfg.dataset.rot_factor
        self.flip = cfg.dataset.flip

        self.input_w = cfg.model.input_w
        self.input_h = cfg.model.input_h
        self.output_w = cfg.model.output_w
        self.output_h = cfg.model.output_h

        self.sigma = cfg.model.sigma

        self.unbiased_encoding = cfg.dataset.unbiased_encoding

        self.transform = transform
        self.db = []

    @abstractmethod
    def _get_db(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        img = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if img is None:
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_visible = db_rec['joints_3d_visible']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                img = img[:, ::-1, :]
                joints, joints_visible = fliplr_joints(
                    joints, joints_visible, img.shape[1], self.flip_pairs)
                c[0] = img.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, (self.input_w, self.input_h))
        inputs = cv2.warpAffine(
            img,
            trans,
            (self.input_w, self.input_h),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            inputs = self.transform(inputs)

        for i in range(self.num_joints):
            if joints_visible[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_visible)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_visible': joints_visible,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return inputs, target, target_weight, meta

    def generate_target(self, joints, joints_visible):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_visible[:, 0]

        target = np.zeros((self.num_joints,
                           self.output_h,
                           self.output_w),
                           dtype=np.float32)

        tmp_size = self.sigma * 3

        if self.unbiased_encoding:
            for joint_id in range(self.num_joints):
                heatmap_vis = joints_visible[joint_id, 0]
                target_weight[joint_id] = heatmap_vis

                feat_stride = [
                    self.input_w / self.output_w,
                    self.input_h / self.output_h,
                ]
                mu_x = joints[joint_id][0] / feat_stride[0]
                mu_y = joints[joint_id][1] / feat_stride[1]
                # Check that any part of the gaussian is in-bounds
                ul = [mu_x - tmp_size, mu_y - tmp_size]
                br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
                if ul[0] >= self.output_w or ul[1] >= self.output_h or br[
                        0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] == 0:
                    continue

                x = np.arange(0, self.output_w, 1, np.float32)
                y = np.arange(0, self.output_h, 1, np.float32)
                y = y[:, None]

                if target_weight[joint_id] > 0.5:
                    target[joint_id] = np.exp(
                        -((x - mu_x)**2 + (y - mu_y)**2) / (2 * self.sigma**2))
        else:
            for joint_id in range(self.num_joints):
                heatmap_vis = joints_visible[joint_id, 0]
                target_weight[joint_id] = heatmap_vis

                feat_stride = [
                    self.input_w / self.output_w,
                    self.input_h / self.output_h,
                ]
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.output_w or ul[1] >= self.output_h or br[
                        0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] > 0.5:
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, None]
                    x0 = y0 = size // 2
                    # The gaussian is not normalized,
                    # we want the center value to equal 1
                    g = np.exp(-((x - x0)**2 + (y - y0)**2) /
                               (2 * self.sigma**2))

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], self.output_w) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], self.output_h) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.output_w)
                    img_y = max(0, ul[1]), min(br[1], self.output_h)

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
