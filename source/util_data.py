#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018

@author: maximov
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

import os
from os import listdir, mkdir
from os.path import isfile, join, isdir, exists
import numpy as np
import importlib
import pickle
import random
import math
from PIL import Image
from collections import defaultdict
import cv2
import numbers


class ImageDataset(torch.utils.data.Dataset):
    """Focal place dataset."""
    def __init__(self, root_dir, label_num=1200, transform_fnc=transforms.Compose([transforms.ToTensor()]),
                 img_size=128, flag_init=True, flag_sample=2, flag_augment=True):

        self.root_dir = root_dir
        with open(os.path.join(root_dir, 'landmarks.pkl'), 'rb') as f:
            self.landmarks = pickle.load(f)
        self.transform_fnc = transform_fnc
        if isinstance(img_size, tuple):
            self.img_shape = img_size
        else:
            self.img_shape = (img_size, img_size)
        self.flag_sample = flag_sample

        self.root_img = root_dir
        self.im_label, self.im_paths, self.im_index = [], [], []

        self.flag_augment = flag_augment

        it_j = 0
        for person_id in os.listdir(root_dir):
            if not os.path.isdir(os.path.join(root_dir, person_id)):
                continue
            imglist_all = [f for f in listdir(os.path.join(root_dir, person_id)) if f[-4:] in [".jpg", ".png"]]
            # imglist_all_int = [int(x[:-4]) for x in imglist_all]
            # imglist_all_int.sort()
            # imglist_all = [(str(x).zfill(6) + ".jpg") for x in imglist_all_int]
            # imglist_all = sorted(imglist_all)

            self.im_label += [int(person_id)] * len(imglist_all)
            self.im_paths += imglist_all
            self.im_index += [it_j] * len(imglist_all)
            it_j += 1
        print("Dataset initialized")

    def __len__(self):
        return len(self.im_label)

    def get_landmark_img(self, img, landmark):
        canvas = np.ones_like(img) * 255
        landmark = landmark[:, :2]
        chin = landmark[0:17]
        left_brow = landmark[17:22]
        right_brow = landmark[22:27]
        left_eye = landmark[36:42]
        right_eye = landmark[42:48]
        nose1 = landmark[27:31]
        nose2 = landmark[31:36]
        mouth = landmark[48:60]
        mouth_internal = landmark[60:68]

        lines = [
            chin,
            mouth_internal,
            nose1
        ]

        color = (0, 0, 50)
        for line in lines:
            cv2.polylines(
                canvas,
                np.int32([line]), False,
                color, thickness=1, lineType=cv2.LINE_AA
            )

        return canvas

    def get_mask_img(self, img, landmark):
        landmark = landmark[:, :2]
        canvas = np.ones_like(img) * 255
        chin = landmark[0:17]
        cv2.fillPoly(
            canvas, np.int32([chin]), (0, 0, 0),
        )
        return canvas

    def denorm(self, landmarks, img):
        landmarks = landmarks.copy()
        landmarks[:, 0] = landmarks[:, 0] * img.shape[1]
        landmarks[:, 1] = landmarks[:, 1] * img.shape[0]
        return landmarks

    def load_img(self, im_path, im_array=None):
        if im_array is not None:
            im = Image.fromarray(im_array)
        else:
            im = Image.open(im_path)

        w, h = im.size

        if self.flag_augment:
            offset_h = 0.1
            center_h = h / 2 + offset_h * h
            center_w = w / 2
            min_sz, max_sz = w / 2, (w - center_w) * 1.5
            diff_sz, crop_sz = (max_sz - min_sz) / 2, min_sz / 2

            img_res = im.crop(
                (int(center_w - crop_sz - diff_sz * self.crop_rnd[0]), int(center_h - crop_sz - diff_sz * self.crop_rnd[1]),
                 int(center_w + crop_sz + diff_sz * self.crop_rnd[2]), int(center_h + crop_sz + diff_sz * self.crop_rnd[3])))
        else:
            offset_h = 0.1
            center_h = h / 2 + offset_h * h
            center_w = w / 2
            min_sz, max_sz = w / 2, (w - center_w) * 1.5
            diff_sz, crop_sz = (max_sz - min_sz) / 2, min_sz / 2
            img_res = im.crop(
                (int(center_w - crop_sz - diff_sz),
                 int(center_h - crop_sz - diff_sz),
                 int(center_w + crop_sz + diff_sz),
                 int(center_h + crop_sz + diff_sz)))

        img_res = img_res.resize(self.img_shape, resample=Image.LANCZOS)
        return self.transform_fnc(img_res)

    def __getitem__(self, idx):
        im_clr, im_lndm, im_msk, im_ind = [], [], [], []
        if self.flag_sample == 1:
            idx = [idx]

        for k_iter in range(self.flag_sample):
            self.crop_rnd = [random.random(), random.random(), random.random(), random.random()]

            im_clr_path = os.path.join(self.root_dir, str(self.im_label[idx[k_iter]]), self.im_paths[idx[k_iter]])
            img = cv2.cvtColor(cv2.imread(im_clr_path), cv2.COLOR_RGB2BGR)
            clr_img = self.load_img(im_clr_path, im_array=img)
            im_clr.append(clr_img)

            key_path = os.path.join(im_clr_path.split('/')[-2], os.path.basename(im_clr_path))
            landmark = self.landmarks[key_path]
            landmark = self.denorm(landmark, img)
            lndm_img = self.load_img(None, im_array=self.get_landmark_img(img, landmark))
            im_lndm.append(lndm_img)

            msk = ((1 - self.load_img(None, im_array=self.get_mask_img(img, landmark))) > 0.2)

            im_msk.append(msk)

            im_ind.append(self.im_index[idx[k_iter]])

        return im_clr, im_lndm, im_msk, im_ind


def load_data(DATA_PATH, WORKERS_NUM, BATCH_SIZE, IMG_SIZE, FLAG_DATA_AUGM, LABEL_NUM, mode_train=True):
    ##### Data loaders
    data_dir = DATA_PATH
    if mode_train:
        dataset_train = ImageDataset(root_dir=data_dir, label_num=LABEL_NUM, transform_fnc=transforms.Compose([transforms.ToTensor()]),
                                     img_size=IMG_SIZE, flag_augment=FLAG_DATA_AUGM)
        total_steps = int(len(dataset_train) / BATCH_SIZE)

        ddict = defaultdict(list)
        for idx, label in enumerate(dataset_train.im_label):
            ddict[label].append(idx)

        list_of_indices_for_each_class = []
        for key in ddict:
            list_of_indices_for_each_class.append(ddict[key])
        loader_train = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=WORKERS_NUM, batch_size=BATCH_SIZE, shuffle=False, sampler=SiameseSampler(list_of_indices_for_each_class, BATCH_SIZE, total_steps))

        print("Total number of steps per epoch:", total_steps)
        print("Total number of training samples:", len(dataset_train))
        return loader_train, total_steps, LABEL_NUM
    else:
        label_num = 363
        dataset_test = ImageDataset(root_dir=data_dir, label_num=label_num,transform_fnc=transforms.Compose([transforms.ToTensor()]), img_size = IMG_SIZE)
        loader_test = torch.utils.data.DataLoader(dataset=dataset_test, num_workers=1, batch_size=1, shuffle=False)
        print("Total number of test samples:", len(dataset_test))
        return loader_test, len(dataset_test), label_num


class SiameseSampler(Sampler):

    def __init__(self, l_inds, batch_size, iterations_per_epoch):
        self.l_inds = l_inds
        self.max = -1
        self.batch_size = batch_size
        self.flat_list = []
        self.iterations_per_epoch = iterations_per_epoch

    def __iter__(self):
        self.flat_list = []

        for ii in range(int(self.iterations_per_epoch)):
            # get half of the images randomly
            sep = int(self.batch_size / 2)
            for i in range(sep):
                first_class = random.choice(self.l_inds)
                second_class = random.choice(self.l_inds)
                first_element = random.choice(first_class)
                second_element = random.choice(second_class)
                self.flat_list.append([first_element, second_element])

            # get the last half as images from the same class
            for i in range(sep, self.batch_size):
                c_class = random.choice(self.l_inds)
                first_element = random.choice(c_class)
                second_element = random.choice(c_class)
                self.flat_list.append([first_element, second_element])

        random.shuffle(self.flat_list)
        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)
