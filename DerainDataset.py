import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
import PIL.Image as Image
from numpy.random import RandomState

class TrainDataset(udata.Dataset):
    def __init__(self, name, gtname,patchsize,length):
        super().__init__()
        self.dataset = name
        self.gtdata=gtname
        self.patch_size=patchsize
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(self.dataset)
        self.gt_dir = os.path.join(self.gtdata)
        self.mat_files = os.listdir(self.root_dir)
        self.file_num = len(self.mat_files)
        self.sample_num = length

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        O = cv2.imread(img_file)
        b, g, r = cv2.split(O)
        input_img = cv2.merge([r, g, b])
        O,row,col= self.crop(input_img)


        gt_file = os.path.join(self.gt_dir, file_name)
        B = cv2.imread(gt_file)
        b, g, r = cv2.split(B)
        gt_img = cv2.merge([r, g, b])
        B = gt_img[row: row + self.patch_size, col : col + self.patch_size]
        O, B = self.augment(O, B)
        O = O.astype(np.float32)
        O = np.transpose(O, (2, 0, 1))
        B = B.astype(np.float32)
        B = np.transpose(B, (2, 0, 1))

        return torch.Tensor(O), torch.Tensor(B)

    def crop(self, img):
        h, w, c = img.shape
        p_h, p_w = self.patch_size, self.patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img[r: r + p_h, c : c + p_w]
        return O,r,c
    def augment(self, O, B):
        if random.random() < 0.5:
            O = O[:,::-1,:]
            B = B[:, ::-1, :]
        return O,B


class SPATrainDataset(udata.Dataset):
    def __init__(self, dir, sub_mat_file, patchSize,sample_num,train_num):
        super().__init__()
        self.dir = dir
        self.patch_size = patchSize
        self.sample_num= sample_num
        self.train_num = train_num
        self.sub_files=sub_mat_file
        self.rand_state = RandomState(66)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        file_name = self.sub_files[idx % int(self.train_num)]
        input_file_name = file_name.split(' ')[0]
        gt_file_name = file_name.split(' ')[1][:-1]

        O = cv2.imread(self.dir+input_file_name)
        b, g, r = cv2.split(O)
        input_img = cv2.merge([r, g, b])
        O,row,col= self.crop(input_img)


        B = cv2.imread(self.dir+ gt_file_name)
        b, g, r = cv2.split(B)
        gt_img = cv2.merge([r, g, b])
        B = gt_img[row: row + self.patch_size, col : col + self.patch_size]


        O, B = self.augment(O, B)
        O = O.astype(np.float32)
        O = np.transpose(O, (2, 0, 1))
        B = B.astype(np.float32)
        B = np.transpose(B, (2, 0, 1))


        return torch.Tensor(O), torch.Tensor(B)

    def crop(self, img):
        h, w, c = img.shape
        p_h, p_w = self.patch_size, self.patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img[r: r + p_h, c : c + p_w]
        return O,r,c
    def augment(self, O, B):
        if random.random() < 0.5:
            O = O[:,::-1,:]
            B = B[:, ::-1, :]
        return O,B
