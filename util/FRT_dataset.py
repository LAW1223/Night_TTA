# By Yuxiang Sun, Jul. 3, 2021
# Email: sun.yuxiang@outlook.com

import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL
from torchvision import transforms

class RFT_dataset(Dataset):
    def __init__(self, data_dir, split, input_h=512, input_w=640 ,transform=[]):
        super(RFT_dataset, self).__init__()

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.n_data    = len(self.names)

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s' % (folder, name))
        image     = np.asarray(PIL.Image.open(file_path))
        return image

    def read_thermal(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s' % (folder, name))
        image=PIL.Image.open(file_path)
        image=image.convert("L")
        image= np.asarray(image)
        return image

    def __getitem__(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'RGB')
        Thermal= self.read_thermal(name, 'Thermal')
        # print(Thermal.shape)
        label = self.read_image(name, 'Labels')
        for func in self.transform:
            image, label = func(image, label)
        # print(image.shape)
        image = np.asarray(PIL.Image.fromarray(np.uint8(image)).resize((self.input_w, self.input_h)))
        image = image.astype('float32')
        image = np.transpose(image, (2,0,1))
        Thermal = np.asarray(PIL.Image.fromarray(np.uint8(Thermal)).resize((self.input_w, self.input_h)))
        Thermal = Thermal.astype('float32')
        Thermal = Thermal/255.0
        label = np.asarray(PIL.Image.fromarray(np.uint8(label)).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST))
        label = label.astype('int64')
        
        return torch.tensor(image), torch.tensor(Thermal),torch.tensor(label), name

    def __len__(self):
        return self.n_data