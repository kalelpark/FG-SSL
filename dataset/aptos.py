
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cupy
from typing import List
from utils import *
from .augmentation import *
from utils import *

class barlowDataset(Dataset):
    def __init__(self, args, images, labels, transforms_1 = None, transforms_2 = None, table_1 = None, table_2 = None, table_3 = None):
        self.args = args

        self.images = images.values
        self.labels = labels.values
        self.transforms_1 = transforms_1
        self.transforms_2 = transforms_2

        self.table_1 = table_1
        self.table_2 = table_2
        self.table_3 = table_3

    def __getitem__(self, idx):
        image_name = self.images[idx]
        img = Image.open(image_name)
        labels = self.labels[idx]
        table_label_0, table_label_1, table_label_2 = random.randrange(0, self.args.table[0]), random.randrange(0, self.args.table[1]), random.randrange(0, self.args.table[2])
        table_1, table_2, table_3 = self.table_1[table_label_0], self.table_2[table_label_1], self.table_3[table_label_2]
        if self.transforms_1 is not None and self.transforms_2 is not None:
            img_1 = self.transforms_1(img)   # origin
            img_2 = self.transforms_2(img)   # transform
            table_1_img, table_2_img, table_3_img = jigsaw_generator(img_1, table_1.type(torch.int), self.args.patches[0]), jigsaw_generator(img_1, table_2.type(torch.int), self.args.patches[1]), jigsaw_generator(img_1, table_3.type(torch.int), self.args.patches[2])

            return img_1, table_1_img, table_2_img, table_3_img, img_2
        else:
            img = self.transforms_1(img)
            return img, labels

    def __len__(self):
        return len(self.images)