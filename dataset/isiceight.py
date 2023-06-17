import numpy as np
import pandas as pd
from PIL import Image

import torch
import cupy
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import *
from .augmentation import *

class barlowDataset(Dataset):
    def __init__(self, args, images, labels, transforms_1 = None, transforms_2 = None):
        self.args = args

        self.images = images.values
        self.labels = labels.values
        self.transforms_1 = transforms_1
        self.transforms_2 = transforms_2

    def __getitem__(self, idx):
        image_name = self.images[idx]
        img = Image.open(image_name)
        labels = self.labels[idx]
        if self.transforms_1 is not None and self.transforms_2 is not None:
            img_1 = self.transforms_1(img)   # origin
            img_2 = self.transforms_2(img)   # transform
            table_1_img, table_2_img, table_3_img = jigsaw_generator(img_1, self.args.patches[0]), jigsaw_generator(img_1, self.args.patches[1]), jigsaw_generator(img_1, self.args.patches[2])
                # 2         # 4         # 8
            return  img_1, table_1_img,  table_2_img, table_3_img, img_2
        else:
            img = self.transforms_1(img)
            return img, labels

    def __len__(self):
        return len(self.images)


def ssl_isiceight_loader(args):
    """
    return traintabledataset and testdataset. 
    If you want change table, check for utils.get_table
    """
    train_df = pd.read_csv("dataset/data/ISIC2018/train.csv")
    train_df["image"] = "dataset/data/ISIC2018/ISIC2018_Task3_Training_Input/" + train_df["image"] + ".jpg"
    train_df["label"] = LabelEncoder().fit_transform(train_df["label"])
    train_x, test_x, train_y, test_y = train_test_split(train_df["image"], train_df["label"], test_size = args.split_size, random_state = 18)
    table_1, table_2, table_3 = torch.Tensor(np.load(f"dataset/tabledata/permutetable_{args.table[0]}_16.npy")), torch.Tensor(np.load(f"dataset/tabledata/permutetable_{args.table[1]}_8.npy")), torch.Tensor(np.load(f"dataset/tabledata/permutetable_{args.table[2]}_4.npy"))

    train_ds = customDataset(args, train_x, train_y, transforms = get_transform(args, train = True), table_1 = table_1, table_2 = table_2, table_3 = table_3)
    test_ds = customDataset(args, test_x, test_y, transforms = get_transform(args, train = False))

    ssl_train_dl = DataLoader(train_ds, batch_size = args.batchsize, pin_memory = True, shuffle = True)
    ssl_test_dl = DataLoader(test_ds, batch_size = args.batchsize, pin_memory = True, shuffle = True)

    return ssl_train_dl, ssl_test_dl
