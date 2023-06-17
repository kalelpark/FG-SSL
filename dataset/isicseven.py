import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import *
from .augmentation import *

def read_dataset():
    train_df = pd.read_csv("dataset/data/ISIC-2017/ISIC-2017_Training_Part3_GroundTruth.csv")
    valid_df = pd.read_csv("dataset/data/ISIC-2017/ISIC-2017_Validation_Part3_GroundTruth.csv")
    test_df = pd.read_csv("dataset/data/ISIC-2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv")

    train_images_dir = "dataset/data/ISIC-2017/ISIC-2017_Training_Data/"
    valid_images_dir = "dataset/data/ISIC-2017/ISIC-2017_Validation_Data/"
    test_images_dir = "dataset/data/ISIC-2017/ISIC-2017_Test_v2_Data/"

    train_df["image_id"] = train_images_dir + train_df["image_id"] + ".jpg"
    test_df["image_id"] = test_images_dir + test_df["image_id"] + ".jpg"

    return train_df, test_df

class customDataset(Dataset):
    def __init__(self, args, df, transforms, target = "seborrheic_keratosis",  table_1 = None, table_2 = None, table_3 = None):
        self.args = args

        self.images = df["image_id"].values
        self.labels = df[target].values
        self.transforms = transforms

        self.table_1 = table_1
        self.table_2 = table_2
        self.table_3 = table_3

    def __getitem__(self, idx):
        image_name = self.images[idx]
        img = Image.open(image_name)
        labels = self.labels[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        if self.table_1 is not None:
            table_label_0, table_label_1, table_label_2 = random.randrange(0, self.args.table[0]), random.randrange(0, self.args.table[1]), random.randrange(0, self.args.table[2])
            table_1, table_2, table_3 = self.table_1[table_label_0], self.table_2[table_label_1], self.table_3[table_label_2]
            return img, table_1, table_2, table_3, labels
        else:
            return img, labels
    
    def __len__(self):
        return len(self.images)

class barlowDataset(Dataset):
    def __init__(self, args, df, target = "seborrheic_keratosis", transforms_1 = None, transforms_2 = None, table_1 = None, table_2 = None, table_3 = None):
        self.args = args

        self.images = df["image_id"].values
        self.labels = df[target].values
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

def isicseven_loader(args):
    """
    return train table dataset and testdataset. 
    If you want change table, check for utils.get_table
    """
    train_df, test_df = read_dataset()
    table_1, table_2, table_3 = torch.Tensor(np.load(f"dataset/tabledata/permutetable_{args.table[0]}_8.npy")), torch.Tensor(np.load(f"dataset/tabledata/permutetable_{args.table[1]}_4.npy")), torch.Tensor(np.load(f"dataset/tabledata/permutetable_{args.table[2]}_2.npy"))
    test_ds = customDataset(args, df = test_df, transforms = get_transform(args, train = False))
    test_dl = DataLoader(test_ds, batch_size = args.batchsize, pin_memory = True, shuffle = True, num_workers = 4)

    train_ds = customDataset(args, df = train_df, transforms = get_transform(args, train = True), table_1 = table_1, table_2 = table_2, table_3 = table_3)
    label_train_dl = DataLoader(train_ds, batch_size = args.batchsize, pin_memory = True, shuffle = True, num_workers = 4)

    transform_1, transform_2 = barlow_transform(args)
    valid_ds = barlowDataset(args, df = train_df, transforms_1 = transform_1, transforms_2 = transform_2, table_1 = table_1, table_2 = table_2, table_3 = table_3)
    unlabel_dl = DataLoader(valid_ds, batch_size = args.batchsize, pin_memory = True, shuffle = True, num_workers = 4)
    
    return label_train_dl, unlabel_dl, test_dl