import os
import imp
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import random

def get_transform(args, train):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                     std=[0.229, 0.224, 0.225])

    if args.dataset in ["aptos", "APTOS2019", "aptos", "aptos2019", "APTOS"]:
        if train:           # Clear
            return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((args.resize, args.resize)),
            transforms.CenterCrop(args.centercrop),
            transforms.ToTensor(),
            normalize,
        ])
        else:
            return transforms.Compose([
            transforms.Resize((args.resize, args.resize)),
            transforms.CenterCrop(args.centercrop),
            transforms.ToTensor(),
            normalize,
        ])
        
    elif args.dataset in ["isic", "ISIC2017", "ISIC2018", "isic2018", "isic2017"]:
        if train:          
            return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((args.resize, args.resize)),
            transforms.RandomResizedCrop((224, 224)),
            # transforms.CenterCrop(args.centercrop),
            transforms.AugMix(severity= 3, mixture_width = 3, chain_depth = -1, alpha = 1.0, all_ops = True, interpolation = transforms.InterpolationMode.BILINEAR, fill = None),
            transforms.ToTensor(),
            normalize,
        ])
        else:
            return transforms.Compose([
            transforms.Resize((args.resize, args.resize)),
            transforms.CenterCrop(args.centercrop),
            transforms.ToTensor(),
            normalize
        ])

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def barlow_transform(args):
    if args.dataset in ["aptos", "APTOS2019", "aptos", "aptos2019", "APTOS"]:
        transform_1 = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

        transform_2 = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(p=0.1),
                    Solarization(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])
    else:
        transform_1 = transforms.Compose([
                    # transforms.Resize((224, 224)),  
                    # transforms.CenterCrop(args.centercrop),
                    transforms.Resize((224, 224)),
                    # transforms.CenterCrop(args.centercrop),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(p=1.0),
                    Solarization(p=0.0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

        transform_2 = transforms.Compose([
                    # transforms.Resize((224, 224)),  
                    # transforms.CenterCrop(args.centercrop),
                    transforms.Resize((224, 224)),
                    # transforms.CenterCrop(args.centercrop),
                    transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(p=0.1),
                    Solarization(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

    return transform_1, transform_2

def rotate_transform(args):
    rotate_0 = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5])])

    rotate_90 = transforms.Compose([
                transforms.RandomRotation(degrees= (90, 90)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])])

    rotate_180 = transforms.Compose([
                 transforms.RandomRotation(degrees= (180, 180)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5])])

    rotate_270 = transforms.Compose([
                 transforms.RandomRotation(degrees = (270, 270)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5])])

    return rotate_0, rotate_90, rotate_180, rotate_270



def moco_transform(args):
    train_transform = transforms.Compose([
                    transforms.Resize((args.resize, args.resize)),
                    transforms.CenterCrop(args.centercrop),
                    transforms.RandomApply([
                            transforms.ColorJitter(0.5, 0.5, 0.5)
                            ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    return train_transform