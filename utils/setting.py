import os
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from torchmetrics.classification import BinaryPrecision
from torchmetrics import Precision, AUROC
import torch.nn.functional as F

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch)) 
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)

def get_softmax():
    soft_change = nn.Softmax()
    return soft_change
    
def get_isicseven_data():
    train_df = pd.read_csv("dataset/data/ISIC-2017/ISIC-2017_Training_Part3_GroundTruth.csv")
    test_df = pd.read_csv("dataset/data/ISIC-2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv")

    train_images_dir = "dataset/data/ISIC-2017/ISIC-2017_Training_Data/"
    test_images_dir = "dataset/data/ISIC-2017/ISIC-2017_Test_v2_Data/"

    train_df["image_id"] = train_images_dir + train_df["image_id"] + ".jpg"
    test_df["image_id"] = test_images_dir + test_df["image_id"] + ".jpg"

    return train_df, test_df

def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    
    block_size = 224 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()

    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0 : block_size, 0 : block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size : (x+1) * block_size,
                                                    y*block_size : (y +1) * block_size].clone()
        jigsaws[..., x * block_size: (x + 1) * block_size, y * block_size : (y + 1) * block_size] = temp

    return jigsaws

def barlow_criterion(x, y, lmbd = 5e-3):
    bs = x.size(0)
    emb = x.size(1)

    xNorm = (x - x.mean(0)) / x.std(0)
    yNorm = (y - y.mean(0)) / y.std(0)
    crossCorMat = (xNorm.T@yNorm) / bs
    loss = (crossCorMat*lmbd - torch.eye(emb, device=torch.device('cuda'))*lmbd).pow(2)
    
    return loss.sum()

def exclude_gt(logit, target, is_log=False):
    logit = F.log_softmax(logit, dim=-1) if is_log else F.softmax(logit, dim=-1)
    mask = torch.ones_like(logit)
    for i in range(logit.size(0)): 
        mask[i, target[i].long()] = 0

    return mask*logit

def kl_loss(x_pred, x_gt, target):
    kl_gt = exclude_gt(x_gt, target, is_log=False)
    kl_pred = exclude_gt(x_pred, target, is_log=True)
    tmp_loss = F.kl_div(kl_pred, kl_gt, reduction='none')
    tmp_loss = torch.exp(-tmp_loss).mean()
    return tmp_loss

def example_aug():
    return transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ])    

def save_state_dict_model(args, model):
    torch.save(model.state_dict(), os.path.join(args.save_model, "model_state.pt"))