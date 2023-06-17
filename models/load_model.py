import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import models

from .fgvc.resnet import *
from .fgvc.pmg import *

def load_model(args, pretrained = True, require_grad = True):
    if args.model in ["resnet50", "resnet"]:
        model = models.resnet50( pretrained = pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_classes)
        return model
    if args.model in ["pmg", "resnetpmg"]:
        model = resnet50(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = require_grad
        model = PMG(args, model, args.featdim, args.num_classes)
        return model

def get_ce_optimizer(args, model):
    return optim.SGD([
        {'params' : model.module.classifier_concat.parameters(), 'lr' : 0.002},
        {'params' : model.module.classifier_concat_block.parameters(), 'lr' : 0.002},
            
        {'params' : model.module.conv_block1.parameters(), 'lr' : 0.002},
        {'params' : model.module.classifier_conv_block1.parameters(), 'lr' : 0.002},
        {'params' : model.module.classifier1.parameters(), 'lr' : 0.002},

        {'params' : model.module.conv_block2.parameters(), 'lr' : 0.002},
        {'params' : model.module.classifier_conv_block2.parameters(), 'lr' : 0.002},
        {'params' : model.module.classifier2.parameters(), 'lr' : 0.002},

        {'params' : model.module.conv_block3.parameters(), 'lr' : 0.002},
        {'params' : model.module.classifier_conv_block3.parameters(), 'lr' : 0.002},
        {'params' : model.module.classifier3.parameters(), 'lr' : 0.002},
        
        {'params' : model.module.features.parameters(), 'lr' : 0.0002},
    ],  momentum = args.momentum, weight_decay = args.weight_decay)

def get_barlow_optimizer(args, model):
    return optim.SGD([
        {'params' : model.module.classifier_concat_block.parameters(), 'lr' : 0.002},
            
        {'params' : model.module.conv_block1.parameters(), 'lr' : 0.002},
        {'params' : model.module.classifier_conv_block1.parameters(), 'lr' : 0.002},

        {'params' : model.module.conv_block2.parameters(), 'lr' : 0.002},
        {'params' : model.module.classifier_conv_block2.parameters(), 'lr' : 0.002},

        {'params' : model.module.conv_block3.parameters(), 'lr' : 0.002},
        {'params' : model.module.classifier_conv_block3.parameters(), 'lr' : 0.002},

        {'params' : model.module.features.parameters(), 'lr' : 0.0002},
    ],  momentum = args.momentum, weight_decay = args.weight_decay)