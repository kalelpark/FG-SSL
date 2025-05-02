from __future__ import print_function
import os
from PIL import Image

import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import *

def general_valid(args, valid_loader, model, loss_fn):
    model.eval()
    test_loss, total, correct = 0, 0, 0
    idx = 0
    for input, target in valid_loader:
        idx += 1
        input, target = input.float().to(args.device), target.type(torch.LongTensor).to(args.device)
        output = model(input)
        loss = loss_fn(output, target)
        test_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()

    test_acc_en = 100. * float(correct) / total
    test_loss = test_loss / (idx + 1)

    return test_acc_en, test_loss

def valid(args, valid_loader, model, loss_fn):

    model.eval()
    test_loss, total = 0, 0
    correct, correct_com = 0, 0
    idx = 0
    for input, target in valid_loader:
        idx += 1
        input, target = input.float().to(args.device), target.type(torch.LongTensor).to(args.device)
        output_1, output_2, output_3, output_concat= model(input)
        outputs_com = output_1 + output_2 + output_3 + output_concat

        loss = loss_fn(output_concat, target)
        test_loss += loss.item()
        _, predicted = torch.max(output_concat.data, 1)
        _, predicted_com = torch.max(outputs_com.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        correct_com += predicted_com.eq(target.data).cpu().sum()
    
    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss