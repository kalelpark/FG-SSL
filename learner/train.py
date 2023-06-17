import torch
from tqdm import tqdm
from utils import *


def barlow_train(args, model, unlabel_dl, optimizer, epoch):
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    losses = 0

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    optimizer.zero_grad()
    for img_1, table_1_img, table_2_img, table_3_img, img_2 in tqdm(unlabel_dl, desc = "barlowtwins train", position = 1, leave = False):
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])    
        img_1, table_1_img, table_2_img, table_3_img, img_2 = img_1.float().to(args.device), table_1_img.float().to(args.device), table_2_img.float().to(args.device), table_3_img.float().to(args.device), img_2.float().to(args.device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            _, _, _, _, x_ft_1, _, _, _ = model(table_1_img)
            _, _, _, _, y_ft_1, _, _, _ = model(img_2)
            barlow_loss_1 = barlow_criterion(x_ft_1, y_ft_1)
        scaler.scale(barlow_loss_1).backward()
        scaler.step(optimizer)
        scaler.update()
        losses += barlow_loss_1

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            _, _, _, _, _, x_ft_2, _, _ = model(table_2_img)
            _, _, _, _, _, y_ft_2, _, _ = model(img_2)
            barlow_loss_2 = barlow_criterion(x_ft_2, y_ft_2)
        scaler.scale(barlow_loss_2).backward()
        scaler.step(optimizer)
        scaler.update()
        losses += barlow_loss_2

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            _, _, _, _, _, _, x_ft_3, _ = model(table_3_img)
            _, _, _, _, _, _, y_ft_3, _ = model(img_2)
            barlow_loss_3 = barlow_criterion(x_ft_3, y_ft_3)
        scaler.scale(barlow_loss_3).backward()
        scaler.step(optimizer)
        scaler.update()
        losses += barlow_loss_3

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            _, _, _, _, _, _, _, x_ft_4 = model(img_1)
            _, _, _, _, _, _, _, y_ft_4 = model(img_2)
            barlow_loss_4 = barlow_criterion(x_ft_4, y_ft_4) * 2
        scaler.scale(barlow_loss_4).backward()
        scaler.step(optimizer)
        scaler.update()
        losses += barlow_loss_4
    
    return losses
