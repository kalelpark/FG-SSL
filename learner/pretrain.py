import torch
from tqdm import tqdm
from utils import *

# def fine_train(args, model, train_dl, optimizer, loss_fn, epoch):
#     lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
#     model.train()
#     scaler = torch.cuda.amp.GradScaler()
#     for idx, (image, label) in enumerate(tqdm(train_dl, desc = "Fine trainer", position = 1, leave = False)):
#         image, label = image.float().to(args.device), label.type(torch.LongTensor).to(args.device)

#         for nlr in range(len(optimizer.param_groups)):
#             optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])
        
#         optimizer.zero_grad()
#         with torch.cuda.amp.autocast():
#             inputs1 = jigsaw_generator(args, image, args.patches[0])        # Step1
#             xl1_fusion_cl, _, _, _ = model(inputs1)
#             # barlow_loss = barlow_criterion(feature, feature_4)
#             # print("barlow_loss 1", barlow_loss)
#             loss1 = loss_fn(xl1_fusion_cl, label) * 1
#             # loss1 = ((1 - args.gamma)*loss1) + (args.gamma*barlow_loss)
#             scaler.scale(loss1).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             optimizer.zero_grad()
#             inputs2 = jigsaw_generator(args, image, args.patches[1])        # Step2
#             _, xl2_fusion_cl, _, _ = model(inputs2)
#             # barlow_loss = barlow_criterion(feature, feature_4)
#             # barlow_loss = barlow_criterion(xct1, xct2) + barlow_criterion(xct3, x_concat_1)
#             # print("barlow_loss 2", barlow_loss)
#             loss2 = loss_fn(xl2_fusion_cl, label) * 1
#             # loss2 = ((1 - args.gamma)*loss2) + (args.gamma*barlow_loss)
#             scaler.scale(loss2).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             optimizer.zero_grad()
#             inputs3 = jigsaw_generator(args, image, args.patches[2])            # Step3
#             _, _, xl3_fusion_cl, _ = model(inputs3)
#             # barlow_loss = barlow_criterion(xct1, xct2) + barlow_criterion(xct3, x_concat_1)
#             # barlow_loss = barlow_criterion(feature, feature_4)
#             # print("barlow_loss 3", barlow_loss)
#             loss3 = loss_fn(xl3_fusion_cl, label) * 1
#             # loss3 = ((1 - args.gamma)*loss3) + (args.gamma*barlow_loss)
#             scaler.scale(loss3).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             optimizer.zero_grad()
#             _, _, _, xl4_fusion_cl = model(image)
#             # barlow_loss = (barlow_criterion(feature_1, feature_4) + barlow_criterion(feature_2, feature_4) + barlow_criterion(feature_3, feature_4)) / 3 
#             concat_loss = loss_fn(xl4_fusion_cl, label) * 2                 # Step4
#             # concat_loss = ((1 - args.gamma)*concat_loss) + (args.gamma*barlow_loss)
#             scaler.scale(concat_loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
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

def super_fine_train(args, model, train_dl, optimizer, loss_fn, epoch):
    # lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    losses = 0

    optimizer.zero_grad()
    for img, table_1, table_2, table_3, labels in tqdm(train_dl, desc = "Cut_Mix train", position = 1, leave = False):
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])
        img, labels = img.to(args.device), labels.to(args.device)

        lam = np.random.beta(args.beta, args.beta)
        rand_index = torch.randperm(img.size()[0]).to(args.device)
        target_a = labels
        target_b = labels[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
        img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]

        table_11_img, table_22_img, table_33_img = list(), list(), list()
        for idx in range(img.size()[0]):
            table_1_img, table_2_img, table_3_img = jigsaw_generator(img[idx, :, :, :], table_1[idx, :].type(torch.int), args.patches[0]), jigsaw_generator(img[idx, :, :, :], table_2[idx, :].type(torch.int), args.patches[1]), jigsaw_generator(img[idx, :, :, :], table_3[idx, :].type(torch.int), args.patches[2])
            table_11_img.append(table_1_img)
            table_22_img.append(table_2_img)
            table_33_img.append(table_3_img)

        table_11_img, table_22_img, table_33_img = torch.stack(table_11_img, 0), torch.stack(table_22_img, 0), torch.stack(table_33_img, 0)
        table_11_img, table_22_img, table_33_img = table_11_img.to(args.device), table_22_img.to(args.device), table_33_img.to(args.device)
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))

        with torch.cuda.amp.autocast(enabled=True):
            output_1, _, _, _, _, _, _, _, _, _, _ = model(table_11_img)
            loss = loss_fn(output_1, target_a) * lam + loss_fn(output_1, target_b) * (1. - lam)
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            _, output_2, _, _, _, _, _, _, _, _, _ = model(table_22_img)
            loss = loss_fn(output_2, target_a) * lam + loss_fn(output_2, target_b) * (1. - lam)
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            _, _, output_3, _, _, _, _, _, _, _, _ = model(table_33_img)
            loss = loss_fn(output_3, target_a) * lam + loss_fn(output_3, target_b) * (1. - lam)
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            _, _, _, output_4, _, _, _, _, _, _, _ = model(img)
            loss = (loss_fn(output_4, target_a) * lam + loss_fn(output_4, target_b) * (1. - lam)) * 2
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses
    

def barlow_jigsaw_train(args, model, train_dl, optimizer_ssl, loss_fn, epoch):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    ssl_loss = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    for img_1, table_1_img, table_2_img, table_3_img, img_2, labels, table_label_0, table_label_1, table_label_2 in tqdm(train_dl, desc = "barlow_jigsaw_train", position = 1, leave = False):
        img_1, table_1_img, table_2_img, table_3_img, img_2 = img_1.float().to(args.device), table_1_img.float().to(args.device), table_2_img.float().to(args.device), table_3_img.float().to(args.device), img_2.float().to(args.device)
        labels, table_label_0, table_label_1, table_label_2 = labels.type(torch.LongTensor).to(args.device), table_label_0.type(torch.LongTensor).to(args.device), table_label_1.type(torch.LongTensor).to(args.device), table_label_2.type(torch.LongTensor).to(args.device)

        for nlr in range(len(optimizer_ssl.param_groups)):
            optimizer_ssl.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])
        
        optimizer_ssl.zero_grad()
        with torch.cuda.amp.autocast():
            _, _, _, _, xt1, _, _, x1_feature_img_1, _, _, _ = model(table_1_img)
            _, _, _, _, _, _, _, x1_feature_img_2, _, _, _ = model(img_2)
            table_loss_1 = loss_fn(xt1, table_label_0) * 1
            barlow_loss = barlow_criterion(x1_feature_img_1, x1_feature_img_2)
            loss_1 = (args.gamma*table_loss_1) + ((1 - args.gamma)*barlow_loss)
            scaler.scale(loss_1).backward()
            scaler.step(optimizer_ssl)
            scaler.update()
            ssl_loss += loss_1         

            _, _, _, _, _, xt2, _, _, x2_feature_img_1, _, _ = model(table_2_img)
            _, _, _, _, _, _, _, _, x2_feature_img_2, _, _ = model(img_2)
            table_loss_1 = loss_fn(xt2, table_label_1) * 1
            barlow_loss = barlow_criterion(x2_feature_img_1, x2_feature_img_2)
            loss_1 = (args.gamma*table_loss_1) + ((1 - args.gamma)*barlow_loss)
            scaler.scale(loss_1).backward()
            scaler.step(optimizer_ssl)
            scaler.update()
            ssl_loss += loss_1        

            _, _, _, _, _, _, xt3, _, _, x3_feature_img_1, _ = model(table_3_img)
            _, _, _, _, _, _, _, _, _, x3_feature_img_2, _ = model(img_2)
            table_loss_1 = loss_fn(xt3, table_label_2) * 1
            barlow_loss = barlow_criterion(x3_feature_img_1, x3_feature_img_2)
            loss_1 = (args.gamma*table_loss_1) + ((1 - args.gamma)*barlow_loss)
            scaler.scale(loss_1).backward()
            scaler.step(optimizer_ssl)
            scaler.update()
            ssl_loss += loss_1        

            _, _, _, _, _, _, _, _, _, _, x_concat_feature_img_1 = model(img_1)
            _, _, _, _, _, _, _, _, _, _, x_concat_feature_img_2 = model(img_2)
            barlow_loss = barlow_criterion(x_concat_feature_img_1, x_concat_feature_img_2)
            scaler.scale(barlow_loss * 2).backward()
            scaler.step(optimizer_ssl)
            scaler.update()
            ssl_loss += barlow_loss   

    return ssl_loss     

def resnet_barlow_train(args, model, unlabel_dl, optimizer):
    model.train()
    losses = 0
    scaler = torch.cuda.amp.GradScaler(enabled = True)

    for img_1, img_2 in tqdm(unlabel_dl, desc = "resnet_barlow_train", position = 1, leave = False):
        img_1, img_2 = img_1.to(args.device), img_2.to(args.device)
        img_1 = jigsaw_generator(img_1, 8)
        with torch.cuda.amp.autocast(enabled = True):
            v_1 = model(img_1)
            v_2 = model(img_2)
            barlow_loss = barlow_criterion(v_1, v_2)
            scaler.scale(barlow_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        losses += barlow_loss
    return losses

def fine_train(args, model, train_dl, optimizer_1, optimizer_2, loss_fn, epoch):
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    lr_1 = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]

    model.train()
    scaler = torch.cuda.amp.GradScaler()
    
    for img, table_1_img, table_2_img, table_3_img, labels, table_label_0 in tqdm(train_dl, desc = "trainer", position = 1, leave = False):

        img, table_1_img, table_2_img, table_3_img = img.float().to(args.device), table_1_img.float().to(args.device), table_2_img.float().to(args.device), table_3_img.float().to(args.device)
        labels, table_label_0 =labels.type(torch.LongTensor).to(args.device), table_label_0.type(torch.LongTensor).to(args.device)
                    
        for nlr in range(len(optimizer_1.param_groups)):
            optimizer_1.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])
        
        for nlr in range(len(optimizer_2.param_groups)):
            optimizer_2.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr_1[nlr])
        
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        with torch.cuda.amp.autocast():
            xc1, _, _, _, table_concat = model(table_1_img)     # patches[0]
            loss_xc1 = loss_fn(xc1, labels) * 1
            loss_xt1 = loss_fn(table_concat, table_label_0)
            loss_1 = ((1 - args.gamma)*loss_xc1) + (args.gamma*loss_xt1)

            scaler.scale(loss_1).backward()
            scaler.step(optimizer_1)
            scaler.update()

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            _, xc2, _, _, _ = model(table_2_img)      # patches[1]
            loss_xc2 = loss_fn(xc2, labels) * 1

            scaler.scale(loss_xc2).backward()
            scaler.step(optimizer_2)
            scaler.update()

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            _, _, xc3, _, _ = model(table_3_img)      # patches[2]
            loss_xc3 = loss_fn(xc3, labels) * 1

            scaler.scale(loss_xc3).backward()
            scaler.step(optimizer_2)
            scaler.update()

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            _, _, _, xc4, _ = model(img)              # patches X
            concat_loss = loss_fn(xc4, labels) * 2
            scaler.scale(concat_loss).backward()
            scaler.step(optimizer_2)
            scaler.update()

# def table_train(args, model, train_dl, optimizer_1, loss_fn, epoch):
#     lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
#     lossss = 0
#     model.train()
#     scaler = torch.cuda.amp.GradScaler()
#     for img, table_1_img, table_2_img, table_3_img, labels, table_label_0 in tqdm(train_dl, desc = "table train", position = 1, leave = False):

#         img, table_1_img, table_2_img, table_3_img = img.float().to(args.device), table_1_img.float().to(args.device), table_2_img.float().to(args.device), table_3_img.float().to(args.device)
#         labels, table_label_0 = labels.type(torch.LongTensor).to(args.device), table_label_0.type(torch.LongTensor).to(args.device)

#         for nlr in range(len(optimizer_1.param_groups)):
#             optimizer_1.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])
        
#         optimizer_1.zero_grad()
#         with torch.cuda.amp.autocast():
#             _, _, _, _, table_concat = model(table_1_img)
#             table_loss_1 = loss_fn(table_concat, table_label_0)
#             scaler.scale(table_loss_1).backward()
#             scaler.step(optimizer_1)
#             scaler.update()

#         lossss += table_loss_1.item()
#     return lossss

# python main.py --seed 0 --model pmg --patches 8 4 2 --table 128 64 24 --dataset aptos --imgsize 550 --crop 448 --gpu_ids 5

def twin_jigsaw_train(args, model, train_dl, optimizer, loss_fn, epoch):
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    lossss = 0
    model.train()

    scaler = torch.cuda.amp.GradScaler()
    for img_1, table_1_img, table_2_img, table_3_img, img_2, table_label_0, table_label_1, table_label_2 in tqdm(train_dl, desc = "twin_jigsaw_train", position = 1, leave = False):
        img_1, table_1_img, table_2_img, table_3_img = img_1.float().to(args.device), table_1_img.float().to(args.device), table_2_img.float().to(args.device), table_3_img.float().to(args.device)
        img_2, table_label_0, table_label_1, table_label_2 = img_2.float().to(args.device), table_label_0.type(torch.LongTensor).to(args.device), table_label_1.type(torch.LongTensor).to(args.device), table_label_2.type(torch.LongTensor).to(args.device)
        
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            # xc1, xc2, xc3, x_concat, xt1, xt2, xt3, xc_1_ft, xc_2_ft, xc_3_ft, x_concat_ft
            _, _, _, _, xt1, _, _, xc_1_ft, _, _, _ = model(table_1_img)
            _, _, _, _, _, _, _, xc_1_ft2, _, _, _ = model(img_2)

            table_loss_1 = loss_fn(xt1, table_label_0)
            barlow_loss_1 = barlow_criterion(xc_1_ft, xc_1_ft2)
            concat_loss_1 = table_loss_1 + barlow_loss_1
            scaler.scale(concat_loss_1).backward()
            scaler.step(optimizer)
            scaler.update()
            lossss += concat_loss_1
            
            optimizer.zero_grad()
            _, _, _, _, _, xt2, _, _, xc_1_ft, _, _ = model(table_2_img)
            _, _, _, _, _, _, _, _, xc_1_ft2, _, _ = model(img_2)
            table_loss_2 = loss_fn(xt2, table_label_1)
            barlow_loss_2 = barlow_criterion(xc_1_ft, xc_1_ft2)
            concat_loss_2 = table_loss_2 + barlow_loss_2
            scaler.scale(concat_loss_2).backward()
            scaler.step(optimizer)
            scaler.update()
            lossss += concat_loss_2
            
            optimizer.zero_grad()
            _, _, _, _, _, _, xt3, _, _, xc_1_ft, _ = model(table_3_img)
            _, _, _, _, _, _, _, _, _, xc_1_ft2,  _ = model(img_2)
            table_loss_3 = loss_fn(xt3, table_label_2)
            barlow_loss_3 = barlow_criterion(xc_1_ft, xc_1_ft2)
            concat_loss_3 = table_loss_3 + barlow_loss_3
            scaler.scale(concat_loss_3).backward()
            scaler.step(optimizer)
            scaler.update()
            lossss += concat_loss_3

            optimizer.zero_grad()
            _, _, _, _, _, _, _, _, _, _, xc_1_ft = model(img_1)
            _, _, _, _, _, _, _, _, _, _, xc_1_ft2 = model(img_2)
            barlow_loss_4 = barlow_criterion(xc_1_ft, xc_1_ft2) * 2
            scaler.scale(barlow_loss_4).backward()
            scaler.step(optimizer)
            scaler.update()
            lossss += barlow_loss_4

    return lossss

def ce_train(args, model, train_dl, optimizer_2, loss_fn, epoch):
    lr_1 = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]

    model.train()
    scaler = torch.cuda.amp.GradScaler()
    
    for img, table_1_img, table_2_img, table_3_img, labels, table_label_0 in tqdm(train_dl, desc = "trainer", position = 1, leave = False):
        img, table_1_img, table_2_img, table_3_img = img.float().to(args.device), table_1_img.float().to(args.device), table_2_img.float().to(args.device), table_3_img.float().to(args.device)
        labels, table_label_0 =labels.type(torch.LongTensor).to(args.device), table_label_0.type(torch.LongTensor).to(args.device)
                
        for nlr in range(len(optimizer_2.param_groups)):
            optimizer_2.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr_1[nlr])
        
        optimizer_2.zero_grad()
        with torch.cuda.amp.autocast():
            xc1, _, _, _, _ = model(table_1_img)     # patches[0]
            loss_1 = loss_fn(xc1, labels) * 1
            scaler.scale(loss_1).backward()
            scaler.step(optimizer_2)
            scaler.update()

            optimizer_2.zero_grad()
            _, xc2, _, _, _ = model(table_2_img)      # patches[1]
            loss_2 = loss_fn(xc2, labels) * 1
            scaler.scale(loss_2).backward()
            scaler.step(optimizer_2)
            scaler.update()

            optimizer_2.zero_grad()
            _, _, xc3, _, _ = model(table_3_img)      # patches[2]
            loss_3 = loss_fn(xc3, labels) * 1

            scaler.scale(loss_3).backward()
            scaler.step(optimizer_2)
            scaler.update()

            optimizer_2.zero_grad()
            _, _, _, xc4, _ = model(img)              # patches X
            loss_4 = loss_fn(xc4, labels) * 2
            scaler.scale(loss_4).backward()
            scaler.step(optimizer_2)
            scaler.update()

def barlow_twins_train(args, model, train_dl, optimizer, epoch):
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    lossss = 0
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    for img_1, img_2, _ in tqdm(train_dl, desc = "barlow train", position = 1, leave = False):
        img_1, img_2 = img_1.float().to(args.device), img_2.float().to(args.device)
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            _, _, _, _, feature_1 = model(img_1)
            _, _, _, _, feature_2 = model(img_2)
            barlow_loss = barlow_criterion(feature_1, feature_2)
            scaler.scale(barlow_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        lossss += barlow_loss.item()
    return lossss

def origin_train(args, model, train_dl, optimizer_2, loss_fn, epoch):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    lr_1 = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]

    for img, labels in tqdm(train_dl, desc = "origin train", position = 1, leave = False):
        img, labels = img.float().to(args.device), labels.type(torch.LongTensor).to(args.device)

        for nlr in range(len(optimizer_2.param_groups)):
            optimizer_2.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr_1[nlr])
        
        optimizer_2.zero_grad()
        with torch.cuda.amp.autocast():
            _, _, _, xc4, _ = model(img)           
            loss_4 = loss_fn(xc4, labels) * 2
            scaler.scale(loss_4).backward()
            scaler.step(optimizer_2)
            scaler.update()
        
def rotate_train(args, model, train_dl, optimizer_1, loss_fn, epoch):
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    lossss = 0
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    for img, labels in tqdm(train_dl, desc = "table train", position = 1, leave = False):

        img, labels = img.float().to(args.device), labels.type(torch.LongTensor).to(args.device)

        for nlr in range(len(optimizer_1.param_groups)):
            optimizer_1.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])
        
        optimizer_1.zero_grad()
        with torch.cuda.amp.autocast():
            _, _, _, _, table_concat = model(img)
            table_loss_1 = loss_fn(table_concat, labels)
            scaler.scale(table_loss_1).backward()
            scaler.step(optimizer_1)
            scaler.update()

        lossss += table_loss_1.item()
    return lossss

def table_ssl_train(args, model, train_dl, optimizer, loss_fn, epoch):
    model.train()
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    losses = 0
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    for table_1_img, table_2_img, table_3_img, table_label_0, table_label_1, table_label_2 in tqdm(train_dl, desc = "table_ssl_train", position = 1, leave = False):
        table_1_img, table_2_img, table_3_img = table_1_img.float().to(args.device), table_2_img.float().to(args.device), table_3_img.float().to(args.device)
        table_label_0, table_label_1, table_label_2 = table_label_0.type(torch.LongTensor).to(args.device), table_label_1.type(torch.LongTensor).to(args.device), table_label_2.type(torch.LongTensor).to(args.device)
        
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            _, _, _, _, xt1, _, _ =  model(table_1_img)     # patches[0]
            loss_xc1 = loss_fn(xt1, table_label_0) * 1
        losses += loss_xc1
        scaler.scale(loss_xc1).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            _, _, _, _, _, xt2, _ = model(table_2_img)     # patches[0]
            loss_xc2 = loss_fn(xt2, table_label_1) * 1
        losses += loss_xc2
        scaler.scale(loss_xc2).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            _, _, _, _, _, _, xt3 = model(table_3_img)     # patches[0]
            loss_xc3 = loss_fn(xt3, table_label_2) * 1
        losses += loss_xc3
        scaler.scale(loss_xc3).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses
    

# def simclr_train(args, model, train_dl, optimizer, epoch):
#     lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
#     losses = 0
#     model.train()
#     sim_fn = SimCLR_Loss(args.batchsize, temperature = 0.5)
#     scaler = torch.cuda.amp.GradScaler()
#     for img_1, img_2, labels in tqdm(train_dl, desc = "simclr train", position = 1, leave = False):
#         img_1, img_2, labels = img_1.float().to(args.device), img_2.float().to(args.device), labels.type(torch.LongTensor).to(args.device)
#         for nlr in range(len(optimizer.param_groups)):
#             optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])
        
#         optimizer.zero_grad()
#         with torch.cuda.amp.autocast():
#             _, _, _, _, feature_1 = model(img_1)
#             _, _, _, _, feature_2 = model(img_2)
#             sim_loss = sim_fn(feature_1, feature_2)
#             scaler.scale(sim_loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#         losses += sim_loss.item()
#     return losses

def moco_init_train(args, model, ss_valid):
    # k 65536 # feature dim 128 # 1000      # 77*32 2468 # feature dim 128 # 3000
    queue = None
    negative_num = 3000
    flag = 0

    if queue is None:
        while True:
            with torch.no_grad():
                for img_1, img_2, _ in tqdm(ss_valid, desc = "moco init train", position = 1, leave = False):
                    img_1, img_2 = img_1.float().to(args.device), img_2.float().to(args.device)
                    _, _, _, _, k = model(img_2)
                    if queue is None:
                        queue = k
                    else:
                        if queue.shape[0] < negative_num:
                            queue = torch.cat((queue,k),0)
                        else:
                            flag = 1 # stop filling the queue
                    if flag == 1:
                        break 
            if flag == 1:
                break
    queue = queue[:negative_num]

    return queue 
    
def moco_train(args, model, model_encoder, train_dl, optimizer, epoch, queue):
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    lossss = 0
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    
    for img_1, img_2, _ in tqdm(train_dl, desc = "moco_trainer", position = 1, leave = False):
        img_1, img_2 = img_1.float().to(args.device), img_2.float().to(args.device)
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            _, _, _, _, feature_1 = model(img_1)
            _, _, _, _, feature_2 = model_encoder(img_2)
            feature_2 = feature_2.detach()

            feature_1 = torch.div(feature_1, torch.norm(feature_1,dim=1).reshape(-1,1))
            feature_2 = torch.div(feature_2, torch.norm(feature_2,dim=1).reshape(-1,1))
            loss = mocov2_loss(feature_1, feature_2, queue)
            lossss += loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update the queue
            queue = torch.cat((queue, feature_2), 0)

            if queue.shape[0] > 3000:
                queue = queue[32:,:]

            for q_params, k_params in zip(model.parameters(), model_encoder.parameters()):
                k_params.data.copy_(0.9*k_params + q_params*(1.0-0.9))

    return lossss
    # return queue

def jigsaw_train(args, model, train_dl, optimizer, loss_fn, epoch):
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    losses = 0
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    for table_1_img, table_2_img, table_3_img, table_label_0, table_label_1, table_label_2 in tqdm(train_dl, desc = "ce_ssl_train", position = 1, leave = False):
        table_1_img, table_2_img, table_3_img = table_1_img.float().to(args.device), table_2_img.float().to(args.device), table_3_img.float().to(args.device)
        table_label_0, table_label_1, table_label_2 = table_label_0.type(torch.LongTensor).to(args.device), table_label_1.type(torch.LongTensor).to(args.device), table_label_2.type(torch.LongTensor).to(args.device)

        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            _, _, _, _, xt1, _, _ = model(table_1_img)     # patches[0]
            loss_1 = loss_fn(xt1, table_label_0) * 1
        losses += loss_1
        scaler.scale(loss_1).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            _, _, _, _, _, xt2, _ = model(table_2_img)      # patches[1]
            loss_2 = loss_fn(xt2, table_label_1) * 1
        losses += loss_2
        scaler.scale(loss_2).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            _, _, _, _, _, _, xt3 = model(table_3_img)      # patches[2]
            loss_3 = loss_fn(xt3, table_label_2) * 1
        losses += loss_3
        scaler.scale(loss_3).backward()
        scaler.step(optimizer)
        scaler.update()

# def pirl_train(args, model, train_dl, optimizer, epoch, queue):
#     lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
#     lossss = 0
#     model.train()
#     scaler = torch.cuda.amp.GradScaler()
    
#     for img_1, img_2, _ in tqdm(train_dl, desc = "moco_trainer", position = 1, leave = False):
#         img_1, img_2 = img_1.float().to(args.device), img_2.float().to(args.device)
#         for nlr in range(len(optimizer.param_groups)):
#             optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])
        
#         optimizer.zero_grad()
#         with torch.cuda.amp.autocast():
#             _, _, _, _, feature_1 = model(img_1)
#             _, _, _, _, feature_2 = model_encoder(img_2)
#             feature_2 = feature_2.detach()

#             feature_1 = torch.div(feature_1, torch.norm(feature_1,dim=1).reshape(-1,1))
#             feature_2 = torch.div(feature_2, torch.norm(feature_2,dim=1).reshape(-1,1))
#             loss = mocov2_loss(feature_1, feature_2, queue)
#             lossss += loss
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             # update the queue
#             queue = torch.cat((queue, feature_2), 0)

#             if queue.shape[0] > 3000:
#                 queue = queue[32:,:]

#             for q_params, k_params in zip(model.parameters(), model_encoder.parameters()):
#                 k_params.data.copy_(0.9*k_params + q_params*(1.0-0.9))

#     return lossss

def barlow_jigsaw_train(args, model, train_dl, optimizer_ssl, loss_fn, epoch):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    ssl_loss = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    for img_1, table_1_img, table_2_img, table_3_img, img_2, labels, table_label_0, table_label_1, table_label_2 in tqdm(train_dl, desc = "barlow_jigsaw_train", position = 1, leave = False):
        img_1, table_1_img, table_2_img, table_3_img, img_2 = img_1.float().to(args.device), table_1_img.float().to(args.device), table_2_img.float().to(args.device), table_3_img.float().to(args.device), img_2.float().to(args.device)
        labels, table_label_0, table_label_1, table_label_2 = labels.type(torch.LongTensor).to(args.device), table_label_0.type(torch.LongTensor).to(args.device), table_label_1.type(torch.LongTensor).to(args.device), table_label_2.type(torch.LongTensor).to(args.device)

        for nlr in range(len(optimizer_ssl.param_groups)):
            optimizer_ssl.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])
        
        optimizer_ssl.zero_grad()
        with torch.cuda.amp.autocast(True):
            _, _, _, _, xt1, _, _, x1_feature_img_1, _, _, _ = model(table_1_img)
            _, _, _, _, _, _, _, x1_feature_img_2, _, _, _ = model(img_2)
            table_loss_1 = loss_fn(xt1, table_label_0) * 1
            barlow_loss = barlow_criterion(x1_feature_img_1, x1_feature_img_2)
            loss_1 = (args.gamma*table_loss_1) + ((1 - args.gamma)*barlow_loss)
            scaler.scale(loss_1).backward()
            scaler.step(optimizer_ssl)
            scaler.update()
            ssl_loss += loss_1         

            _, _, _, _, _, xt2, _, _, x2_feature_img_1, _, _ = model(table_2_img)
            _, _, _, _, _, _, _, _, x2_feature_img_2, _, _ = model(img_2)
            table_loss_1 = loss_fn(xt2, table_label_1) * 1
            barlow_loss = barlow_criterion(x2_feature_img_1, x2_feature_img_2)
            loss_1 = (args.gamma*table_loss_1) + ((1 - args.gamma)*barlow_loss)
            scaler.scale(loss_1).backward()
            scaler.step(optimizer_ssl)
            scaler.update()
            ssl_loss += loss_1        

            _, _, _, _, _, _, xt3, _, _, x3_feature_img_1, _ = model(table_3_img)
            _, _, _, _, _, _, _, _, _, x3_feature_img_2, _ = model(img_2)
            table_loss_1 = loss_fn(xt3, table_label_2) * 1
            barlow_loss = barlow_criterion(x3_feature_img_1, x3_feature_img_2)
            loss_1 = (args.gamma*table_loss_1) + ((1 - args.gamma)*barlow_loss)
            scaler.scale(loss_1).backward()
            scaler.step(optimizer_ssl)
            scaler.update()
            ssl_loss += loss_1        

            _, _, _, _, _, _, _, _, _, _, x_concat_feature_img_1 = model(img_1)
            _, _, _, _, _, _, _, _, _, _, x_concat_feature_img_2 = model(img_2)
            barlow_loss = barlow_criterion(x_concat_feature_img_1, x_concat_feature_img_2)
            scaler.scale(barlow_loss * 2).backward()
            scaler.step(optimizer_ssl)
            scaler.update()
            ssl_loss += barlow_loss   

    return ssl_loss  

def base_train(args, model, train_dl, optimizer, loss_fn):
    model.train()
    losses = 0
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    for idx, (image, label) in enumerate(tqdm(train_dl, desc = "Base trainer", position = 1, leave = False)):
        image, label = image.float().to(args.device), label.type(torch.LongTensor).to(args.device)
        optimizer.zero_grad()
        # image = jigsaw_generator(image, 8)
        with torch.cuda.amp.autocast(enabled = True):
            proba = model(image)
            loss = loss_fn(proba, label)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses += loss
    
    return losses
    

def cutmix_trainer(args, model, train_dl, optimizer, loss_fn, epoch):
    model.train()
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    losses = 0
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    for img, table_1, table_2, table_3, labels in tqdm(train_dl, desc = "cutmix_trainer", position = 1, leave = False):
        img, labels = img.to(args.device), labels.to(args.device)
        
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])


        r = 2
        if r > args.cut_prob:
            pass
        else:
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(img.size()[0]).to(args.device)
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
        
        table_11_img, table_22_img, table_33_img = list(), list(), list()
        for idx in range(img.size()[0]):
            table_1_img, table_2_img, table_3_img = jigsaw_generator(img[idx, :, :, :], table_1[idx, :].type(torch.int), args.patches[0]), jigsaw_generator(img[idx, :, :, :], table_2[idx, :].type(torch.int), args.patches[1]), jigsaw_generator(img[idx, :, :, :], table_3[idx, :].type(torch.int), args.patches[2])
            table_11_img.append(table_1_img)
            table_22_img.append(table_2_img)
            table_33_img.append(table_3_img)

        table_11_img, table_22_img, table_33_img = torch.stack(table_11_img, 0), torch.stack(table_22_img, 0), torch.stack(table_33_img, 0)
        table_11_img, table_22_img, table_33_img = table_11_img.to(args.device), table_22_img.to(args.device), table_33_img.to(args.device)
        

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            output_1, _, _, _, _, _, _ =  model(table_11_img)     # patches[0]
            if r > args.cut_prob:
                loss = loss_fn(output_1, labels)
            else:
                loss = loss_fn(output_1, target_a) * lam + loss_fn(output_1, target_b) * (1. - lam)
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            _, output_2, _, _, _, _, _ = model(table_22_img)     # patches[0]
            if r > args.cut_prob:
                loss = loss_fn(output_2, labels)
            else:
                loss = loss_fn(output_2, target_a) * lam + loss_fn(output_2, target_b) * (1. - lam)
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            _, _, output_3, _, _, _, _ = model(table_33_img)     # patches[0]
            if r > args.cut_prob:
                loss = loss_fn(output_3, labels)
            else:
                loss = loss_fn(output_3, target_a) * lam + loss_fn(output_3, target_b) * (1. - lam)
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            _, _, _, output_4, _, _, _ = model(img)     # patches[0]
            if r > args.cut_prob:
                loss = loss_fn(output_4, labels) * 2 
            else:
                loss = (loss_fn(output_4, target_a) * lam + loss_fn(output_4, target_b) * (1. - lam)) * 2
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses

def moco_trainer(args, model, train_dl, model_optimizer, loss_fn):
    model.train()
    training_losses = 0
    for img, labels in tqdm(train_dl, desc = "origin_fine_train", position = 1, leave = False):
        img, labels = img.to(args.device), labels.to(args.device)

        model_optimizer.zero_grad()

        output_encoder = model(img)
        loss = loss_fn(output_encoder, labels)
        training_losses += loss
        loss.backward()
        model_optimizer.step()
        
    return training_losses

def simsiam_train(args, model, unlabel_dl, optimizer, loss_fn):
    model.train()
    losses = 0
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    if args.model in ["pmg", "resnetpmg"]:
        optimizer.zero_grad()
        for img_1, img_2 in tqdm(unlabel_dl, desc = "simsiam_train", position = 1, leave = False):
            img_1, img_2 = img_1.to(args.device), img_2.to(args.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled = True):
                # jigsaw_img = jigsaw_generator(img_1, 8)
                _, _, _, _, z1, _, _, _  = model(img_1)
                _, _, _, _, y1, _, _, _  = model(img_2)
                loss_1 = simsiam_loss(z1, y1)
            scaler.scale(loss_1).backward()
            scaler.step(optimizer)
            scaler.update()
            losses += loss_1

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled = True):
                # jigsaw_img = jigsaw_generator(img_1, 4)
                _, _, _, _, _, z1, _, _  = model(img_1)
                _, _, _, _, _, y1, _, _  = model(img_2)
                loss_1 = simsiam_loss(z1, y1)
            scaler.scale(loss_1).backward()
            scaler.step(optimizer)
            scaler.update()
            losses += loss_1

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled = True):
                # jigsaw_img = jigsaw_generator(img_1, 2)
                _, _, _, _, _, _, z1, _  = model(img_1)
                _, _, _, _, _, _, y1, _  = model(img_2)
                loss_1 = simsiam_loss(z1, y1)
            scaler.scale(loss_1).backward()
            scaler.step(optimizer)
            scaler.update()
            losses += loss_1

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled = True):
                _, _, _, _, _, _, _, z1  = model(img_1)
                _, _, _, _, _, _, _, y1  = model(img_2)
                loss_1 = simsiam_loss(z1, y1)
            scaler.scale(loss_1).backward()
            scaler.step(optimizer)
            scaler.update()
            losses += loss_1
        
        return losses
    else:
        for img_1, img_2 in tqdm(unlabel_dl, desc = "simsiam_train", position = 1, leave = False):
            img_1, img_2 = img_1.to(args.device), img_2.to(args.device)
            img_1 = jigsaw_generator(img_1, 8)
            with torch.cuda.amp.autocast(enabled = True):
                p1, p2, z1, z2 = model(img_1, img_2)
                loss = -(loss_fn(p1, z2).mean() + loss_fn(p2, z1).mean()) * 0.5
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses += loss
        
        return losses
        
    

def origin_fine_train(args, model, train_dl, optimizer, loss_fn, epoch):

    model.train()
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    losses = 0
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    for img, table_1, table_2, table_3, labels in tqdm(train_dl, desc = "origin_fine_train", position = 1, leave = False):
        if args.dataset in ["isic2017"]:
            img, labels = img.to(args.device), labels.type(torch.LongTensor).to(args.device)  # if you select isisc 
        else:
            img, labels = img.to(args.device), labels.to(args.device)
        
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])


        r = np.random.rand(1)
        if r > args.cut_prob:
            pass
        else:
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(img.size()[0]).to(args.device)
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
        
        table_11_img, table_22_img, table_33_img = list(), list(), list()
        for idx in range(img.size()[0]):
            table_1_img, table_2_img, table_3_img = jigsaw_generator(img[idx, :, :, :], table_1[idx, :].type(torch.int), args.patches[0]), jigsaw_generator(img[idx, :, :, :], table_2[idx, :].type(torch.int), args.patches[1]), jigsaw_generator(img[idx, :, :, :], table_3[idx, :].type(torch.int), args.patches[2])
            table_11_img.append(table_1_img)
            table_22_img.append(table_2_img)
            table_33_img.append(table_3_img)

        table_11_img, table_22_img, table_33_img = torch.stack(table_11_img, 0), torch.stack(table_22_img, 0), torch.stack(table_33_img, 0)
        table_11_img, table_22_img, table_33_img = table_11_img.to(args.device), table_22_img.to(args.device), table_33_img.to(args.device)
        

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            output_1, _, _, _, _, _, _, _ =  model(table_11_img)     # patches[0]
            if r > args.cut_prob:
                loss = loss_fn(output_1, labels)
            else:
                loss = loss_fn(output_1, target_a) * lam + loss_fn(output_1, target_b) * (1. - lam)
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            _, output_2, _, _, _, _, _, _ = model(table_22_img)     # patches[0]
            if r > args.cut_prob:
                loss = loss_fn(output_2, labels)
            else:
                loss = loss_fn(output_2, target_a) * lam + loss_fn(output_2, target_b) * (1. - lam)
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            _, _, output_3, _, _, _, _, _ = model(table_33_img)     # patches[0]
            if r > args.cut_prob:
                loss = loss_fn(output_3, labels)
            else:
                loss = loss_fn(output_3, target_a) * lam + loss_fn(output_3, target_b) * (1. - lam)
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            _, _, _, output_4, _, _, _, _ = model(img)     # patches[0]
            if r > args.cut_prob:
                loss = loss_fn(output_4, labels) * 2 
            else:
                loss = (loss_fn(output_4, target_a) * lam + loss_fn(output_4, target_b) * (1. - lam)) * 2
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses

def double_fine_train(args, model, train_dl, optimizer, loss_fn, epoch):
    model.train()
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    losses = 0
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    for img, labels in tqdm(train_dl, desc = "origin_fine_train", position = 1, leave = False):
        if args.dataset in ["isic2017"]:
            img, labels = img.to(args.device), labels.type(torch.LongTensor).to(args.device)  # if you select isisc 
        else:
            img, labels = img.to(args.device), labels.to(args.device)
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])

        

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            jigsaw_img = jigsaw_generator(img, 8)
            xc, _, _, _, _, _, _, _  = model(jigsaw_img)
            loss = loss_fn(xc, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses += loss

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            jigsaw_img = jigsaw_generator(img, 4)
            _, xc, _, _, _, _, _, _ = model(jigsaw_img)
            loss = loss_fn(xc, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses += loss

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            jigsaw_img = jigsaw_generator(img, 2)
            _, _, xc, _, _, _, _, _ = model(jigsaw_img)
            loss = loss_fn(xc, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses += loss

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            _, _, _, xc, _, _, _, _ = model(img)
            loss = loss_fn(xc, labels) * 2
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses += loss

    return losses

def rotation_train(args, model, unlabel_dl, optimizer, loss_fn):
    model.train()
    losses = 0
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    if args.model in ["pmg", "resnetpmg"]:
        optimizer.zero_grad()
        for img_1, labels in tqdm(unlabel_dl, desc = "rotation train", position = 1, leave = False):
            img_1, labels = img_1.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled = True):
                xc1, _, _, _, _, _, _, _  = model(img_1)
                loss_1 = loss_fn(xc1, labels)
            scaler.scale(loss_1).backward()
            scaler.step(optimizer)
            scaler.update()
            losses += loss_1

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled = True):
                _, xc1, _, _, _, _, _, _  = model(img_1)
                loss_1 = loss_fn(xc1, labels)
            scaler.scale(loss_1).backward()
            scaler.step(optimizer)
            scaler.update()
            losses += loss_1

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled = True):
                _, _, xc1, _, _, _, _, _  = model(img_1)
                loss_1 = loss_fn(xc1, labels)
            scaler.scale(loss_1).backward()
            scaler.step(optimizer)
            scaler.update()
            losses += loss_1

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled = True):
                _, _, _, xc1, _, _, _, _  = model(img_1)
                loss_1 = loss_fn(xc1, labels)
            scaler.scale(loss_1).backward()
            scaler.step(optimizer)
            scaler.update()
            losses += loss_1
        
        return losses
    else:
        for img_1, labels in tqdm(unlabel_dl, desc = "rotation train", position = 1, leave = False):
            img_1, labels = img_1.to(args.device), labels.to(args.device)
            # img_1 = jigsaw_generator(img_1, 8)
            with torch.cuda.amp.autocast(enabled = True):
                xc1  = model(img_1)
                origin_loss = loss_fn(xc1, labels)
            scaler.scale(origin_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses += origin_loss
        
        return losses


def batch_trainier(args, model, train_dl, optimizer, loss_fn, epoch):
    model.train()
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    losses = 0
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    for img, labels in tqdm(train_dl, desc = "batch_trainier", position = 1, leave = False):
        img, labels = img.to(args.device), labels.to(args.device)
        
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.epochs, lr[nlr])
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            output_1, _, _, _, _, _, _, _ =  model(img)   
            loss = loss_fn(output_1, labels)
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            table_22_img = jigsaw_generator(img, args.patches[1])
            _, output_2, _, _, _, _, _, _ = model(table_22_img)  
            loss = loss_fn(output_2, labels)
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            table_33_img = jigsaw_generator(img, args.patches[2])
            _, _, output_3, _, _, _, _, _ = model(table_33_img)     
            loss = loss_fn(output_3, labels)
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled = True):
            table_4_img = jigsaw_generator(img, 8)
            _, _, _, output_4, _, _, _, _ = model(table_4_img)   
            loss = loss_fn(output_4, labels) * 2 
        losses += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses