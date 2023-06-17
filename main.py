from __future__ import absolute_import, division, print_function
import os
import wandb
import torch
import torch.nn as nn
import argparse
import warnings

from tqdm import tqdm
from models import *
from dataset import *
from learner import *
from utils import *
warnings.filterwarnings(action = "ignore")      # 6, 7

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, required = True)
    parser.add_argument("--model", type = str, required = True)
    parser.add_argument("--patches", type = int, nargs = "*", required = True)
    parser.add_argument("--table", type = int, nargs = "*", required = False)

    parser.add_argument("--dataset", type = str, required = True)                   # aptos, isic2017, isic2018
    parser.add_argument("--beta", type = int, default = 1, required = False)
    parser.add_argument("--cut_prob", type = float, default = -1.1, required = False)

    parser.add_argument("--resize", type = int, default = 256, required = False)
    parser.add_argument("--centercrop", type = int, default = 224, required = False)

    parser.add_argument("--featdim", type = int, default = 512, required = False)
    parser.add_argument("--epochs", type = int, default = 1000, required = False)
    parser.add_argument("--batchsize", type = int, default = 200, required = False)
    parser.add_argument("--lr", type = float, default = 1e-2, required = False)
    parser.add_argument("--momentum", type = float, default = 0.9, required = False)
    parser.add_argument("--weight_decay", type = float, default = 5e-4, required = False)
    parser.add_argument("--gpu_ids", type = str, required = True)
    
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    seed_everything(args.seed)
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.num_classes = 2 if args.dataset in ["isic2017", "ISIC-2017", "ISIC2017"] else 7 if args.dataset in ["isic2018", "ISIC-2018", "ISIC2018"] else 5
    model = load_model(args, pretrained = True, require_grad = True)
    model = model.to(args.device)
    # model = nn.DataParallel(model).to(args.device)

    label_train_dl, unlabel_dl, test_dl = get_loader(args)
    loss_fn = nn.CrossEntropyLoss().to(args.device)
    barlow_optimizer = get_barlow_optimizer(args, model)
    init_ssl_loss = 1e10

    wandb.init(name = f"SSL (Server 6) + isic2018 (feature 128) : (100%)", project = args.dataset + " SSL img 224 dimension", reinit = True, entity = "XXXXX", config = args)
    for epoch in tqdm(range(args.epochs), desc = "SSL", position = 0, leave = True):
        ssl_loss = barlow_train(args, model, unlabel_dl, barlow_optimizer, epoch)

        if init_ssl_loss > ssl_loss:
        #     torch.save(model.state_dict(), "batchsave/isic2018_128.pt")
            torch.save(model.state_dict(), "featuresave/isic2018_128.pt")
            init_ssl_loss = ssl_loss

        wandb.log({
            "ssl train" : ssl_loss
        })

    for epoch in tqdm(range(args.epochs), desc = "ALL", position = 0, leave = True):
        losses = origin_fine_train(args, model, label_train_dl, ce_optimizer, loss_fn, epoch)
        test_acc, test_loss, test_f1, test_precision, test_recall = fine_vaild(args, model, test_dl, loss_fn)
        wandb.log({
            "SSL_Accuracy" : test_acc,
            "SSL_test_Loss" : test_loss,
            "SSL_F1 score" : test_f1,
            "SSL_Precision" : test_precision,
            "SSL_Recall" : test_recall,
            "SSL_train_Loss" : losses
        })
    