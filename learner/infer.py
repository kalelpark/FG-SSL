import torch
from tqdm import tqdm
from utils import *
import times

def base_valid(args, model, valid_dl, loss_fn):
    model.eval()
    pred_list, label_list = torch.Tensor([]), torch.Tensor([])

    test_loss, total, correct = 0, 0, 0
    for idx, (image, label) in enumerate(tqdm(valid_dl, desc = "Base tester", position = 1, leave = False)):
        image, label = image.float().to(args.device), label.type(torch.LongTensor).to(args.device)
        with torch.cuda.amp.autocast():
            proba = model(image)
            loss = loss_fn(proba, label)
        
        test_loss += loss.item()

        _, predicted = torch.max(proba.data, 1)
        total += label.size(0)
        correct += predicted.eq(label.data).cpu().sum()
            
        pred_list = torch.cat([pred_list.cpu(), predicted.cpu()], dim = 0)
        label_list = torch.cat([label_list.cpu(), label.cpu()], dim = 0)

    test_acc = 100. * float(correct) / total
    test_loss = test_loss / len(valid_dl)

    test_f1, test_precision, test_recall = get_metrics(args, label_list, pred_list)
    return test_acc, test_loss, test_f1, test_precision, test_recall


def ssl_valid(args, model, test_dl, loss_fn):
    model.eval()
    pred_list, label_list = torch.Tensor([]), torch.Tensor([])
    test_loss, total, correct = 0, 0, 0
    for image, label in tqdm(test_dl, desc = "tester", position = 1, leave = False):
        image, label = image.float().to(args.device), label.type(torch.LongTensor).to(args.device)
        
        with torch.cuda.amp.autocast():
            output_1, output_2, output_3, output_concat, _, _, _, _, _, _, _ = model(image)
            outputs_com = output_1 + output_2 + output_3 + output_concat
            loss = loss_fn(outputs_com, label)

        test_loss += loss.item()
        _, pred = torch.max(outputs_com.data, 1)
        total += label.size(0)
        correct += pred.eq(label.data).cpu().sum()

        pred_list = torch.cat([pred_list.cpu(), pred.cpu()], dim = 0)
        label_list = torch.cat([label_list.cpu(), label.cpu()], dim = 0)

    test_acc = 100. * float(correct) / total
    test_loss = test_loss / len(test_dl)

    test_f1, test_precision, test_recall = get_metrics(args, label_list, pred_list)
    return test_acc, test_loss, test_f1, test_precision, test_recall

def moco_vaild(args, model, test_dl, loss_fn):
    model.eval()
    pred_list, label_list = torch.Tensor([]), torch.Tensor([])

    test_loss, total, correct = 0, 0, 0
    for idx, (img, labels) in enumerate(tqdm(test_dl, desc = "Fine tester", position = 1, leave = False)):
        img, labels = img.to(args.device), labels.to(args.device)
        
        output_encoder = model(img)
        loss = loss_fn(output_encoder, labels)
        test_loss += loss.item()
        _, pred = torch.max(output_encoder.data, 1)
        total += labels.size(0)
        correct += pred.eq(labels.data).cpu().sum()

        pred_list = torch.cat([pred_list.cpu(), pred.cpu()], dim = 0)
        label_list = torch.cat([label_list.cpu(), labels.cpu()], dim = 0)

    test_acc = 100. * float(correct) / total
    test_loss = test_loss / len(test_dl)

    test_f1, test_precision, test_recall = get_metrics(args, label_list, pred_list)
    return test_acc, test_loss, test_f1, test_precision, test_recall


# def fine_vaild(args, model, test_dl, loss_fn):
#     model.eval()
#     pred_list, label_list = torch.Tensor([]), torch.Tensor([])
#     test_loss, total, correct = 0, 0, 0
#     for image, label in tqdm(test_dl, desc = "fine_vaild", position = 1, leave = False):
#         image, label = image.float().to(args.device), label.type(torch.LongTensor).to(args.device)

#         with torch.cuda.amp.autocast():
#             output_1, output_2, output_3, output_concat, _, _, _  = model(image)        # , _, _, _, _
#             outputs_com = output_1 + output_2 + output_3 + output_concat
#             loss = loss_fn(outputs_com, label)
#              # xc1, xc2, xc3, x_concat, xc_1, xc_2, xc_3, x_concat_ft       
#         test_loss += loss.item()
#         _, pred = torch.max(outputs_com.data, 1)
#         total += label.size(0)
#         correct += pred.eq(label.data).cpu().sum()

#         pred_list = torch.cat([pred_list.cpu(), pred.cpu()], dim = 0)
#         label_list = torch.cat([label_list.cpu(), label.cpu()], dim = 0)

#     test_acc = 100. * float(correct) / total
#     test_loss = test_loss / len(test_dl)

#     test_f1, test_precision, test_recall = get_metrics(args, label_list, pred_list)
#     return test_acc, test_loss, test_f1, test_precision, test_recall
def fine_vaild(args, model, test_dl, loss_fn):
    model.eval()
    pred_list, label_list = torch.Tensor([]), torch.Tensor([])

    test_loss, total, correct = 0, 0, 0
    for idx, (image, label) in enumerate(tqdm(test_dl, desc = "Fine tester", position = 1, leave = False)):
        image, label = image.float().to(args.device), label.type(torch.LongTensor).to(args.device)

        with torch.cuda.amp.autocast():
            output_1, output_2, output_3, output_concat, _, _, _, _ = model(image)
            # output_1, output_2, output_3, output_concat = model(image)
            outputs_com = output_1 + output_2 + output_3 + output_concat
            loss = loss_fn(outputs_com, label)
        
        test_loss += loss.item()
        _, pred = torch.max(outputs_com.data, 1)
        total += label.size(0)
        correct += pred.eq(label.data).cpu().sum()

        pred_list = torch.cat([pred_list.cpu(), pred.cpu()], dim = 0)
        label_list = torch.cat([label_list.cpu(), label.cpu()], dim = 0)

    test_acc = 100. * float(correct) / total
    test_loss = test_loss / len(test_dl)

    test_f1, test_precision, test_recall = get_metrics(args, label_list, pred_list)
    return test_acc, test_loss, test_f1, test_precision, test_recall