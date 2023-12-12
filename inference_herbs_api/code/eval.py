from typing import Union
# import os
import torch.nn as nn
import torch.nn.functional as F
# import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import csv
import yaml
# from math import sqrt
import torch
# import numpy as np
with open("./configs/config.yaml", "r", encoding = "utf8") as stream:
    conf = yaml.load(stream, Loader=yaml.CLoader)

# store all classifier result (predict)
true = []
predds = []
for times in range(0,9):
    true.append([])
    predds.append([])
cls_cm_index = 0

# make Confusion Matrix and classification report
def Confusion_Matrix(true, predds):
    print("inference_stop")
    true = torch.Tensor(true,device = 'cpu')
    predds = torch.Tensor(predds,device = 'cpu')
    print(len(true),len(predds ))
    with open("./result/classification_report.csv", 'w', newline = '', encoding = "utf8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([classification_report(true, predds, target_names = conf['target_name'])])
    cm = confusion_matrix(true,predds)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = conf['target_name'])
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(ax = ax,cmap = plt.cm.Blues,values_format = 'g')
    plt.xticks(rotation=+60)
    plt.savefig("./result/CM.jpg")
    print(true)
    print(predds)

def suppression(target: torch.Tensor, threshold: torch.Tensor, temperature: float = 2):
    """
    target size: [B, S, C]
    threshold: [B',]
    """
    B = target.size(0)
    target = torch.softmax(target / temperature, dim=-1)
    # target = 1 - target
    return target

@torch.no_grad()
def cal_train_metrics(args, msg: dict, outs: dict, labels: torch.Tensor, batch_size: int,
                      thresholds: dict):
    """
    only present top-1 training accuracy
    """
    total_loss = 0.0

    if args.use_fpn:
        for i in range(1, 5):
            acc = top_k_corrects(outs["layer"+str(i)].mean(1), labels, tops=[1])["top-1"] / batch_size
            acc = round(acc * 100, 2)
            msg["train_acc/layer{}_acc".format(i)] = acc
            loss = F.cross_entropy(outs["layer" + str(i)].mean(1), labels)
            msg["train_loss/layer{}_loss".format(i)] = loss.item()
            total_loss += loss.item()
            gt_score_map = outs["layer" + str(i)]
            thres = torch.Tensor(thresholds["layer" + str(i)])
            gt_score_map = suppression(gt_score_map, thres)
            logit = F.log_softmax(outs["FPN1_layer" + str(i)] / args.temperature, dim=-1)
            loss_b = nn.KLDivLoss()(logit, gt_score_map)
            msg["train_loss/layer{}_FPN1_loss".format(i)] = loss_b.item()

    if args.use_selection:
        for name in outs:
            if "select_" not in name:
                continue
            B, S, _ = outs[name].size()
            logit = outs[name].view(-1, args.num_classes)
            labels_true = labels.unsqueeze(1).repeat(1, S).flatten(0)
            acc = top_k_corrects(logit, labels_true, tops=[1])["top-1"] / (B * S)
            acc = round(acc * 100, 2)
            msg["train_acc/{}_acc".format(name)] = acc
            labels_true = torch.zeros([B * S, args.num_classes]) - 1
            labels_true = labels_true.to(args.device)
            loss = F.mse_loss(F.tanh(logit), labels_true)
            msg["train_loss/{}_loss".format(name)] = loss.item()
            total_loss += loss.item()

        for name in outs:
            if "drop_" not in name:
                continue
            B, S, _ = outs[name].size()
            logit = outs[name].view(-1, args.num_classes)
            labels_1 = labels.unsqueeze(1).repeat(1, S).flatten(0)
            acc = top_k_corrects(logit, labels_1, tops=[1])["top-1"] / (B*S)
            acc = round(acc * 100, 2)
            msg["train_acc/{}_acc".format(name)] = acc
            loss = F.cross_entropy(logit, labels_1)
            msg["train_loss/{}_loss".format(name)] = loss.item()
            total_loss += loss.item()

    if args.use_combiner:
        acc = top_k_corrects(outs['combiner'], labels, tops=[1])["top-1"] / batch_size
        acc = round(acc * 100, 2)
        msg["train_acc/combiner_acc"] = acc
        loss = F.cross_entropy(outs['combiner'], labels)
        msg["train_loss/combiner_loss"] = loss.item()
        total_loss += loss.item()

    if "ori_out" in outs:
        acc = top_k_corrects(outs["ori_out"], labels, tops=[1])["top-1"] / batch_size
        acc = round(acc * 100, 2)
        msg["train_acc/ori_acc"] = acc
        loss = F.cross_entropy(outs["ori_out"], labels)
        msg["train_loss/ori_loss"] = loss.item()
        total_loss += loss.item()

    msg["train_loss/total_loss"] = total_loss



@torch.no_grad()
def top_k_corrects(preds: torch.Tensor, labels: torch.Tensor, tops: list = [1, 3, 5]):
    """
    preds: [B, C] (C is num_classes)
    labels: [B, ]
    """
    if preds.device != torch.device('cpu'):
        preds = preds.cpu()
    if labels.device != torch.device('cpu'):
        labels = labels.cpu()
    tmp_cor = 0
    corrects = {"top-"+str(x):0 for x in tops}
    sorted_preds = torch.sort(preds, dim=-1, descending=True)[1]
    for number in range(tops[-1]):
        tmp_cor += sorted_preds[:, number].eq(labels).sum().item()
        # records
        if "top-"+str(number + 1) in corrects:
            corrects["top-"+str(number + 1)] = tmp_cor
    return corrects


@torch.no_grad()
def _cal_evalute_metric(corrects: dict,
                        total_samples: dict,
                        logits: torch.Tensor,
                        labels: torch.Tensor,
                        this_name: str,
                        scores: Union[list, None] = None,
                        score_names: Union[list, None] = None):
    global cls_cm_index
    for i in range(len(labels)):
        pred_temp = torch.max(logits[i], dim=-1)[1]
        pred_logit = torch.max(logits[i], dim=-1)[0]
        label_temp = labels[i]
        predds[cls_cm_index].append(pred_temp)
        true[cls_cm_index].append(label_temp)

    # return top-1, top-3, top-5 accuracy
    tmp_corrects = top_k_corrects(logits, labels, tops=[1, 3])

    # each layer's top-1, top-3 accuracy
    for name in tmp_corrects:
        eval_name = this_name + "-" + name
        if eval_name not in corrects:
            corrects[eval_name] = 0
            total_samples[eval_name] = 0
        corrects[eval_name] += tmp_corrects[name]
        total_samples[eval_name] += labels.size(0)

    if scores is not None:
        scores.append(logits)
    if score_names is not None:
        score_names.append(this_name)

    cls_cm_index += 1

@torch.no_grad()
def _average_top_k_result(corrects: dict, total_samples: dict, scores: list, labels: torch.Tensor,
    tops: list = [1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """
    scores is a list contain:
    [
        tensor1, 
        tensor2,...
    ] tensor1 and tensor2 have same size [B, num_classes]
    """
    # initial
    global cls_cm_index
    temp = cls_cm_index
    for t in tops:
        eval_name = "highest-{}".format(t)
        if eval_name not in corrects:
            corrects[eval_name] = 0
            total_samples[eval_name] = 0
        total_samples[eval_name] += labels.size(0)

    if labels.device != torch.device('cpu'):
        labels = labels.cpu()
    batch_size = labels.size(0)

    # B, 5, C (batch size: 9(layer * 4 + fpn * 4 + combine) : class num)
    scores_t = torch.cat([s.unsqueeze(1) for s in scores], dim=1)

    if scores_t.device != torch.device('cpu'):
        scores_t = scores_t.cpu()

    # batch size: 9(layer * 4 + fpn * 4 + combine)
    max_scores = torch.max(scores_t, dim=-1)[0]

    for b in range(batch_size):
        cls_cm_index = temp
        tmp_logit = None

        # 將 layer * 4 + fpn * 4 + combine 由大到小排列
        ids = torch.sort(max_scores[b], dim=-1)[1]

        # height 1 -> height 9
        for i in range(tops[-1]):
            top_i_id = ids[i]
            if tmp_logit is None:
                tmp_logit = scores_t[b][top_i_id]
            else:
                tmp_logit += scores_t[b][top_i_id]
            # Record results
            if i + 1 in tops:
                if torch.max(tmp_logit, dim=-1)[1] == labels[b]:
                    eval_name = "highest-{}".format(i + 1)
                    corrects[eval_name] += 1
            # Confusion matrix
            pred_logit = torch.max(tmp_logit, dim=-1)[0]
            pred_temp = torch.max(tmp_logit, dim=-1)[1]
            label_temp = labels[b]
            predds[cls_cm_index].append(pred_temp)
            true[cls_cm_index].append(label_temp)
            cls_cm_index += 1

def evaluate(args, model, test_loader):
    """
    [Notice: Costom Model]
    If you use costom model, please change fpn module return name (under
    if args.use_fpn: ...)
    [Evaluation Metrics]
    We calculate each layers accuracy, combiner accuracy and average-higest-1 ~
    average-higest-5 accuracy (average-higest-5 means average all predict scores
    as final predict)
    """
    model.eval()
    corrects = {}
    total_samples = {}
    # just for log
    total_batchs = len(test_loader)
    # just for log
    show_progress = [x / 10 for x in range(11)]

    progress_i = 0
    with torch.no_grad():
        """ accumulate """
        for batch_id, (ids, datas, labels) in enumerate(test_loader):
            global cls_cm_index
            cls_cm_index = 0
            score_names = []
            scores = []
            datas = datas.to(args.device)
            outs = model(datas)
            
            if args.use_combiner:
                this_name = "combiner"
                _cal_evalute_metric(corrects, total_samples, outs["combiner"], labels, this_name, scores, score_names)
            
            if args.use_fpn:
                for i in range(1, 5):
                    this_name = "layer" + str(i)
                    _cal_evalute_metric(corrects, total_samples, outs[this_name].mean(1), labels, this_name, scores, score_names)
                    this_name = "FPN1_layer" + str(i)
                    _cal_evalute_metric(corrects, total_samples, outs[this_name].mean(1), labels, this_name, scores, score_names)
                

            if "ori_out" in outs:
                this_name = "original"
                _cal_evalute_metric(corrects, total_samples, outs["ori_out"], labels, this_name)

            # ensemble for research
            # _average_top_k_result(corrects, total_samples, scores, labels)

            eval_progress = (batch_id + 1) / total_batchs
            if eval_progress > show_progress[progress_i]:
                print(".." + str(int(show_progress[progress_i] * 100)) + "%", end = '', flush = True)
                progress_i += 1
        """ calculate accuracy """
        cls_cm_find = {}
        temp_count = 0
        best_top1 = 0.0
        best_top1_name = ""
        eval_acces = {}
        for name in corrects:
            acc = corrects[name] / total_samples[name]
            acc = round(100 * acc, 3)
            eval_acces[name] = acc
            ### only compare top-1 accuracy
            if "top-1" in name or "highest" in name:
                cls_cm_find.setdefault(name, temp_count)
                if acc >= best_top1:
                    best_top1 = acc
                    best_top1_name = name
                temp_count += 1
        
        Confusion_Matrix(true[cls_cm_find[best_top1_name]],predds[cls_cm_find[best_top1_name]])
    return best_top1, best_top1_name, eval_acces