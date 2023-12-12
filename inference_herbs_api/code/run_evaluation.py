import torch
import warnings
torch.autograd.set_detect_anomaly(True)
warnings.simplefilter("ignore")
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import json
import argparse
import timm
import matplotlib
import matplotlib.pyplot as plt
import tqdm

from utils.config_utils import load_yaml
from vis_utils import ImgLoader

from math import ceil
import csv
import yaml
from data.dataset import build_loader
from torch.autograd import Variable
from sklearn.metrics import classification_report, confusion_matrix ,ConfusionMatrixDisplay
import openpyxl
from eval import evaluate
with open("./configs/config.yaml", "r") as stream:
    conf = yaml.load(stream, Loader=yaml.CLoader)




def build_model(pretrainewd_path: str,
                img_size: int, 
                fpn_size: int, 
                num_classes: int,
                num_selects: dict,
                use_fpn: bool = True, 
                use_selection: bool = True,
                use_combiner: bool = True, 
                comb_proj_size: int = None):
    from models.pim_module.pim_module import PluginMoodel
    backbone = timm.create_model('swin_base_patch4_window12_384_in22k', pretrained=True) 
    
    model = PluginMoodel(backbone = backbone,                                              
                         return_nodes = None,                                               
                        img_size = img_size,
                        use_fpn = use_fpn,
                        fpn_size = fpn_size,
                        proj_type = "Linear",
                        upsample_type = "Conv",
                        use_selection = use_selection,
                        num_classes = num_classes,
                        num_selects = num_selects, 
                        use_combiner = use_combiner,
                        comb_proj_size = comb_proj_size)

    if pretrainewd_path != "":
        ckpt = torch.load(pretrainewd_path)
        model.load_state_dict(ckpt['model'])
    
    model.eval()

    return model

@torch.no_grad()
def sum_all_out(out, sum_type="softmax"):
    target_layer_names = \
    ['layer1', 'layer2', 'layer3', 'layer4',
    'FPN1_layer1', 'FPN1_layer2', 'FPN1_layer3', 'FPN1_layer4', 
    'comb_outs']

    sum_out = None
    for name in target_layer_names:
        if name != "comb_outs":
            tmp_out = out[name].mean(1)
        else:
            tmp_out = out[name]
        
        if sum_type == "softmax":
            tmp_out = torch.softmax(tmp_out, dim=-1)
        if sum_out is None:
            sum_out = tmp_out
        else:
            sum_out = sum_out + tmp_out # note that use '+=' would cause inplace error
    return sum_out



if __name__ == "__main__":
    # ===== 0. get setting =====
    parser = argparse.ArgumentParser("Visualize SwinT Large")
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.save_dir = './result'

    # for watchdog
    init_status = dict()
    init_status['status'] = "Testing"
    init_status['idle'] = False
    init_status['completed'] = False
    with open('./status.json', 'w') as f:
        json.dump(init_status, f)

    load_yaml(args, "./configs/config.yaml")
    
    # ===== 1. build model =====
    model = build_model(pretrainewd_path = args.eval_model + "/best.pt",             
                        img_size = args.data_size, 
                        fpn_size = args.fpn_size, 
                        num_classes = args.num_classes,
                        num_selects = args.num_selects)
    model.cuda()
    img_loader = ImgLoader(img_size = args.data_size)
    
    args.train_root = None
    train_loader,val_loader = build_loader(args)
    best_top1, best_top1_name, eval_acces = evaluate(args, model, val_loader)
    print('best_top1' , best_top1)
    print('best_top1_name' , best_top1_name)
    print('eval_acces' , eval_acces)
    
    # for watchdog
    status = dict()
    status['status'] = "Testing"
    status['idle'] = True
    status['completed'] = True
    with open('./status.json', 'w') as f:
        json.dump(status, f)

