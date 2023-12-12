import yaml
import os
import shutil
import argparse

def load_yaml(args, yml):
    with open(yml, 'r', encoding='utf-8') as fyml:
        dic = yaml.load(fyml.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])

def build_record_folder(args):

    if not os.path.isdir("./records"):
        os.mkdir("./records")
    args.save_dir = os.path.join("./records", args.exp_name)
    print(args.save_dir)
    with open("./configs/config.yaml", 'r') as rfile:
        cfg = yaml.safe_load(rfile)
        
    cfg['eval_model'] = args.save_dir
    
    with open("./configs/config.yaml", 'w') as file:
        yaml.dump(cfg, file, sort_keys=False)
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "backup"), exist_ok=True)
    # shutil.copy(args.c, os.path.join(args.save_dir, "config.yaml"))

def update_record_yaml(args):
    args.save_dir = os.path.join("./records", args.exp_name)
    shutil.copy(args.c, os.path.join(args.save_dir, "config.yaml"))


def get_args(with_deepspeed: bool=False):

    parser = argparse.ArgumentParser("Fine-Grained Visual Classification")
    parser.add_argument("--c", default="", type=str, help="config file path")
    args = parser.parse_args()

    load_yaml(args, args.c)
    build_record_folder(args)
    print(args.save_dir)
    return args

