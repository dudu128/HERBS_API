# FastAPI
from typing import Union
from enum import Enum
from fastapi import FastAPI, File, UploadFile, Response, status, BackgroundTasks
from fastapi.responses import FileResponse
import cv2
import numpy as np
import json
import zipfile
import os
import aiofiles
import shutil
import sys
import csv
import torch
import yaml
import torchvision.transforms as transforms
from PIL import Image
import cv2
import timm
from torch.nn import DataParallel
from models.pim_module.pim_module import PluginMoodel
import uvicorn
import logging
from multiprocessing import Process, Array, Lock, Manager
import io
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

lock = Lock()
download_dir = "./download"
weight_dir = "./weights"
weight_info = "./weights/info.json"
image_type = ('.png', '.jpg', '.jpeg', '.bmp')
weight_type = ('h5', 'ckpt', 'pth', 'pt')

manager = Manager()
share_models = manager.list([])
process_model = [{'model_id' : None, 'model' : None}]

app = FastAPI()

class Device(str, Enum):
    cpu = "cpu"
    gpu = "gpu"

def transform_image(image):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
    tf = transforms.Compose([
        transforms.Resize((512, 512), Image.BILINEAR),
        transforms.CenterCrop((384, 384)),
        transforms.ToTensor(),
        normalize])
    return tf(image)  

def _critical_section_updata_weight_list(weight_list_id, w):
    with lock:
        # print(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        logger.info(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        weight_list = _get_weight_list()
        weight_list[weight_list_id] = w
        _updata_weight_list(weight_list)
        logger.info(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")
        # print(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")

def _critical_section_weight_id():
    with lock:
        print(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        w = {"weight_id": None, "name": None, "info": None, "file_name": None, "file_path" : None}
        # get the current weight list
        weight_list = _get_weight_list()
        # assign weight_id
        if len(weight_list):
            w["weight_id"] = weight_list[-1]["weight_id"] + 1
        else:
            w["weight_id"] = 0
        # add into weight list
        weight_list.append(w)
        # update the weight_list
        _updata_weight_list(weight_list)
        weight_list_id,_ = _get_weight_index(w["weight_id"])
        print("weight_id : " + str(w["weight_id"]) + " is created.")
        print(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")
    return weight_list_id, w

def _critical_section_share_models(model_list):
    with lock:
        print(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        if len(share_models) == 0:
            model_list['model_id'] = 0
            share_models.append(model_list)
        else:
            model_list['model_id'] = int(len(share_models))
            share_models.append(model_list)
        print("model ID " + str(model_list['model_id']) + " is created.")
        print(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")
    return model_list['model_id']

def _build_model(cfg):
    backbone = timm.create_model('swin_base_patch4_window12_384_in22k', pretrained=True) 
    model = PluginMoodel(backbone = backbone,
                        return_nodes = None,
                        img_size = cfg['data_size'],
                        use_fpn =  cfg['use_fpn'],
                        fpn_size =  cfg['fpn_size'],
                        proj_type = "Linear",
                        upsample_type = "Conv",
                        use_selection =  cfg['use_selection'],
                        num_classes =  cfg['num_classes'],
                        num_selects =  cfg['num_selects'],
                        use_combiner =  cfg['use_combiner'],
                        comb_proj_size =  None)
    return model

def _get_weight_list():
    if not os.path.isfile(weight_info):
        return []
    else:
        with open(weight_info, mode='r') as file:
            weight_list = json.load(file)
        return weight_list

def _updata_weight_list(weight_list):
    with open(weight_info, mode='w') as file:
        json.dump(weight_list, file, ensure_ascii=False, indent=4)
    logger.info("weight list update!")
    # print("weight list update!")
 
def _load_model(w_dir, weight_info):
    config_file = os.path.join(w_dir, "config.yaml")
    with open(config_file, 'r') as stream:
        try:
            cfg = yaml.load(stream, Loader=yaml.CLoader)
        except yaml.YAMLError as exc:
            print(exc)
    model = _build_model(cfg)
    parallel_model = DataParallel(model)
    weight_path = os.path.join(w_dir, str(weight_info["file_name"]))
    weight = torch.load(weight_path, map_location = 'cpu')
    parallel_model.load_state_dict(weight['model'],strict = False)
    return parallel_model, cfg['target_name'], cfg['best_cls']

def _inference(imgs, model, best_cls):
    cls_name = str(best_cls[:-6]) #刪除'-top-1' 
    with torch.no_grad():
        if cls_name == 'combiner':
            logits = model(imgs)[cls_name]
        else:
            logits = model(imgs)[cls_name].mean(1)
    
    pred = torch.max(logits, dim=-1)[1]
    score = torch.max(logits, dim=-1)[0]
    return pred, score

def _check_weight_list(weight_list, weight_id):
    w_idx = next((i for i in range(len(weight_list))
                  if weight_list[i]['weight_id'] == weight_id), None)
    if w_idx is None:
        return False
    else:
        return True

def _get_weight_index(weight_id):
    weight_list = _get_weight_list()
    w_idx = next((i for i in range(len(weight_list))
                  if weight_list[i]['weight_id'] == weight_id), None)
    if w_idx is None:
        return w_idx, None
    else:
        return w_idx, weight_list[w_idx]

# @app.on_event("startup")
# async def startup_event():

#     # log_format
#     logger = logging.getLogger("uvicorn.access")
#     console_formatter = uvicorn.logging.ColourizedFormatter(
#         "(Time) : {asctime} - (Pid) : {process} - (IP-Response) : {message}",
#         style="{", use_colors=True)
#     logger.handlers[0].setFormatter(console_formatter)

#     if not os.path.exists(weight_dir):
#         os.mkdir(weight_dir)


@app.get("/weight/")
async def get_weight_list(response: Response):
    """Return weight lis

    Args:
        response (Response): response

    Returns:
        list: [dict: weight_list, int: error_code]
    """
    weight_list = _get_weight_list()

    error_code = 0
    logger.info("get_weight_list!")
    return {"weight_list": weight_list, "error_code": error_code}

@app.post("/weight/{name}")
async def post_weight(response: Response, weight: UploadFile, name: str, info: Union[str, None] = None):
    """Received the zip file of the Weights, create weight_info, 
    assign weight_id, add weight_info into record(json file),
    store the weights into weight folder by the weight_id

    Args:
        response (Response): response
        name (str): the name of the weights
        info (sre): the annotation of the weights
        weight (UploadFile): the zip file of the weights

    Returns:
        list: [int: weight_id, int: error_code]
    """
    weight_list_id, w = _critical_section_weight_id()
    w["name"] = name
    w["info"] = info
    # Store the zip
    zip_path = os.path.join(weight_dir, "{}.zip".format(str(w["weight_id"])))

    async with aiofiles.open(zip_path, mode="wb") as out_file:
        content = await weight.read()
        await out_file.write(content)

    if not zipfile.is_zipfile(zip_path):
        os.remove(zip_path)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 1, "error_msg": "Upload file is not a zip file."}

    # Extract the zip
    w_dir_path = os.path.join(weight_dir, str(w["weight_id"]))
    if not os.path.exists(w_dir_path):
        os.mkdir(w_dir_path)
    with zipfile.ZipFile(zip_path, mode='r') as zip_file:
        zip_file.extractall(w_dir_path)
    # Delete zip
    os.remove(zip_path)
    
    w["file_path"] = w_dir_path
    for root, dirs, files in os.walk(w_dir_path):
        for f in files:
            print(f)
            if f.endswith(weight_type):
                w["file_name"] = f
    # Update the weight_list
    _critical_section_updata_weight_list(weight_list_id, w)
    
    error_code = 0
    logger.info("post_weight!")
    return {"weight_id": w["weight_id"], "error_code": error_code}

@app.get("/weight/{weight_id}")
async def download_weight(response: Response, background_tasks: BackgroundTasks, weight_id: int):
    """Download the zip file contains the weights

    Args:
        response (Response): response
        background_tasks (BackgroundTasks): Do something after response
        weight_id (int): The weights of weight_id to be downloaded

    Returns:
        return:
            error 1: [int weight_id, error_code]
            error 0: zip file
    """
    error_code = 0
    w_dir_path = os.path.join(weight_dir, str(weight_id))
    zip_path = os.path.join(weight_dir, "{}.zip".format(str(weight_id)))

    if os.path.isdir(w_dir_path):
        files = os.listdir(w_dir_path)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zfile:
            for file in files:
                zfile.write(os.path.join(w_dir_path, file))
    else:
        error_code = 1

    if error_code != 0:
        return {"weight_id": weight_id, "error_code": error_code, "error_msg": "Model is not exist."}
    else:
        background_tasks.add_task(os.remove, zip_path)
        return FileResponse(zip_path)

@app.delete("/weight/{weight_id}")
async def delete_weight(response: Response, weight_id: int):
    """Delete the weights of weight_id

    Args:
        response (Response): response
        weight_id (int): the weights of weight_id to be deleted

    Returns:
        int: error_code
    """
    error_code = 0
    weight_list = _get_weight_list()
    w_idx, _ = _get_weight_index(weight_id)
    if w_idx is None:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 1, "error_msg": "Model is not exist."}
    else:
        del weight_list[w_idx]
        _updata_weight_list(weight_list)
        shutil.rmtree(os.path.join(weight_dir, str(weight_id)))
        logger.info("delete_weight!")
        

    return {"error_code": error_code}

@app.post("/weight/load/{weight_id}")
async def post_load_weight(response: Response, weight_id: int):
    """Load the weight of weight_id

    Args:
        response (Response): response
        weight_id (int): the weight of weight_id to be loaded
    Returns:
        int: error_code
    """

    weight_list = _get_weight_list()

    if _check_weight_list(weight_list, weight_id):
        id, weight_list = _get_weight_index(weight_id)
        model, classes_names, best_cls = _load_model(str(weight_list['file_path']), weight_list)
        model_list = {"model_id" : None, "name" : str(weight_list['name']), "model" : model.to('cpu'), "classes_names" : classes_names, "best_cls_name" : best_cls }
        model_id = _critical_section_share_models(model_list)
    else:
        return {"error_code": 1, "error_msg": "Model is not exist."}

    return {"error_code": 0, "model_id": model_id}

@app.get("/weight/load/")
async def get_load_weight(response: Response):
    """Get the model list
    Args:
        response (Response): response
    Returns:
        loaded models: models_list, int: error_code
    """
    models_list = []
    for i in share_models:
        temp = {"model_id" : i['model_id'], "name" :str(i['name']) }
        models_list.append(temp) 
    return {"loaded models": models_list, "error_code": 0}

def find(weight_id):
    model_exist = next((item for item in share_models if item["model_id"] == weight_id), None)
    return model_exist

def delete(weight_id):
    model_index = next((id for id, item in enumerate(share_models) if item["model_id"] == weight_id), None)
    del share_models[model_index]

@app.delete("/weight/load/{weight_id}")
async def delete_load_weight(response: Response, model_id: int):
    """Unload the model

    Args:
        response (Response): response

    Returns:
        int: error_code
    """

    model_exist = find(model_id)
    if model_exist is not None:
        delete(model_id)
        global process_model
        process_model = share_models[:]
        print(len(process_model))
        torch.cuda.empty_cache()
        logger.info("delete_load_weight!")
        return {"error_code": 0}
    else:
        return {"error_code": 1, "error_code": "Model is not loaded in the memory."}

@app.post("/inference/")
async def Inference(response: Response, file: UploadFile, model_id: int, device: Device = "cpu"):
    """Inference one image with loaded model(model_id).

    Args:
        response (Response): Http reponse
        file (UploadFile): One image
        model_id (int): Loaded Model to used
        device : cpu / gpu
    Returns:
        return: the prediction of the image.
    """
    
    share_model_index = next((id for id, item in enumerate(share_models) if item["model_id"] == model_id), None)
    if share_model_index is None:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 1, "error_msg": "Model is not exist."}
    
    '''
    將share_memory的model複製到目前process的memory中
    '''
    global process_model
    model_exist = 0
    model_index = 0
    for ids, index in enumerate(process_model):
        if index['model_id'] == share_models[share_model_index]['model_id']:
            model_index = ids
            model_exist = 1
            break
    if model_exist == 0:
        process_model = share_models[:]
        model_index = share_model_index
        print(f"Share models is copyed. (PID: {os.getpid()})")

    if device == 'cpu':
        device = 'cpu'
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")

    best_cls = process_model[model_index]['best_cls_name']
    classes_name = process_model[model_index]['classes_names']
    p_model = process_model[model_index]['model']
    model = p_model.module.to(device)
    filename = file.filename

    if not filename.lower().endswith(image_type):
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 2, "error_msg": "Upload file is not a valid file, only jpg, jpeg, png, bmp are valid format."}

    contents = await file.read()
    file_bytes = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform_image(img)
    img.unsqueeze_(0)
    imgs = img.to(device)
    preds, scores = _inference(imgs, model, best_cls)
    logger.info("Inference Finished!")
    return {"input": filename, "predict label": int(preds[0]), "predict class name": classes_name[preds[0]], "error_code": 0}

@app.post("/inference/batch")
async def Inference_Batch(response: Response, file: UploadFile, model_id: int, device: Device = "cpu"):
    """Inference a batch of images with loaded model (model_id)

    Args:
        response (Response): HTTP response
        file (UploadFile): A batch of images compressed by zip
        model_id (int): loaded model to used
        device : cpu / gpu
    Returns:
        return: the prediction of the batch of images.
    """

    inference_start = time.time()
    share_model_index = next((id for id, item in enumerate(share_models) if item["model_id"] == model_id), None)
    if share_model_index is None:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 1, "error_msg": "Model is not exist."}
    '''
    將share_memory的model複製到目前process的memory中
    '''
    global process_model
    model_exist = 0
    model_index = 0
    for ids, index in enumerate(process_model):
        if index['model_id'] == share_models[share_model_index]['model_id']:
            model_index = ids
            model_exist = 1
            break
    if model_exist == 0:
        process_model = share_models[:]
        model_index = share_model_index
        print(f"Share models is copyed. (PID: {os.getpid()})")
    copy_end = time.time()

    if device == 'cpu':
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")

    best_cls = process_model[model_index]['best_cls_name']
    classes_name = process_model[model_index]['classes_names']
    p_model = process_model[model_index]['model']
    model = p_model.module.to(device)
    filename = file.filename

    if not filename.lower().endswith('zip'):
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 2, "error_msg": "Upload file is not a valid file, only zip format."}

    file_content = await file.read()
    zip_file = io.BytesIO(file_content)
    batch_imgs = None
    with zipfile.ZipFile(zip_file) as img_zip:
        img_names = [i for i in img_zip.namelist() if i.lower().endswith(image_type)]
        for img in img_names:
            img_raw = img_zip.read(img)
            img_buf = np.frombuffer(img_raw, dtype=np.uint8)
            x = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            x = Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
            x = transform_image(x)
            x.unsqueeze_(0)
            if batch_imgs is None:
                batch_imgs = x
            else:
                batch_imgs = torch.cat([batch_imgs, x], dim=0)
    imgs = batch_imgs.to(device)
    preds, scores = _inference(imgs, model, best_cls)
    result = []
    for i in range(len(preds)):
        result.append({"input": img_names[i], "predict label": int(
            preds[i]), "predict class name": classes_name[preds[i]]})
    result.append({"error_code": 0})
    inference_end = time.time()
    # print("copy time : ", (copy_end - inference_start))
    # print("inference time : ", (inference_end - inference_start))
    logger.info("copy time : ", (copy_end - inference_start))
    logger.info("inference time : ", (inference_end - inference_start))
    logger.info("Inference_Batch Finished!")
    return result

if __name__ == "__main__":
    logger.info("Fast API Activate !!!")
