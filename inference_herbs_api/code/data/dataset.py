import random
import csv
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch
import yaml
import os

with open("./configs/config.yaml", "r", encoding = "utf8") as stream:
    conf = yaml.load(stream, Loader=yaml.CLoader)

def build_loader(args):
    train_set, train_loader = None, None
    if args.train_file is not None:
        train_set = ImageDataset(istrain=True, root=args.train_file, data_size=args.data_size, return_index=True)
        train_loader = torch.utils.data.DataLoader(train_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)

    val_set, val_loader = None, None
    if args.val_file is not None:
        val_set = ImageDataset(istrain=False, root=args.val_file, data_size=args.data_size, return_index=True)
        val_loader = torch.utils.data.DataLoader(val_set, num_workers=1, shuffle=False, batch_size=args.batch_size)

    return train_loader, val_loader

def random_sample(args):
    train_set, train_loader = None, None
    if args.train_file is not None:
        train_set = ImageDataset(istrain=True, root=args.train_file, data_size=args.data_size, return_index=True)
        train_loader = torch.utils.data.DataLoader(train_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)
    return train_loader

def get_dataset(args):
    if args.train_file is not None:
        train_set = ImageDataset(istrain=True, root=args.train_file, data_size=args.data_size, return_index=True)
        return train_set
    return None

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 istrain: bool,
                 root: str,
                 data_size: int,
                 return_index: bool = False):

        """ basic information """
        self.root = root
        self.data_size = data_size
        self.return_index = return_index

        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                    )

        if istrain:
            self.transforms = transforms.Compose([
                        transforms.Resize((512, 512), Image.BILINEAR),
                        transforms.RandomCrop((data_size, data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((512, 512), Image.BILINEAR),
                        transforms.CenterCrop((data_size, data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)
    # openset
    def getDataInfo(self, root):
        print(str(root))
        data_infos = []
        temp_class = []
        temp = []
        img_num = 0
        class_num = []
        now_class = 1
        with open(str(root), newline='', encoding = "utf8") as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                if conf['randomsampler'] == True :
                    if int(row[1]) != 0:
                        temp_class.append({"path": os.path.join(conf['image_root'], row[0]), "label": int(row[1])})
                        img_num += 1
                        if int(row[1]) != now_class:
                            class_num.append(img_num-1)
                            now_class +=1
                    else:
                        temp.append({"path": os.path.join(conf['image_root'], row[0]), "label": 0})
                else :
                    data_infos.append({"path": os.path.join(conf['image_root'], row[0]), "label": int(row[1])})
            if conf['randomsampler'] is True:
                # last class
                class_num.append(img_num)

        if conf['randomsampler'] is True:
            if "train.csv" in root:
                    choice_others = random.sample(range(len(temp)), conf['randomsampler_num'])
                    for i in choice_others:
                        data_infos.append(temp[i])
            else:
                for i in range(len(temp)):
                    data_infos.append(temp[i])
            if "train.csv" in root:
                temp = 0
                sum = 0
                for i in range(len(class_num)):
                    choice_others = random.sample(range(temp,class_num[i]), conf['randomsampler_num'])
                    for j in choice_others:
                        data_infos.append(temp_class[j])
                    sum += conf['randomsampler_num']
                    temp = class_num[i]
            else:
                for i in range(len(temp_class)):
                    data_infos.append(temp_class[i])
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        # read image by opencv.
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image from {image_path}")
        img = img[:, :, ::-1] # BGR to RGB.
        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)
        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return index, img, label
        # return img, sub_imgs, label, sub_boundarys
        return img, label