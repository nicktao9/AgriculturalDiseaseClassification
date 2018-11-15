import os,glob,json
import numpy as np 
import torch.utils.data as Data
import config
from torchvision import transforms as T
from PIL import Image

class datasets1(Data.Dataset):
    def __init__(self,is_train,transform = None):
        if is_train:
            root = config.trainset_dir1
            json_path = os.path.join(root,"train_annotations.json")
            images_path = os.path.join(root,"images")
        else:
            root = config.valset_dir1
            json_path = os.path.join(root,"val_annotations.json")
            images_path = os.path.join(root,"images")
        with open(json_path) as f:
            data = json.load(f)
        imgs = []
        labels = []
        for i in range(len(data)):
            imgs.append(images_path +"/"+ data[i]["image_id"])
            labels.append(data[i]["disease_class"])
        self.images = imgs
        self.labels = labels
        if transform is None:
            normalize = T.Normalize(mean = [0.5,0.5,0.5],
                                            std = [0.5,0.5,0.5])
            if is_train:
                self.transforms = T.Compose([
                T.Resize((224,224)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(45),
                # T.ColorJitter(),
                T.ToTensor(),   
                normalize
               ])
            if not is_train:
                self.transforms = T.Compose([
                T.Resize((224,224)),
                # T.RandomHorizontalFlip(),
                T.ToTensor(),   
                normalize,
               ])
    def __getitem__(self,index):
        img_path = self.images[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        label = self.labels[index]
        return data,label
    def __len__(self):
        return len(self.labels)
class datasets2(Data.Dataset):
    def __init__(self,is_train = True,transform = None):
        if is_train:
            imgs = []
            labels = []
            for root,dirs,files in os.walk(config.trainset_dir2):
                for file in files:
                    imgs.append(os.path.join(root,file))
                    labels.append(int(os.path.join(root,file).split("/")[-2]))
        else:
            root = config.valset_dir1
            json_path = os.path.join(root,"val_annotations.json")
            images_path = os.path.join(root,"images")
            with open(json_path) as f:
                data = json.load(f)
            imgs = []
            labels = []
            for i in range(len(data)):
                imgs.append(images_path +"/"+ data[i]["image_id"])
                labels.append(data[i]["disease_class"])
                
        self.images = imgs
        self.labels = labels
        if transform is None:
            normalize = T.Normalize(mean = [0.5, 0.5, 0.5],
                                            std = [0.5, 0.5, 0.5])
            if is_train:
                self.transforms = T.Compose([
                    T.Resize((224,224)),
                    # T.FixRandomRotate(bound='Random'),
                    # T.RandomHflip(),
                    # T.RandomVflip(),
                    # T.RandomHorizontalFlip(),
                    T.ToTensor(),   
                    normalize,
                ])
            if not is_train:
                self.transforms = T.Compose([
                    T.Resize((224,224)),
                    T.ToTensor(),   
                    normalize,
                ])
    def __getitem__(self,index):
        img_path = self.images[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        label = self.labels[index]
        return data,label
    def __len__(self):
        return len(self.images)

""" 
TODO:



 """



