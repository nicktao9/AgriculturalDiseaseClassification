import torch,json
import os,glob,json
import torch.nn as nn 
import numpy as np 
import torch.utils.data  as Data
from sklearn import metrics
import config
from models.senet import SENet,SEResNetBottleneck
# from models.residual_attention_network import ResidualAttentionModel_56
# import gluoncvth as gcv


from torchvision import transforms as T
from PIL import Image
batch_size = config.batch_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class datasets(Data.Dataset):
    def __init__(self,transform = None):
        root = config.testset_dir
        imgs = []
        images_path = os.path.join(root,"images")
        for i in glob.glob(images_path + "/*"):
            imgs.append(i)
        self.images = imgs

        
        if transform is None:
            normalize = T.Normalize(mean = [0.5,0.5,0.5],
                                            std = [0.5,0.5,0.5])
        self.transforms = T.Compose([
            T.Resize((224,224)),
            # T.RandomHorizontalFlip(p = 1),
            # T.RandomRotation(degrees = 45),
            T.ToTensor(),   
            normalize
        ])
    def __getitem__(self,index):
        img_path = self.images[index]
        img_name = img_path.split("/")[-1]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data,img_name
    def __len__(self):
        return len(self.images)
test_datasets = datasets()
test_loader = Data.DataLoader(test_datasets,batch_size = batch_size,shuffle = False)

model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=1000)
model.last_linear = nn.Linear(2048,61)
model.load_state_dict(torch.load(config.loadModel_path))

# model = ResidualAttentionModel_56()
# model.load_state_dict(torch.load(config.loadModel_path))

# model = gcv.models.model_zoo.get_model("resnet50",pretrained = False)
# model.avgpool = nn.AdaptiveAvgPool2d(1)
# model.fc = nn.Linear(2048,61)
# model.load_state_dict(torch.load(config.loadModel_path))


model.to(device)
model.eval()
result = []
with torch.no_grad():
    for i,(data,img_name) in enumerate(test_loader):
        data = data.to(device)
        outputs = model(data)
        for j in range(len(outputs)):
            dict1 = {}
            dict1["disease_class"] = torch.max(outputs[j],0)[1].item()
            dict1["image_id"] = img_name[j]
            result.append(dict1)
            del dict1
with open(config.outJson_path,"w") as f:
   json.dump(result,f)

