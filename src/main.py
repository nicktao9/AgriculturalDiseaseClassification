import torch,os 
import logging
import gluoncvth as gcv
import torch.nn as nn 
import numpy as np 
from torch.optim import lr_scheduler
import torch.utils.data  as Data
# from sklearn import metrics
# from fastai import *
# from fastai.vision import *
from utils import Visualizer,log,create_folder
from datasets import datasets1
from tqdm import tqdm
import config

from models.senet import SEResNetBottleneck
from models.senet import SENet
# from models.residual_attention_network import ResidualAttentionModel_56
# from models.xception import Xception

if config.verbose == True:
  log = log()  # 创建日志类
if config.display_plot == True:
  vis = Visualizer(config.visualize_win)  #创建可视化类

batch_size = config.batch_size
num_epochs = config.num_epochs
learn_rate = config.learn_rate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
create_folder("../Result")
create_folder("../saved_weight")

train_datasets = datasets1(is_train = True)
train_loader = Data.DataLoader(train_datasets,batch_size= batch_size ,shuffle=True)
val_datasets = datasets1(is_train = False)
val_loader = Data.DataLoader(val_datasets,batch_size = batch_size,shuffle = False)

# data = ImageDataBunch.from_folder(path = "../../datasets/ai_challenge/",train = 'new_train_set',valid = 'new_val_set',size = 224,bs = 80)
# data.normalize(imagenet_stats)
# train_loader,val_loader = data.train_dl,data.valid_dl

model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=0.5, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=1000)
model.last_linear = nn.Linear(2048,61)
model.load_state_dict(torch.load(config.loadModel_path),strict = True)
# model = ResidualAttentionModel_56()

# model.load_state_dict(torch.load(config.loadModel_path),strict=True)
# model = gcv.models.model_zoo.get_model("resnet50",pretrained = False)
# model.load_state_dict(torch.load("../saved_weight/resnet50-0ef8ed2d.pth"))
# model.avgpool = nn.AdaptiveAvgPool2d(1)
# model.fc = nn.Linear(2048,61)
# model = Xception().to(device)
# model.load_state_dict(torch.load("xception-b429252d.pth"))
# model.fc = nn.Linear(2048,61)
model.to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9, nesterov = True,weight_decay = 1e-4)
lr_schedule1 = lr_scheduler.MultiStepLR(optimizer,milestones = [15,23,30] ,gamma= 0.1)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    lr_schedule1.step(epoch)
    correct = 0
    total = 0
    loss_ = 0

    with tqdm(total = len(train_loader),ascii = True,ncols = 120,unit = 'bs') as pbar:
      for i, (x,y) in enumerate(train_loader):
          data = x.to(device)
          label = y.to(device)
          outputs = model(data)
          loss = criterion(outputs,label)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          loss_ += loss.item() 
          _,predicted = torch.max(outputs.data,1)
          correct += predicted.eq(label.data).cpu().sum()
          total += label.size(0)
          pbar.set_description('train')
          pbar.set_postfix_str(' | train_loss: {:.3f} | train_acc: {:.3f}% ({}/{})'.format(loss_/(i+1),100*correct.item()/total, correct, total))
          pbar.update(1)
          if (i + 1) % 20 == 0:
              if config.verbose == True:
                log.printf("| train_loss: {:.3f} | train_acc: {:.3f}% ({}/{})".format(loss_/(i+1),100*correct.item()/total, correct, total))
    return 100 * correct.item()/total,loss_ / i

def test():
    model.eval() 
    loss_ = 0
    total = 0
    correct = 0
    with torch.no_grad():
      with tqdm(total = len(val_loader),ascii = True,ncols = 120,unit = 'bs') as pbar:
        for i,(data,labels) in enumerate(val_loader):
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs,labels)
            loss_ += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            pbar.set_description('test')
            pbar.set_postfix_str(" | test_loss: {:.3f} | test_acc: {:.3f} ({}/{})".format(loss_/(i+1),100*correct.item()/total, correct, total))
            pbar.update(1)
        acc =  100*correct.item()/total
        if config.verbose == True:
          log.printf("| test_loss: {:.3f} | test_acc: {:.3f} ({}/{})".format(loss_/(i+1),100*correct.item()/total, correct, total))
    return acc,loss_ / i
def main():
  ac_ = 0 #上一次预测正确的个数
  for epoch in range(num_epochs):
      train_acc,train_loss = train(epoch)
      ac,val_loss = test() # 这一次预测正确的个数
      if config.display_plot == True:
        vis.plot("train_val_loss",[train_loss,val_loss])
        vis.plot("acc",ac)
      if ac > ac_ :
        torch.save(model.state_dict(),config.savedModel_path + "checkpoint.pth" )
        ac_ = ac
        print('Update model √')
        if config.verbose == True:
          log.printf("update the model")
          print('*'*50)
      else:
        print('Not update model')
        print('*'*50)
  print("[BINGO!] The acc is %s"%ac_)
  if config.verbose == True:
    log.printf("The acc is %s"%ac_)
if __name__ == '__main__':
  main()
    
