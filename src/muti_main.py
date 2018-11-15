import torch,os 
import logging
import torch.nn as nn 
import numpy as np 
from torch.optim import lr_scheduler
import torch.utils.data  as Data
from sklearn import metrics
from visualize import Visualizer
from dataset import datasets
import config
from models.senet import SEResNetBottleneck
from models.senet import SENet
from models.xception import Xception
if config.verbose == True:
  logging.basicConfig(
          level=logging.DEBUG, # 定义输出到文件的log级别，大于此级别的都被输出
          format='%(asctime)s %(filename)s : %(levelname)s %(message)s', # 定义输出log的格式
          datefmt='%Y-%m-%d %A %H:%M:%S', # 时间
          filename=os.path.join(os.getcwd(),config.log_name), # log文件名
          filemode='w') # 写入模式“w”或“a”
batch_size = config.batch_size
num_epochs = config.num_epochs
learn_rate = config.learn_rate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if config.display_plot == True:
  vis = Visualizer(config.visualize_win)

train_datasets = datasets(is_train = True)
train_loader = Data.DataLoader(train_datasets,batch_size= batch_size ,shuffle=True)

val_datasets = datasets(is_train = False)
val_loader = Data.DataLoader(val_datasets,batch_size = batch_size,shuffle = False)

model1 = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=0.5, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=1000)
model1.last_linear = nn.Linear(2048,61)
model1.load_state_dict(torch.load(config.loadModel_path1))
model1.to(device)

model2 = Xception()
model2.load_state_dict(torch.load(config.loadModel_path2))
model2.fc = nn.Linear(2048,61)
model2.to(device)

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss() 

optimizer1 = torch.optim.SGD(model1.parameters(), lr=learn_rate, momentum=0.9, nesterov = True,weight_decay = 0.001)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=learn_rate, momentum=0.9, nesterov = True,weight_decay = 0.001)

lr_schedule1 = lr_scheduler.StepLR(optimizer1,step_size = 8,gamma= 0.1)
lr_schedule2 = lr_scheduler.StepLR(optimizer2,step_size = 8,gamma= 0.1)

def train(epoch):
    model1.train()
    model2.train()
    lr_schedule1.step(epoch)
    lr_schedule2.step(epoch)
    total_step = len(train_datasets)
    for i, (x,y) in enumerate(train_loader):
        data = x.to(device)
        label = y.to(device)
        outputs1 = model1(data)
        outputs2 = model2(data)
        loss1 = criterion1(outputs1,label)
        loss2 = criterion2(outputs2,label)
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        loss1.backward()
        loss2.backward()
        optimizer1.step()
        optimizer2.step()

        if (i + 1) % 20 == 0:
            print('After [{}/{}] epoch | [{}/{}] batch,loss1 is {:0.4f},loss2 is {:0.4f},lr is {:0.4f}'.format(epoch + 1,num_epochs,i+1,total_step//batch_size,loss1.item(),loss2.item(),lr_schedule1.get_lr()[0]))
            if config.verbose == True:
              logging.info('After [{}/{}] epoch | [{}/{}] batch,loss1 is {:0.4f},loss2 is {:0.4f},lr is {:0.4f}'.format(epoch + 1,num_epochs,i+1,total_step//batch_size,loss1.item(),loss2.item(),lr_schedule1.get_lr()[0]))
            if config.display_plot == True:
              vis.plot("loss",loss1.item())
def test():
    model1.eval() 
    model2.eval() 
    data_pre = []
    data_true = []
    weight1 = config.weight1
    weight2 = config.weight2
    with torch.no_grad():
        for data,labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs1 = model1(data)
            outputs2 = model2(data)
            outputs = outputs1 * weight1 + outputs2 * weight2
            for j in outputs:
              data_pre.append(torch.max(j,0)[1].item())
            for i in labels:
              data_true.append(i.item())
        acc = metrics.accuracy_score(data_true,data_pre)
        print("The accuracy is {:0.2f} %".format(acc * 100))
        if config.verbose == True:
          logging.info("The accuracy is {:0.2f} %".format(acc * 100))
        if config.display_plot == True:
          vis.plot("acc",acc)
        del data_pre
        del data_true
        return acc
def main():
  ac_ = 0 #上一次预测正确的个数
  for epoch in range(num_epochs):
      train(epoch)
      ac = test() # 这一次预测正确的个数
      if ac > ac_ :
        torch.save(model1.state_dict(),config.savedModel_path1)
        torch.save(model2.state_dict(),config.savedModel_path2)
        ac_ = ac
        print('update the model')
        if config.verbose == True:
          logging.info("update the model")
      else:
        pass
  print("The acc is %s"%ac_)
  if config.verbose == True:
    logging.info("The acc is %s"%ac_)
if __name__ == '__main__':
  main()
    
