import sys, os, time
import visdom
import time,glob
import numpy as np
import cv2,json,shutil
import logging
from tqdm import tqdm

""" Create a new dir """
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
""" Visuallizer Module """
class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)

        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
""" log module """
class log(object):
    """
    记录日志
    log = log()
    log.printf("This is a good start {}".format(1))
    """
    def __init__(self,
    level = logging.DEBUG,
    format1 = '%(asctime)s %(filename)s : %(levelname)s %(message)s',
    datefmt = '%Y-%m-%d %A %H:%M:%S',
    filename = os.path.join("../Result/","log.txt"),
    filemode = 'w'):
        logging.basicConfig(
        level= level, # 定义输出到文件的log级别，大于此级别的都被输出
        format= format1, # 定义输出log的格式
        datefmt= datefmt, # 时间
        filename= filename, # log文件名
        filemode=filemode) # 写入模式“w”或“a”
    def printf(self,str):
        logging.info(str)
def img2classfication(input_json_path,input_file_path,outputs_folders_path):
    """put the picture  of json file in the folders of corresponding label
    Args:
        input_json_path     :origion json path
        input_file_path     :all images folder  
        outputs_folders_path:outputs path of file
    Returns:
        different label folders in outputs_folders_path
    """
    with open(input_json_path,'r') as f:
        data_dict = json.load(f)
        with tqdm(total = len(data_dict),unit= 'pic') as pbar:
            for data in data_dict:
                data_name = data['image_id']
                data_label = data['disease_class']
                create_folder(outputs_folders_path +"/"+str(data_label))                
                shutil.copy(input_file_path + "/" + data_name,outputs_folders_path + "/" + str(data_label) +"/" + data_name)                
                pbar.update(1)
if __name__ == "__main__":
    img2classfication("../../datasets/ai_challenge/val_set/val_annotations.json","../../datasets/ai_challenge/val_set/images/","../../datasets/ai_challenge/new_val_set/")