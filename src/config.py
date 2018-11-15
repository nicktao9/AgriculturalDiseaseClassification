""" 
About Config 
"""

# main.py
## hyperparameter
batch_size = 80
num_epochs = 10
learn_rate = 1e-3
## 画曲线
display_plot = False
visualize_win = "senet"
## 日志
verbose = True

## 模型路径

savedModel_path = "../saved_weight/"              
loadModel_path = "../saved_weight/checkpoint.pth"

# dataset1.py
trainset_dir1 = "../dataset/train_set/"
valset_dir1 = "../dataset/val_set/" 

# dataset2.py
trainset_dir2 = "../dataset/new_train_set/"   
valset_dir2 = "../dataset/new_val_set/"


# submit.py
testset_dir = "../dataset/testB_set/"
outJson_path = "../Result/result.json"
