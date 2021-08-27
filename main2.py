from Data_utils import MuraDataset
from Data_read import Data_df_precessing
import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.optim as optim
from model import Loss, PretrainedDensenet, PretrainedDensenet_one,PretrainedDensenet_two
from train_model import train
from torch.autograd import Variable

# 1:读取文件为pd.DataFrame
cwd = os.getcwd()
# print(cwd)
# 打印 : D:\新疆大学医疗图像研究-Densenet\MURA(1)\DenseNet-Code
path = cwd + R"\data\MURA-v1.1"
# print(path + R"\train_image_paths.csv")
# 打印 D:\新疆大学医疗图像研究-Densenet\MURA(1)\DenseNet-Code\data\MURA-v1.1
train_df = pd.read_csv(path + r"\train_image_paths.csv", header=None, names=['FilePath'])
valid_df = pd.read_csv(path + r"\valid_image_paths.csv", header=None, names=['FilePath'])

train_labels_df = pd.read_csv(path + r"\train_labeled_studies.csv", header=None, names=['FilePath', 'Labels'])
valid_labels_df = pd.read_csv(path + r"\valid_labeled_studies.csv", header=None, names=['FilePath', 'Labels'])


train_transform = transforms.Compose([
    transforms.Resize([320,320]),
    # transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.456], [0.224])
])
val_transform = transforms.Compose([
    transforms.Resize([320, 320]),
    # transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.456], [0.224])
])

train_df, valid_df = Data_df_precessing(train_df, valid_df)
train_dataset = MuraDataset(df=train_df, transform=train_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=8, num_workers=0, shuffle=True)
val_dataset = MuraDataset(df=valid_df, transform=val_transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=8, num_workers=0, shuffle=True)

print("train_dataset lenth:", len(train_dataset))
print("val_dataset lenth:", len(val_dataset))

# 3: train_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3-1 优化Loss
total_positive_images_train = (train_df.Label == 1).sum()
total_negative_images_train = (train_df.Label == 0).sum()
Wt1_train = total_negative_images_train / (total_negative_images_train + total_positive_images_train)
Wt0_train = total_positive_images_train / (total_negative_images_train + total_positive_images_train)

total_positive_images_valid = (valid_df.Label == 1).sum()
total_negative_images_valid = (valid_df.Label == 0).sum()
Wt1_valid = total_negative_images_valid / (total_negative_images_valid + total_positive_images_valid)
Wt0_valid = total_positive_images_valid / (total_negative_images_valid + total_positive_images_valid)


Wt = dict()
Wt_train = {}
Wt_valid = {}


Wt_train['Wt1'] = torch.from_numpy(np.asarray(Wt1_train)).double().type(torch.FloatTensor).to(device)
Wt_train['Wt0'] = torch.from_numpy(np.asarray(Wt0_train)).double().type(torch.FloatTensor).to(device)
Wt_valid['Wt1'] = torch.from_numpy(np.asarray(Wt1_valid)).double().type(torch.FloatTensor).to(device)
Wt_valid['Wt0'] = torch.from_numpy(np.asarray(Wt0_valid)).double().type(torch.FloatTensor).to(device)

Wt['train'] = Wt_train
Wt['valid'] = Wt_valid
print(Wt)

# comment = "EPOCH ={},LR = {},BATCH_size = {} on network".format(run.epochs, run.lr, run.batch_size)
comment = "-test"
tb = SummaryWriter(comment=comment)
tb_images_toshow, tb_labels_toshow = next(iter(train_loader))
print(tb_images_toshow.shape)
grid = torchvision.utils.make_grid(tb_images_toshow)
tb.add_image("imagse", grid)

total_positive_images_train = (train_df.Label == 1).sum()
total_negative_images_train = (train_df.Label == 0).sum()

total_positive_images_val = (valid_df.Label == 1).sum()
total_negative_images_val = (valid_df.Label == 0).sum()
print("total_positive_images_train:", total_positive_images_train)
print("total_negative_images_train:", total_negative_images_train)
print("total_positive_images_val:", total_positive_images_val)
print("total_negative_images_val:", total_negative_images_val)


model = PretrainedDensenet_two() # 目前跑这个
tb.add_graph(model, tb_images_toshow)

criterion = Loss(Wt)



optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=0.0001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)

model.to(device)

PATH = "./Dense_net_MURA_two.pth"
model_ft = train(criterion=criterion, optimizer=optimizer, model=model, n_epochs=20,
                 device=device, train_loader=train_loader, val_loader=val_loader, tb=tb, PATH=PATH)

torch.save(model_ft.state_dict(), PATH)
tb.close()
# 4：模型保存
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
#
#
#
# batch_num = 8
# y_pred_list = []
# y_test_list = []
# for images, labels in val_loader:
#     images, labels = Variable(images.to(device)), Variable(labels.to(device)) # 加速
#     # images, labels = images.to(device), labels.to(device)
#     labels = labels.view(-1,1)
#
#     output = model(images)
#     y_pred = torch.round(output)
#     correct_results_sum = (y_pred == labels).sum().float()
#     y_pred_list.append(y_pred.cpu().detach().numpy())
#     y_test_list.append(labels.cpu().detach().numpy())
#
# y_pred_list = [j for batch in y_pred_list for j in batch]
# y_test_list = [j for batch in y_test_list for j in batch]
#
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import cohen_kappa_score
# result = confusion_matrix(y_test_list, y_pred_list)
# keppa = cohen_kappa_score(y_test_list, y_pred_list)
