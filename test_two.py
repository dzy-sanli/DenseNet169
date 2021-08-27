import torch
from model import PretrainedDensenet,PretrainedDensenet_one,PretrainedDensenet_two
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from Data_utils import  MuraDataset
import os
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Data_read import Data_df_precessing
# 1:读取文件为pd.DataFrame
cwd = os.getcwd()
# print(cwd)
# 打印 : D:\新疆大学医疗图像研究-Densenet\MURA(1)\DenseNet-Code
path = cwd + R"\data\MURA-v1.1"
# print(path + R"\train_image_paths.csv")
# 打印 D:\新疆大学医疗图像研究-Densenet\MURA(1)\DenseNet-Code\data\MURA-v1.1
train_df = pd.read_csv(path + r"\train_image_paths.csv",header=None,names=['FilePath'])
valid_df = pd.read_csv(path + r"\valid_image_paths.csv",header=None,names=['FilePath'])

train_labels_df = pd.read_csv(path + r"\train_labeled_studies.csv",header=None,names=['FilePath', 'Labels'])
valid_labels_df = pd.read_csv(path + r"\valid_labeled_studies.csv",header=None,names=['FilePath', 'Labels'])

# 2:precessing
# train_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.RandomCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ToTensor(),
#     transforms.Normalize([0.456], [0.224])
# ])
# val_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.RandomCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.456], [0.224])
# ])
# train_transform = transforms.Compose([
#     transforms.Resize([320,320]),  # resize the image to 320x320
#     # transforms.RandomCrop(320),
#     transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(), # transform it into a torch tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize to weights from ImageNet
# ])
#
# val_transform = transforms.Compose([
#     transforms.Resize([320,320]),  # resize the image to 320x320
#     # transforms.RandomCrop(320),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(), # transform it into a torch tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize to weights from ImageNet
# ])

train_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    # transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.456], [0.224])
])
val_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    # transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.456], [0.224])
])

train_df,valid_df = Data_df_precessing(train_df,valid_df)
train_dataset = MuraDataset(df=train_df, transform=train_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=8, num_workers=0, shuffle=True)
val_dataset = MuraDataset(df=valid_df, transform=val_transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=8, num_workers=0, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = "./Dense_net_MURA_two.pth"
model = PretrainedDensenet_one()
model.to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

batch_num = 8
y_pred_list = []
y_test_list = []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = Variable(images.to(device)), Variable(labels.to(device)) # 加速
        # images, labels = images.to(device), labels.to(device)
        labels = labels.view(-1,1)

        output = model(images)
        y_pred = torch.round(output)
        correct_results_sum = (y_pred == labels).sum().float()
        y_pred_list.append(y_pred.cpu().detach().numpy())
        y_test_list.append(labels.cpu().detach().numpy())

y_pred_list = [j for batch in y_pred_list for j in batch]
y_test_list = [j for batch in y_test_list for j in batch]


result = confusion_matrix(y_test_list, y_pred_list)
kappa = cohen_kappa_score(y_test_list, y_pred_list)
print(result)
print(kappa)