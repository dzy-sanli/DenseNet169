import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os


cwd = os.getcwd()
# # print(cwd)
# # 打印 : D:\新疆大学医疗图像研究-Densenet\MURA(1)\DenseNet-Code
# path = cwd + R"\data\MURA-v1.1"
# # print(path + R"\train_image_paths.csv")
# # 打印 D:\新疆大学医疗图像研究-Densenet\MURA(1)\DenseNet-Code\data\MURA-v1.1
# train_df = pd.read_csv(path + r"\train_image_paths.csv",header=None,names=['FilePath'])
# valid_df = pd.read_csv(path + r"\valid_image_paths.csv",header=None,names=['FilePath'])
#
# train_labels_df = pd.read_csv(path + r"\train_labeled_studies.csv",header=None,names=['FilePath', 'Labels'])
# valid_labels_df = pd.read_csv(path + r"\valid_labeled_studies.csv",header=None,names=['FilePath', 'Labels'])
#
# # print(train_df.iloc[5,0])  iloc[0:4,1] 代表选取0到第四行的数据，且只取第一列的


def images_toshow(columns,rows,train_df):
    fig = plt.figure(figsize=(32, 21))
    ax = []
    for i in range(columns*rows):
        # img = np.array(Image.open(train_img_path.values[k][0]))
        k = random.randint(0, (train_df.size - 1))
        img = mpimg.imread(cwd + r"/data/""" + train_df.iloc[k, 0])
        title = (train_df.iloc[k, 0]).split('/')
        title = title[2] + '-' + title[3] + '-' + title[4]
        # create subplot and append to ax
        ax.append(fig.add_subplot(columns,rows, i + 1))
        ax[-1].set_title(title)  # set title
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap="gray")
    plt.tight_layout(True,h_pad=4)
    plt.show()  # finally, render the plot


# images_toshow(2,3,train_df)
# DateFrame 加列，数据打标签
def Data_df_precessing(train_df,valid_df):
    train_df['Label'] = train_df.apply(lambda x:1 if 'positive' in x.FilePath else 0, axis=1) # 1代表阳性健康
    train_df['BodyPart'] = train_df.apply(lambda x: x.FilePath.split('/')[2][3:],axis=1) # 添加身体部位类别
    train_df['StudyType'] = train_df.apply(lambda x: x.FilePath.split('/')[4][:6],axis=1)  #研究类别

    valid_df['Label'] = valid_df.apply(lambda x:1 if 'positive' in x.FilePath else 0, axis=1)
    valid_df['BodyPart'] = valid_df.apply(lambda x: x.FilePath.split('/')[2][3:],axis=1)
    valid_df['StudyType'] = valid_df.apply(lambda x: x.FilePath.split('/')[4][:6],axis=1)
    return train_df,valid_df


def Date_df_plot_bar(train_df,valid_df):
    # DateFrame 索引改变，各类查看
    print(train_df.set_index(["FilePath", "BodyPart"]).count(level="BodyPart"))
    # train_df 正负标签
    print("训练集样本")
    print(train_df.set_index(["FilePath", "Label"]).count(level="Label"))
    print("验证集样本")
    print(valid_df.set_index(["FilePath", "Label"]).count(level="Label"))

    plt.figure(figsize=(15, 7))
    sns.countplot(data=train_df, x='Label', hue='Label')
    plt.savefig("train_df_all_label.jpg")
    plt.figure(figsize=(15, 7))
    sns.countplot(data=train_df, x='BodyPart', hue='Label')
    plt.savefig("train_df_BodyPart_label.jpg")
    plt.figure(figsize=(15, 7))
    sns.countplot(data=train_df, x='StudyType', hue='Label')
    plt.savefig("train_df_StudyType_label.jpg")

    plt.figure(figsize=(15, 7))
    sns.countplot(data=valid_df, x='Label', hue='Label')
    plt.savefig("valid_df_all_label.jpg")
    plt.figure(figsize=(15, 7))
    sns.countplot(data=valid_df, x='BodyPart', hue='Label')
    plt.savefig("valid_df_BodyPart_label.jpg")
    plt.figure(figsize=(15, 7))
    sns.countplot(data=valid_df, x='StudyType', hue='Label')
    plt.savefig("valid_df_StudyType_label.jpg")
    plt.show()

# Data_df_precessing(train_df,valid_df)
# print(train_df,valid_df)



# images_toshow = []
# for i in range(2):
#     k = random.randint(0, (train_df.size - 1))
#     img = mpimg.imread(cwd + r"/data/""" + train_df.iloc[k, 0])
#     # title = (train_df.iloc[k, 0]).split('/')
#     # title = title[2] + '-' + title[3] + '-' + title[4]
#     images_toshow.append(img)
# print(type(images_toshow))
# images_toshow = np.array(images_toshow)
# print(images_toshow.shape)
# tb = SummaryWriter(comment="MUCAData_image_Tensorboard_show")
# grid = torchvision.utils.make_grid(images_toshow)
# tb.add_image("images",images_toshow)
# tb = SummaryWriter(comment="MUCAData_image_Tensorboard_show")
# grid = torchvision.utils.make_grid(images_toshow)
# tb.add_image("images")
# def Data_image_Tensorboard_show(tb,images_toshow):
#     grid = torchvision.utils.make_grid(images_toshow)
#     tb.add_image("imagse",grid)
