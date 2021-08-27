import torch.utils.data.dataset
import torch
from PIL import Image
import numpy as np




class MuraDataset(torch.utils.data.Dataset):

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img = Image.open('data/'+img_name).convert('L')
        label = self.df.iloc[idx, 1]
        # print (np.array(img).shape)

        if self.transform:
            img = self.transform(img)
        label = torch.from_numpy(np.asarray(label)).double().type(torch.FloatTensor)
        #         img = torch.stack(img)
        return img, label


# The paper uses the same standard deviation and mean as that of IMAGENET dataset
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
#
# train_df,valid_df = Data_df_precessing(train_df,valid_df)
# train_dataset = MuraDataset(df=train_df, transform=train_transform)
# train_loader = DataLoader(dataset=train_dataset, batch_size=8, num_workers=0, shuffle=True)
# val_dataset = MuraDataset(df=valid_df, transform=val_transform)
# val_loader = DataLoader(dataset=val_dataset, batch_size=8, num_workers=0, shuffle=True)