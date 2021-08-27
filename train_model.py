import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt


def plotify(train_losses, val_losses):
    plt.plot(train_losses, label = 'Train Loss')
    plt.plot(val_losses, label = 'Test Loss')
    plt.show()


def train(criterion, model, optimizer, n_epochs, device,train_loader,val_loader,tb,PATH):
    import time
    import copy
    from tqdm import tqdm
    since = time.time()
    train_losses = []
    valid_losses = []
    model.to(device)
    valid_loss_min = np.Inf
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in tqdm(range(1, n_epochs + 1)):
        print('\n')
        print("=" * 30)
        print(f'\nEpoch : {epoch}')
        train_loss = 0.0
        valid_loss = 0.0
        running_corrects = 0.0
        model.train()
        for images, labels in train_loader:
            images, labels = Variable(images.to(device)), Variable(labels.to(device))
            # print (images.size(0))
            # print (labels.shape)
            labels = labels.view(-1, 1)
            # print (labels.shape)
            optimizer.zero_grad()
            # print (images[0])
            outputs = model(images)
            # print (outputs)
            # outputs = torch.mean(outputs)
            # print (outputs)
            # print (labels)
            # print (outputs.shape)
            # _, preds = torch.max(outputs, 1)
            # loss = criterion(outputs, labels)
            loss = criterion(outputs, labels, cat='train')
            # print(loss)

            loss.backward()
            optimizer.step()
            # print (loss)
            train_loss += loss.item() * images.size(0)
            # print (loss.item())
            # running_corrects += torch.sum(preds == labels.data)
            # break
        else:
            with torch.no_grad():
                model.eval()
                for images, labels in val_loader:
                    images, labels = Variable(images.to(device)), Variable(labels.to(device))
                    # images, labels = images.to(device), labels.to(device)
                    labels = labels.view(-1, 1)

                    output = model(images)

                    # loss = criterion(output, labels)
                    loss = criterion(output, labels, cat='valid')

                    valid_loss += loss.item() * images.size(0)
        train_loss = train_loss / len(train_loader.sampler)
        train_losses.append(train_loss)
        valid_loss = valid_loss / len(val_loader.sampler)
        valid_losses.append(valid_loss)

        tb.add_scalar("Every epoch train loss", train_loss, epoch)
        tb.add_scalar("Every epoch valid loss", valid_loss, epoch)

        print(f"\nTraining Loss : {train_loss} \nValidation Loss : {valid_loss}")

        if valid_loss < valid_loss_min:
            print(f"Validation Loss decreased from {valid_loss} to  {valid_loss_min} ....Saving model")
            torch.save(model.state_dict(), PATH)
            valid_loss_min = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    plotify(train_losses, valid_losses)
    return model