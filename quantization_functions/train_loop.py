import torch
import torch.nn as nn
import tqdm
import os

def train_model(train_dl, val_dl, model, optimizer, criterion, epochs=10, save='checkpoint/origin_training'):
    
    model = model.cuda()
    min_loss = float('inf')
    _step=0
    for epoch in range(epochs):
        train_bar = tqdm.tqdm(enumerate(train_dl), total=len(train_dl.dataset) // train_dl.batch_size)
        train_bar.set_description(f'train - epoch:{epoch:3d}')
        loss_mean = []
        model.train()
        for step, (x, label) in train_bar:
            x, label = x.cuda(), label.cuda()
            optimizer.zero_grad()
            y = model(x)
            loss = criterion(y, label)
            loss.backward()
            loss_mean.append(loss.item())
            train_loss = sum(loss_mean) / len(loss_mean)
            train_bar.set_postfix({'loss': train_loss})
            optimizer.step()
            _step +=1
        train_loss = sum(loss_mean) / len(loss_mean)

        val_bar = tqdm.tqdm(enumerate(val_dl), total=len(val_dl.dataset) // val_dl.batch_size)
        val_bar.set_description(f'val - epoch:{epoch:3d}')
        model.eval()
        loss_mean, acc_mean = [], []
        for step, (x, label) in val_bar:
            x, label = x.cuda(), label.cuda()
            y = model(x)
            loss = criterion(y, label)
            loss_mean.append(loss.item())
            acc = (y.argmax(dim=1) == label)
            acc_mean.extend(acc.tolist())
            val_bar.set_postfix({'val_loss': sum(loss_mean) / len(loss_mean), 'train_loss': train_loss,
                                 'acc': sum(acc_mean) / len(acc_mean)})
        val_loss = sum(loss_mean) / len(loss_mean)
        val_acc = sum(acc_mean) / len(acc_mean)
        if val_loss < min_loss:
            min_loss = val_loss
            if save:
                os.makedirs(save, exist_ok=True)
                with open(f'{save}/model_weights.pt', 'wb') as f:
                    torch.save(model.state_dict(), f)
                with open(f'{save}/simple_log.txt', 'w') as f:
                    f.write(f'epoch:{epoch}\n'
                            f'train_loss:{train_loss}\n'
                            f'val_loss:{val_loss}\n'
                            f'acc:{val_acc}')