import numpy as np
import time, json, os
import torch
import torch.nn as nn
from tqdm import tqdm
import logging

def get_nb_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def train(device, model, train_loader, optimizer, scheduler, reg=1, pos_norm=0, norm_norm=0, out_norm=1, pos_mean=None, pos_std=None, norm_mean=None, norm_std=None, out_mean=None, out_std=None, full=False):
    model.train()

    criterion_func = nn.MSELoss(reduction='none')
    losses_mse = []
    for x, y, pos, geom, edge in train_loader:
        x = x.to(device)
        pos = pos.to(device)
        if pos_norm:
            pos = (pos - pos_mean) / pos_std
            x[:, :, :3] = pos
        y = y.to(device)
        geom = geom.to(device)
        optimizer.zero_grad()
        out = model((x, pos, geom))

        if out_norm:
            y = (y - out_mean) / (out_std + 1e-6)
        
        loss_press = criterion_func(out, y).mean()
        
        total_loss = loss_press

        total_loss.backward()
        
        # clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        losses_mse.append(loss_press.item())

    return np.mean(losses_mse)


@torch.no_grad()
def test(device, model, test_loader, pos_norm=0, norm_norm=1, out_norm=1, pos_mean=None, pos_std=None, norm_mean=None, norm_std=None, out_mean=None, out_std=None, full=False):
    model.eval()

    criterion_func = nn.MSELoss(reduction='none')
    losses_mse = []
    losses_l2re = []
    
    for x, y, pos, geom, edge in test_loader:
        # 计时
        x = x.to(device)
        pos = pos.to(device)
        if pos_norm:
            pos = (pos - pos_mean) / pos_std
            x[:, :, :3] = pos
        y = y.to(device)
        if geom is not None:
            geom = geom.to(device)
        out = model((x, pos, geom))
        
        if out_norm:
            y_norm = (y - out_mean) / (out_std + 1e-6) # normalize label
            loss_mse = criterion_func(out, y_norm).mean()
            out = out * out_std + out_mean # denormalize output
            loss_l2re = torch.norm(out[:, :, -1] - y[:, :, -1]) / torch.norm(y[:, :, -1]) #l2re after denormalization
        else:
            loss_mse = criterion_func(out[:, :, -1], y[:, :, -1]).mean()
            loss_l2re = torch.norm(out[:, :, -1] - y[:, :, -1]) / torch.norm(y[:, :, -1])
        losses_mse.append(loss_mse.item())
        losses_l2re.append(loss_l2re.item())

    return np.mean(losses_mse), np.mean(losses_l2re)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(device, train_loader, val_loader, Net, hparams, path, reg=1, val_iter=1, pos_norm=0, out_norm=1, norm_norm=0, pos_mean=None, pos_std=None, out_mean=None, out_std=None, norm_mean=None, norm_std=None, full=False):
    model = Net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=int((len(train_loader) // hparams['batch_size'] + 1) * hparams['nb_epochs']),
        final_div_factor=1000.,
    )
    start = time.time()

    train_loss, val_loss_mse, val_loss_l2re = 1e5, 1e5, 1e5
    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    cnt = 0
    for epoch in pbar_train:
        loss_mse = train(device, model, train_loader, optimizer, lr_scheduler, reg=reg, pos_norm=pos_norm, out_norm=out_norm, norm_norm=norm_norm, pos_mean=pos_mean, pos_std=pos_std, out_mean=out_mean, out_std=out_std, norm_mean=norm_mean, norm_std=norm_std, full=full)
        train_loss = loss_mse

        if val_iter is not None and (epoch == hparams['nb_epochs'] - 1 or epoch % val_iter == 0):
            loss_mse, loss_l2re = test(device, model, val_loader, pos_norm=pos_norm, out_norm=out_norm, norm_norm=norm_norm, pos_mean=pos_mean, pos_std=pos_std, out_mean=out_mean, out_std=out_std, norm_mean=norm_mean, norm_std=norm_std, full=full)
            val_loss_mse = loss_mse
            val_loss_l2re = loss_l2re

            pbar_train.set_postfix(train_loss=train_loss, val_loss_mse=val_loss_mse, val_loss_l2re=val_loss_l2re)
            print(f"Epoch {epoch} train loss: {train_loss}, val loss mse: {val_loss_mse}, val loss l2re: {val_loss_l2re}")
            logging.info(f'Epoch {epoch}, train_loss: {train_loss}, val_loss_mse: {val_loss_mse}, val_loss_l2re: {val_loss_l2re}')
        else:
            pbar_train.set_postfix(train_loss=train_loss)
            print(f"Epoch {epoch} train loss: {train_loss}")
            logging.info(f'Epoch {epoch}, train_loss: {train_loss}')
        
        if (cnt + 1) % 50 == 0:
            torch.save(model, path + os.sep + f'model_{epoch}.pth')
        cnt += 1

    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
    logging.info(f'Number of parameters: {params_model}')
    logging.info(f'Time elapsed: {time_elapsed} seconds')
    torch.save(model, path + os.sep + f'model_{hparams["nb_epochs"]}.pth')

    if val_iter is not None:
        with open(path + os.sep + f'log_{hparams["nb_epochs"]}.json', 'a') as f:
            json.dump(
                {
                    'nb_parameters': params_model,
                    'time_elapsed': time_elapsed,
                    'hparams': hparams,
                    'train_loss': train_loss,
                    'val_loss_mse': val_loss_mse,
                    'val_loss_l2re': val_loss_l2re,
                }, f, indent=12, cls=NumpyEncoder
            )

    return model
