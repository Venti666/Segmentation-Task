import time
import os
import logging
from tqdm import tqdm

from utils import unet_dataset
from models import unetFEGcn
from metrics import eval_metrics
from lr_schedule import step_lr, exp_lr_scheduler

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(config):
    device = torch.device('cuda:0')
    selected = config['train_model']['model'][config['train_model']['select']]
    if selected == 'unetFEGcn':
        model = unetFEGcn.UNet(num_classes=config['num_classes'])

    model.to(device)
    logger = initLogger(selected)
    criterion = nn.CrossEntropyLoss()

    # Data loading
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.209, 0.394, 0.380, 0.344, 0.481], std=[0.141, 0.027, 0.032, 0.046, 0.069])
    ])
    dst_train = unet_dataset.UnetDataset(config['train_list'], transform=transform, train=True)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=config['batch_size'])
    dst_valid = unet_dataset.UnetDataset(config['test_list'], transform=transform, train=False)
    dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=config['batch_size'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=[config['momentum'], 0.999], weight_decay=config['weight_decay'])

    # Initialize tracking variables
    best_val_loss = float('inf')
    best_mIoU = 0.0
    best_epoch_loss = -1
    best_epoch_mIoU = -1

    for epoch in range(config['num_epoch']):
        epoch_start = time.time()
        model.train()
        loss_sum = 0.0
        correct_sum = 0.0
        labeled_sum = 0.0
        inter_sum = 0.0
        unoin_sum = 0.0
        conf_matrix_train = np.zeros((config['num_classes'], config['num_classes']))

        # Training loop
        tbar = tqdm(dataloader_train, ncols=120)
        for batch_idx, (data, target, path) in enumerate(tbar):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

            correct, labeled, inter, unoin, conf_matrix_train = eval_metrics(output, target, config['num_classes'], conf_matrix_train)
            correct_sum += correct
            labeled_sum += labeled
            inter_sum += inter
            unoin_sum += unoin
            pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
            IoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)
            tbar.set_description('TRAIN ({}) | Loss: {:.5f} | OA {:.5f} mIoU {:.5f} | bt {:.2f} et {:.2f}|'.format(
                epoch, loss_sum/((batch_idx+1)*config['batch_size']), pixelAcc, IoU.mean(),
                time.time()-epoch_start, time.time()-epoch_start))

        logger.info('TRAIN ({}) | Loss: {:.5f} | OA {:.5f} IOU {} mIoU {:.5f}'.format(
            epoch, loss_sum/((batch_idx+1)*config['batch_size']), pixelAcc, toString(IoU), IoU.mean()))

        # Validation loop
        model.eval()
        val_loss_sum = 0.0
        val_correct_sum = 0.0
        val_labeled_sum = 0.0
        val_inter_sum = 0.0
        val_unoin_sum = 0.0
        conf_matrix_val = np.zeros((config['num_classes'], config['num_classes']))
        class_precision = np.zeros(config['num_classes'])
        class_recall = np.zeros(config['num_classes'])
        class_f1 = np.zeros(config['num_classes'])

        with torch.no_grad():
            tbar = tqdm(dataloader_valid, ncols=120)
            for batch_idx, (data, target, path) in enumerate(tbar):
                output = model(data)
                loss = criterion(output, target)
                val_loss_sum += loss.item()

                correct, labeled, inter, unoin, conf_matrix_val = eval_metrics(output, target, config['num_classes'], conf_matrix_val)
                val_correct_sum += correct
                val_labeled_sum += labeled
                val_inter_sum += inter
                val_unoin_sum += unoin

                for i in range(config['num_classes']):
                    class_precision[i] = 1.0 * conf_matrix_val[i,i] / conf_matrix_val[:,i].sum()
                    class_recall[i] = 1.0 * conf_matrix_val[i,i] / conf_matrix_val[i].sum()
                    class_f1[i] = (2.0 * class_precision[i] * class_recall[i]) / (class_precision[i] + class_recall[i])

                val_pixelAcc = 1.0 * val_correct_sum / (np.spacing(1) + val_labeled_sum)
                val_mIoU = 1.0 * val_inter_sum / (np.spacing(1) + val_unoin_sum)
                tbar.set_description('VAL ({}) | Loss: {:.5f} | Acc {:.5f} mIoU {:.5f}'.format(
                    epoch, val_loss_sum/((batch_idx+1)*config['batch_size']), val_pixelAcc, val_mIoU.mean()))

        # Save models based on both criteria
        if not os.path.exists(config['save_model']['save_path']):
            os.makedirs(config['save_model']['save_path'])

        current_val_loss = val_loss_sum / ((batch_idx + 1) * config['batch_size'])
        current_mIoU = val_mIoU.mean()

        # Save best loss model
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch_loss = epoch
            torch.save(model.state_dict(), os.path.join(config['save_model']['save_path'], f'{selected}_best_loss.pth'))
            np.savetxt(os.path.join(config['save_model']['save_path'], f'{selected}_best_loss_conf_matrix.txt'), conf_matrix_val, fmt="%d")
            np.savetxt(os.path.join(config['save_model']['save_path'], f'{selected}_best_loss_epoch.txt'), [epoch, conf_matrix_val.sum()])

        # Save best mIoU model
        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            best_epoch_mIoU = epoch
            torch.save(model.state_dict(), os.path.join(config['save_model']['save_path'], f'{selected}_best_mIoU.pth'))
            np.savetxt(os.path.join(config['save_model']['save_path'], f'{selected}_best_mIoU_conf_matrix.txt'), conf_matrix_val, fmt="%d")
            np.savetxt(os.path.join(config['save_model']['save_path'], f'{selected}_best_mIoU_epoch.txt'), [epoch, conf_matrix_val.sum()])

        logger.info('VAL ({}) | Loss: {:.5f} (Best: {:.5f} @ {}) | mIoU: {:.5f} (Best: {:.5f} @ {})'.format(
            epoch, current_val_loss, best_val_loss, best_epoch_loss,
            current_mIoU, best_mIoU, best_epoch_mIoU))

def toString(IOU):
    result = '{'
    for i, num in enumerate(IOU):
        result += f'{i}: {num:.4f}, '
    return result[:-2] + '}'

def initLogger(model_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = 'logs'
    os.makedirs(log_path, exist_ok=True)
    log_name = os.path.join(log_path, f"new_{model_name}_jx_new_metrics_{rq}.log")
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

if __name__ == '__main__':
    # train()
    pass