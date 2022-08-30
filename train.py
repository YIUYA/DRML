import os
from pickletools import optimize
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
from model.DRML import DRML
from dataset import DISFA 
from utils import statistics, AverageMeter, update_statistics_list, calc_f1_score, calc_acc, DISFA_infolist, multi_label_sigmoid_cross_entropy_loss
from conf import get_config,set_env, set_outdir,set_logger 
def get_dataloader(conf):
    print('==> Preparing data...')

    trainset = DISFA(conf.dataset_path, train=True, crop_size=conf.crop_size)
    train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True)
    valset = DISFA(conf.dataset_path, train=False, crop_size=conf.crop_size)
    val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False)

    return train_loader, val_loader, len(trainset), len(valset)

def train(conf, net, train_loader, optimizer, epoch, criterion):

    net.train()
    losses = AverageMeter()
    train_loader_len = len(train_loader)
    for batch_idx, (inputs,  targets) in enumerate(tqdm(train_loader)):
        targets = targets.float()
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        else: 
            assert False, 'gpu no available'
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        #print(f'loss: {batch_idx}, {loss}')
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))
       
    return losses.avg

# Val
def val(net, val_loader, criterion):
    losses = AverageMeter()
    net.eval()
    statistics_list = None
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        targets = targets.float()
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.data.item(), inputs.size(0))
            update_list = statistics(outputs, targets.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return losses.avg, mean_f1_score, f1_score_list, mean_acc, acc_list

   
def main(conf):

    dataset_info = DISFA_infolist

    start_epoch = 0
    # data
    train_loader,val_loader,train_data_num,val_data_num = get_dataloader(conf)


    net = DRML(12)
    
    print('gpu, ',  torch.cuda.is_available())

    if torch.cuda.is_available():
        net.cuda()
    assert torch.cuda.is_available(), 'cuda is not avaliable'
    criterion = multi_label_sigmoid_cross_entropy_loss()
    #optimizer = optim.AdamW(net.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    optimizer = optim.SGD(net.parameters(), lr=conf.learning_rate, momentum=conf.momentum, weight_decay=conf.weight_decay)
    print('the init learning rate is ', conf.learning_rate)

    #train and val
    best = 0
    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        train_loss = train(conf,net,train_loader,optimizer,epoch,criterion)
        val_loss, val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val(net, val_loader, criterion)

        # log
        infostr = {'Epoch:  {}   train_loss: {:.5f}  val_loss: {:.5f}  val_mean_f1_score {:.2f},val_mean_acc {:.2f}'
                .format(epoch + 1, train_loss, val_loss, 100.* val_mean_f1_score, 100.* val_mean_acc)}
        logging.info(infostr)
        infostr = {'F1-score-list:'}
        logging.info(infostr)
        infostr = dataset_info(val_f1_score, conf.action_units)
        logging.info(infostr)
        infostr = {'Acc-list:'}
        logging.info(infostr)
        infostr = dataset_info(val_acc,  conf.action_units)
        logging.info(infostr)

        # save checkpoints
        if (epoch+1) % 4 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '.pth'))


        if val_mean_f1_score > best:
            best = val_mean_f1_score
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(conf['outdir'], 'best_py' + str(epoch + 1) + '.pth'))


# ---------------------------------------------------------------------------------


if __name__=="__main__":
    conf = get_config()
    print(conf)
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)



