import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from conf import get_config,set_logger,set_outdir,set_env
from numba import cuda
from model.DRML import DRML
from dataset import DISFA 
from utils import statistics, update_statistics_list, calc_f1_score, calc_acc, DISFA_infolist, load_state_dict
from conf import get_config,set_env, set_outdir,set_logger 

def get_dataloader(conf):

    valset = DISFA(conf.dataset_path, train=False,crop_size=conf.crop_size)
    val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False)

    return val_loader, len(valset)


# Val
def val(net, val_loader):

    net.eval()
    statistics_list = None
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        targets = targets.float()
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            update_list = statistics(outputs, targets.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return mean_f1_score, f1_score_list, mean_acc, acc_list


def main(conf):

    dataset_info = DISFA_infolist

    # data
    val_loader, val_data_num = get_dataloader(conf)
    logging.info("val_data_num: {} ]".format(val_data_num))
    net = DRML(len(conf.action_units))

    if torch.cuda.is_available():
        net = nn.DataParallel(net.cuda())

    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)



    #test
    val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val(net, val_loader)

    # log
    infostr = {'val_mean_f1_score {:.2f} val_mean_acc {:.2f}' .format(100.* val_mean_f1_score, 100.* val_mean_acc)}
    logging.info(infostr)
    infostr = {'F1-score-list:'}
    logging.info(infostr)
    infostr = dataset_info(val_f1_score, conf.action_units)
    logging.info(infostr)
    infostr = {'Acc-list:'}
    logging.info(infostr)
    infostr = dataset_info(val_acc, conf.action_units)
    logging.info(infostr)



# ---------------------------------------------------------------------------------


if __name__=="__main__":

    torch.cuda.empty_cache()
    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)
    conf = get_config()
    conf.evaluate = True
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)

