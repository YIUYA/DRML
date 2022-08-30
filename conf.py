import argparse
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import os
import argparse
import pprint
import numpy as np
import torch
import random
import logging
import shutil
import yaml
from datetime import datetime 


parser = argparse.ArgumentParser(description='PyTorch Training')

"""
Create argument to allow change config value in command line 
"""
parser.add_argument('--dataset', default="DISFA+", type=str, help="experiment dataset DISFA+")


def parser2dict():
    config, unparsed = parser.parse_known_args()
    cfg = edict(config.__dict__)
    return edict(cfg)



def print_conf(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        # default = self.parser.get_default(k)
        # if v != default:
        #     comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    return message



def get_config():

    # args from argparser
    cfg = parser2dict()
    if cfg.dataset == 'DISFA+':
        #with open('config/DISFAplus.yaml', 'r') as f:
        with open('E:/au_detection/DRML/config/DISFAplus.yaml', 'r') as f:
            datasets_cfg = yaml.load(f, Loader=yaml.FullLoader)
            datasets_cfg = edict(datasets_cfg)

    else:
        raise Exception("Unkown Datsets:",cfg.dataset)

    cfg.update(datasets_cfg)
    return cfg



def set_env(cfg):
    # set seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    if 'cudnn' in cfg:
        torch.backends.cudnn.benchmark = cfg.cudnn
    else:
        torch.backends.cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids

def set_outdir(conf):
    default_outdir = 'results'
    if 'timedir' in conf:
        timestr = datetime.now().strftime('%d-%m-%Y_%I_%M-%S_%p')
        outdir = os.path.join(default_outdir,conf.exp_name,timestr)
    else:
        outdir = os.path.join(default_outdir,conf.exp_name)
        prefix = 'bs_'+str(conf.batch_size)+'_seed_'+str(conf.seed)+'_lr_'+str(conf.learning_rate)
        outdir = os.path.join(outdir,prefix)
    ensure_dir(outdir)
    conf['outdir'] = outdir
    return conf

# check if dir exist, if not create new folder
def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('{} is created'.format(dir_name))



def set_logger(cfg):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """

    if 'loglevel' in cfg:
        loglevel = eval('logging.'+loglevel)
    else:
        loglevel = logging.INFO


    print(cfg.evaluate)
    if cfg.evaluate:
        outname = 'test.log'
    else:
        outname = 'train.log'

    outdir = cfg['outdir']
    log_path = os.path.join(outdir,outname)

    logger = logging.getLogger()
    logger.setLevel(loglevel)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

    logging.info(print_conf(cfg))
    logging.info('writting logs to file {}'.format(log_path))
