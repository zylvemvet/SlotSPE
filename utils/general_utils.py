#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: general_utils.py
@author:zyl
@contact:yilan.zhang@kaust.edu.sa
@time:12/21/24 4:04 PM
'''

import os
import pandas as pd
import torch
import numpy as np
import pickle
import os


class TemperatureAnnealer:
    def __init__(self, start_temp=1.0, end_temp=0.01, anneal_rate=0.999, update_every=1):
        """
        start_temp: initial temperature
        end_temp: minimum temperature (stops decaying below this)
        anneal_rate: multiplicative decay rate per step (0.999 ~ slow decay)
        update_every: how often to decay (in steps)
        """
        self.temp = start_temp
        self.end_temp = end_temp
        self.anneal_rate = anneal_rate
        self.update_every = update_every
        self.step_count = 0

    def get(self):
        """Return current temperature"""
        return self.temp

    def step(self):
        """Decay temperature after every 'update_every' calls"""
        self.step_count += 1
        if self.step_count % self.update_every == 0:
            self.temp = max(self.temp * self.anneal_rate, self.end_temp)
        return self.temp



def _prepare_for_experiment(args):
    r"""
    Creates experiment code which will be used for identifying the experiment later on. Uses the experiment code to make results dir.
    Prints and logs the important settings of the experiment. Loads the pathway composition dataframe and stores in args for future use.

    Args:
        - args : argparse.Namespace

    Returns:
        - args : argparse.Namespace

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)
    # args.split_dir = os.path.join("dataset_csv",)
    args.split_dir = os.path.join(args.data_path, "splits", args.which_splits, args.study)
    # args.combined_study = args.study
    _seed_torch(args.seed,args.device)

    assert os.path.isdir(args.split_dir)
    print('Split dir:', args.split_dir)


    #---> store the settings
    settings = {'num_splits': args.k,
                'k_start': args.k_start,
                'k_end': args.k_end,
                'max_epochs': args.max_epochs,
                'results_dir': args.results_dir,
                'lr': args.lr,
                'study': args.study,
                'reg': args.reg,
                'label_col': args.label_col,
                'bag_loss': args.bag_loss,
                'seed': args.seed,
                'opt': args.opt,
                "num_patches": args.num_patches,
                "num_genes": args.num_genes,
                'split_dir': args.split_dir,
                'signature': args.signature,
                'rna_format': args.rna_format,
                'method': args.method,
                'slot_num_wsi': args.slot_num_wsi,
                'slot_num_omics': args.slot_num_omics,
                'slot_iters': args.slot_iters,
                'topk_ratio': args.topk_ratio,
                'top_k_method': args.top_k_method,
                }

    if not args.only_test:
        #---> custom experiment code
        args = _get_custom_exp_code(args)
        # ---> where to stroe the experiment related assets
        _create_results_dir(args)
        # ---> bookkeping
        _print_and_log_experiment(args, settings)
    else:
        _reading_experiment_settings(args)

    return args


def _print_and_log_experiment(args, settings):
    r"""
    Prints the expeirmental settings and stores them in a file

    Args:
        - args : argspace.Namespace
        - settings : dict

    Return:
        - None

    """

    with open(args.results_dir + '/experiment_settings.txt', 'w') as f:
        print(settings, file=f)

    f.close()

    print("")
    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))
    print("")


def _reading_experiment_settings(args):
    r"""
    Reads the experiment settings from the file and prints them

    Args:
        - args : argspace.Namespace

    Returns:
        - None

    """

    settings = {}
    with open(args.results_dir + '/experiment_settings.txt', 'r') as f:
        for line in f:
            print(line)
            # 去除“{}”和“\n”
            line = line.replace("{", "").replace("}", "").replace("\n", "")
            param = line.strip().split(",")
            for p in param:
                key, val = p.strip().split(":")
                settings[key.strip().replace("'", "")] = val.strip().replace("'", "")
                print(key, val)
    f.close()

    print("reading settings and resetting the args...")
    args.seed = int(settings['seed'])
    # args.sample_num = int(settings['sample_num'])


def _get_custom_exp_code(args):
    r"""
    Updates the argparse.NameSpace with a custom experiment code.

    Args:
        - args (NameSpace)

    Returns:
        - args (NameSpace)

    """
    param_code = ''

    # ----> Study
    # param_code += args.study

    param_code += str(args.lr)

    # ----> Batch Size
    param_code += '_b%s' % str(args.batch_size)

    # label col
    param_code += "_" + args.label_col

    param_code += "_Dim_" + str(args.wsi_projection_dim)
    param_code += "_e_" + str(args.max_epochs)
    param_code += "_g_" + str(args.rna_format)
    param_code += "_sig_" + str(args.signature)
    param_code += "_seed" + str(args.seed)
    param_code += "_rW_" + str(args.slot_num_wsi)
    param_code += "_rG_" + str(args.slot_num_omics)
    param_code += "_sp_" + str(args.specific_simple)

    # ----> Updating
    args.param_code = param_code

    return args


def _seed_torch(seed=7, device='cuda'):
    r"""
    Sets custom seed for torch

    Args:
        - seed : Int

    Returns:
        - None

    """
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _create_results_dir(args):
    r"""
    Creates a dir to store results for this experiment. Adds .gitignore

    Args:
        - args: argspace.Namespace

    Return:
        - None

    """
    args.results_dir = os.path.join(args.results_dir, args.study)  # create an experiment specific subdir in the results dir
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
        # ---> add gitignore to results dir
        f = open(os.path.join(args.results_dir, ".gitignore"), "w")
        f.write("*\n")
        f.write("*/\n")
        f.write("!.gitignore")
        f.close()

    args.results_dir = os.path.join(args.results_dir, args.method)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    # ---> results for this specific experiment
    args.results_dir = os.path.join(args.results_dir, args.param_code)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)


def _get_start_end(args):
    r"""
    Which folds are we training on

    Args:
        - args : argspace.Namespace

    Return:
       folds : np.array

    """
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    folds = np.arange(start, end)
    return folds

def _save_pkl(filename, save_object):
    writer = open(filename,'wb')
    pickle.dump(save_object, writer)
    writer.close()

def _load_pkl(filename):
    loader = open(filename,'rb')
    file = pickle.load(loader)
    loader.close()
    return file


def _print_network(results_dir, net):
    r"""

    Print the model in terminal and also to a text file for storage

    Args:
        - results_dir : String
        - net : PyTorch model

    Returns:
        - None

    """
    num_params = 0
    num_params_train = 0

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)

    # print(net)

    fname = "model_parameters.txt"
    path = os.path.join(results_dir, fname)
    f = open(path, "w")
    f.write(str(net))
    f.write("\n")
    f.write('Total number of parameters: %d \n' % num_params)
    f.write('Total number of trainable parameters: %d \n' % num_params_train)
    f.close()
