#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: survival.py
@author:zyl
@contact:yilan.zhang@kaust.edu.sa
@time:12/23/24 11:56 AM
'''

import pandas as pd
import os
from timeit import default_timer as timer
from dataset.dataset_survival import SurvivalDatasetFactory
from utils.core_utils import _train_val
from utils.general_utils import _get_start_end, _prepare_for_experiment, _save_pkl

from utils.process_args import _process_args
from utils.visual_utils import process_results_km

import warnings
warnings.filterwarnings("ignore")

import torch, multiprocessing as mp
torch.multiprocessing.set_sharing_strategy("file_descriptor")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass



def main(args):

    #----> prep for 5 fold cv study
    folds = _get_start_end(args)

    #----> storing the val and test cindex for 5 fold cv
    all_val_cindex = []
    all_val_cindex_ipcw = []
    all_val_BS = []
    all_val_IBS = []
    all_val_iauc = []
    all_val_loss = []

    # ----> log
    if args.only_test:
        log_path = os.path.join(args.results_dir, 'log_test.txt')
    else:
        log_path = os.path.join(args.results_dir, 'log_start_{}_end_{}.txt'.format(args.k_start, args.k_end))

    log_file = open(log_path, 'w')
    dataset_factory = SurvivalDatasetFactory(
        study = args.study,
        data_path = args.data_path,
        rna_format= args.rna_format,
        signature= args.signature,
        n_bins= args.n_classes,
        label_col=args.label_col,
        num_genes=args.num_genes,
        num_patches=args.num_patches)

    for i in folds:
        args.max_cindex = 0.0
        args.max_cindex_epoch = 0.0
        print("Training fold {}".format(i))
        log_file.write("Training fold {}\n".format(i))
        results, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, val_loss) = _train_val(args, dataset_factory, i, log_file)
        # store the results
        filename = os.path.join(args.results_dir, 'split_{}_results_final.pkl'.format(i))
        print("Saving results...")
        _save_pkl(filename, results)

        all_val_cindex.append(val_cindex)
        all_val_cindex_ipcw.append(val_cindex_ipcw)
        all_val_BS.append(val_BS)
        all_val_IBS.append(val_IBS)
        all_val_iauc.append(val_iauc)
        all_val_loss.append(val_loss)

    log_file.close()

    final_df = pd.DataFrame({
        'folds': folds,
        'val_cindex': all_val_cindex,
        'val_cindex_ipcw': all_val_cindex_ipcw,
        # 'val_BS': all_val_BS,
        'val_IBS': all_val_IBS,
        'val_iauc': all_val_iauc,
        'val_loss': all_val_loss
    })

    # calculate mean and std for each row, except for the folds
    final_df.set_index('folds', inplace=True)
    final_df.loc['mean'] = final_df.mean()
    final_df.loc['std'] = final_df.std()
    print("The final results are: \n")
    print("The average cindex is: ", final_df.loc['mean']['val_cindex'])
    print("The std of cindex is: ", final_df.loc['std']['val_cindex'])

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(args.k_start, args.k_end)
    else:
        if args.only_test:
            save_name = 'summary_test.csv'
        else:
            save_name = 'summary.csv'

    final_df.to_csv(os.path.join(args.results_dir, save_name))


    # ----> process the results for Kaplan-Meier
    process_results_km(args.results_dir, args.k)


if __name__ == '__main__':
    start = timer()
    # ----> read the args
    args = _process_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args = _prepare_for_experiment(args)
    main(args)

    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))  # ----> pytorch imports
