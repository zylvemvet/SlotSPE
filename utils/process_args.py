#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: process_args.py
@author:zyl
@contact:yilan.zhang@kaust.edu.sa
@time:12/23/24 4:21 PM
'''

import argparse

def _process_args():
    r"""
    Function creates a namespace to read terminal-based arguments for running the experiment

    Args
        - None

    Return:
        - args : argparse.Namespace

    """

    parser = argparse.ArgumentParser(description='Configurations for SurvPath Survival Prediction Training')

    #---> study related
    parser.add_argument('--study', type=str, default='coadread',help='study name')
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes (4 bins for survival)')
    parser.add_argument('--results_dir', default="/home/zhany0x/Documents/experiment/Survival/Rebuttal/", help='results directory (default: ./results)')
    parser.add_argument('--specific_simple', default="", help='specific simple name')

    #----> data related
    parser.add_argument('--data_root_dir', type=str, default="/Data/Pathology/UNI/coadread/pt_files/", help='data directory')
    parser.add_argument('--data_path', type=str, default="./dataset_csv", help='Path to csv with labels')
    parser.add_argument('--num_patches', type=int, default=4096, help='number of patches')
    parser.add_argument('--num_genes', type=int, default=None, help='number of genes when rna_format is RNASeq or GeneEmbedding')
    parser.add_argument('--label_col', type=str, default="survival_months_dss", help='type of survival (OS, DSS, PFI)')
    parser.add_argument('--rna_format', type=str, default="Pathways", choices=["RNASeq", "Pathways", "GeneEmbedding"],
                        help='format of omics data')
    parser.add_argument("--signature", type=str, default="combine",choices=["all", "six", "hallmarks", "combine", "xena"])

    #----> split related
    parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--which_splits', type=str, default="5fold", help='where are splits')

    #----> training related
    parser.add_argument('--max_epochs', type=int, default=30, help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=3, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--opt', type=str, default="adam", help="Optimizer")
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--bag_loss', type=str, choices=["nll_surv", "rank_surv", "cox_surv","sinkhorn_surv"], default="nll_surv",
                        help='survival loss function (default: ce)')
    parser.add_argument('--alpha_surv', type=float, default=0.5, help='weight given to uncensored patients')
    parser.add_argument('--reg', type=float, default=1e-3, help='weight decay / L2 (default: 1e-5)')
    parser.add_argument('--max_cindex', type=float, default=0.0, help='maximum c-index')

    #---> model related
    # parser.add_argument('--fusion', type=str, default='concat', choices=["concat", "bilinear","lrb"], help='fusion method')
    parser.add_argument('--method', type=str, default="SlotSPE",choices=['SlotSPE'], help='method type')
    parser.add_argument('--encoding_dim', type=int, default=1024, help='WSI encoding dim (1024 for resnet50, 768 for swin)')
    parser.add_argument('--wsi_projection_dim', type=int, default=256, help="projection dim of features")

    # loss related
    parser.add_argument('--lambda_recon_loss', type=float, default=0.01, help="lambda for reconstruction loss")

    # lr_scheduler
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine','step'], help='lr scheduler')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='minimum learning rate for cosine scheduler')
    parser.add_argument('--step_size', type=int, default=10, help='step size for step scheduler')

    #---> gpu
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')

    #---> only test the model
    parser.add_argument('--only_test', action='store_true', default=False, help='only test')
    parser.add_argument('--omic_missing', action='store_true', default=False, help='omic missing')

    #---> for slot numbers
    parser.add_argument('--slot_num_wsi', type=int, default=8, help='number of slots')
    parser.add_argument('--slot_num_omics', type=int, default=8, help='number of slots')
    parser.add_argument('--slot_iters', type=int, default=10, help='number of slot attention iterations')

    # ---> for gating mechanism, temperature=args.temperature
    parser.add_argument('--temperature', type=float, default=0.01, help='temperature for gating mechanism')
    # topk_ratio=0.25
    parser.add_argument('--topk_ratio', type=float, default=0.25, help='topk ratio for selecting important patches/features')
    parser.add_argument('--top_k_method', type=str, default='parallel_topk_st', choices=['gumbel_topk_st', 'parallel_topk_st'], help='topk method for selecting important patches/features')


    args = parser.parse_args()

    return args
