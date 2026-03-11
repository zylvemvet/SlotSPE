#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: model_utils.py
@author:zyl
@contact:yilan.zhang@kaust.edu.sa
@time:12/27/24 3:41 PM
'''


from models.SlotSPE import SlotSPE
from utils.general_utils import _print_network
import torch


def _init_model(args, dataset_factory):
    print('\nInit model...', end=' ')
    if args.rna_format == "RNASeq":
        if dataset_factory.num_genes is not None:
            omics_input_dim = dataset_factory.num_genes
        else:
            omics_input_dim = dataset_factory.omic_sizes
    elif args.rna_format == "gene_embeddings":
        omics_input_dim = 768
    else:
        omics_input_dim = None

    args.omic_sizes = dataset_factory.omic_sizes
    args.omic_names = dataset_factory.omic_names

    if args.method == "SlotSPE":
        model_dict = {'args': args,
                      'omic_input_dim': omics_input_dim, }
        model = SlotSPE(**model_dict)

    else:
        raise ValueError(f"Method {args.method} not implemented")

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))

    print('Done!')
    _print_network(args.results_dir, model)

    return model

