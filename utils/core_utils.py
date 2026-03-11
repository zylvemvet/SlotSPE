#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: core_utils_rebuttal.py
@author:zyl
@contact:yilan.zhang@kaust.edu.sa
@time:12/21/24 3:56 PM
'''

from ast import Lambda
import numpy as np
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc, brier_score, integrated_brier_score
from sksurv.util import Surv
from utils.general_utils import _save_pkl
from utils.loss_func import NLLSurvLoss, SurvPLE, RankLoss, SinkhornSurvLoss
import torch.optim as optim
import torch
from dataset.dataset_survival import SurvivalDataset, _collate_pathways
import os
from utils.model_utils import _init_model
import torch.nn.functional as F
import gc

def free_loader(loader):
    if loader is None:
        return
    # stop worker processes if an iterator exists (private API but effective)
    try:
        it = getattr(loader, "_iterator", None)
        if it is not None:
            it._shutdown_workers()
    except Exception:
        pass
    # drop references and run GC
    del loader
    gc.collect()

def _get_split(args, dataset_factory, cur):
    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    train_data = SurvivalDataset(dataset_factory, args.data_root_dir, 'train', cur, args.encoding_dim)
    test_data = SurvivalDataset(dataset_factory, args.data_root_dir, 'val', cur, args.encoding_dim)
    if args.rna_format == "Pathways" or args.rna_format == "RankedGenes":
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn=_collate_pathways, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, collate_fn=_collate_pathways, pin_memory=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True,pin_memory=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0,pin_memory=False)
    print('Done!')
    print("Training on {} samples".format(len(train_data)))
    print("Validating on {} samples".format(len(test_data)))

    return train_data, test_data, train_loader, test_loader


def _init_loss_function(args):
    r"""
    Init the survival loss function

    Args:
        - args : argspace.Namespace

    Returns:
        - loss_fn : NLLSurvLoss or NLLRankSurvLoss

    """
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox_surv':
        loss_fn = SurvPLE() 
    elif args.bag_loss == 'rank_surv':
        loss_fn = RankLoss()
    elif args.bag_loss == 'sinkhorn_surv':
        loss_fn = SinkhornSurvLoss(alpha=args.alpha_surv)
    else:
        raise NotImplementedError
    print('Done!')
    return loss_fn


def _init_optim(args, model):
    r"""
    Init the optimizer

    Args:
        - args : argspace.Namespace
        - model : torch model

    Returns:
        - optimizer : torch optim
    """
    print('\nInit optimizer ...', end='\n')

    if args.opt == "adam":

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "lamb":
        optimizer = Lambda(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError

    return optimizer

def _init_scheduler(args, optimizer):
    r"""
    Init the scheduler

    Args:
        - args : argspace.Namespace
        - optimizer : torch optim

    Returns:
        - scheduler : torch.optim.lr_scheduler
    """
    print('\nInit scheduler ...', end='\n')

    if args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.eta_min)
    else:
        raise NotImplementedError

    return scheduler

def _extract_survival_metadata(dataset_factory):

    all_censorships = dataset_factory.clinical_df[[dataset_factory.censorship_var]]
    #dataframe to numpy array
    all_censorships = all_censorships.to_numpy().flatten()

    all_event_times = dataset_factory.clinical_df[[dataset_factory.label_col]]
    #dataframe to numpy array
    all_event_times = all_event_times.to_numpy().flatten()

    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)

    return all_survival
    
def _unpack_data(data, device, omics_format):
    # [img, omic_data_list, label, event_time, c]
    data_wsi = data[0].to(device)
    
    if omics_format == "Pathways" or omics_format == "RankedGenes":
        data_omics = [] # TODO: check
        for idx,item in enumerate(data[1]):
            for idy,omic in enumerate(item):
                omic = omic.to(device)
                omic = omic.unsqueeze(0)
                if idx == 0:
                    data_omics.append(omic)
                else:
                    data_omics[idy] = torch.cat((data_omics[idy],omic),dim=0)
    else:
        data_omics = data[1].to(device)
    
    y_disc = data[2].to(device)
    event_time = data[3].to(device)
    c = data[4].to(device)

    return data_wsi, data_omics, y_disc, event_time, c

def _process_data_and_forward(args, model, data, device, test=False):
    data_wsi, data_omics, y_disc, event_time, c = _unpack_data(data, device, args.rna_format)
    
    input_args = {"x_wsi": data_wsi}

    input_args["cur_epoch"] = args.cur_epoch
    input_args['omic_missing'] = False
    if test:
        input_args['y'] = None
        input_args['c'] = None

        input_args['omic_missing'] = args.omic_missing
    else:
        input_args['y'] = y_disc
        input_args['c'] = c


    if args.rna_format == "Pathways" or args.rna_format == "RankedGenes":
        for i in range(len(data_omics)):
            input_args['x_omic%s' % str(i+1)] = data_omics[i]

        out = model(**input_args)
    else:
        input_args['x_omics'] = data_omics
        out = model(**input_args)

    return out, y_disc, event_time, c


def _calculate_risk(h):
    hazards = torch.sigmoid(h) # h: the output of the model
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()


def _update_arrays(all_risk_scores, all_censorships, all_event_times, event_time, censor, risk, clinical_data_list):

    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    return all_risk_scores, all_censorships, all_event_times

def _train_loop_survival(args, epoch, model,loader, optimizer, scheduler, loss_fn, log_file):
    if args.opt == "adam_seperate":
        optimizer_club = optimizer[1]
        optimizer = optimizer[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    args.cur_epoch = epoch


    accumulation_steps = 1
    # one epoch
    for batch_idx, data in enumerate(loader):

        h, y_disc, event_time, c = _process_data_and_forward(args, model, data, device)

        if args.method.startswith("SlotSPE"):
            logits, slot_loss = h
        else:
            raise ValueError(f"Method {args.method} not implemented")

        if args.bag_loss == "cox_surv":
            loss_surv = loss_fn(logits, event_time, c) #y_hat, T, E
        else:
            loss_surv = loss_fn(logits, y_disc, event_time, c)
        loss_surv = loss_surv / y_disc.shape[0]

        if args.method.startswith("SlotSPE"):
            loss = loss_surv + slot_loss
        else:
            raise ValueError(f"Method {args.method} not implemented")

        loss = loss / accumulation_steps
        loss.backward()


        if args.batch_size != 1:
            optimizer.step()
            optimizer.zero_grad()

        else:
            # accumulate gradients, only for batch_size ==1
            accumulation_steps = 32
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()


        total_loss += loss.item()
        risk, _ = _calculate_risk(logits)
        all_risk_scores, all_censorships, all_event_times = _update_arrays(all_risk_scores, all_censorships,
                                                                           all_event_times, event_time, c, risk, data)

        if batch_idx % accumulation_steps == 0:
            print('batch:{}, loss:{:.4f}, loss_surv: {:.4f}'.format(batch_idx, loss.item(), loss_surv.item()))
            log_file.write('batch:{}, loss:{:.4f}, loss_surv: {:.4f}\n'.format(batch_idx, loss.item(), loss_surv.item()))

    scheduler.step()

    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores,tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))
    log_file.write('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}\n'.format(epoch, total_loss, c_index))

    return


def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times,
                       all_risk_by_bin_scores):

    data = loader.dataset.label_df[dataset_factory.label_col]
    bins_original = dataset_factory.bins
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    # ---> delete the nans and corresponding elements from other arrays
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    # <---

    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times,
                                         all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.

    # change the datatype of survival test to calculate metrics
    try:
        survival_test = Surv.from_arrays(event=(1 - all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS, iauc

    # cindex2 (cindex_ipcw)
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.

    # brier score
    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores,
                            times=which_times_to_eval_at)
    except:
        print('An error occured while computing BS')
        BS = 0.

    # IBS
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores,
                                     times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.

    # iauc
    try:
        _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1 - all_risk_by_bin_scores[:, 1:],
                                         times=which_times_to_eval_at[1:])
    except:
        print('An error occured while computing iauc')
        iauc = 0.

    return c_index, c_index_ipcw, BS, IBS, iauc


def _summary(args, dataset_factory, model, loader, loss_fn, survival_train=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.
    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_logits = []
    all_case_ids = []

    case_ids = loader.dataset.label_df["case id"]
    count = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            h, y_disc, event_time, c = _process_data_and_forward(args, model, data, device, test=True)

            if args.method.startswith("SlotSPE"):
                logits, _ = h
            else:
                raise ValueError(f"Method {args.method} not implemented")

            if args.bag_loss == "cox_surv":
                loss = loss_fn(logits, event_time, c)  # y_hat, T, E
            elif args.bag_loss == "nll_surv":
                loss = loss_fn(logits, y_disc, event_time, c)
            else:
                raise ValueError(f"Loss function {args.bag_loss} not implemented")

            total_loss += loss.item()
            risk, risk_by_bin = _calculate_risk(logits)
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times = _update_arrays(all_risk_scores, all_censorships,
                                                                               all_event_times, event_time, c, risk, data)
            all_logits.append(logits.detach().cpu().numpy())
            all_case_ids.append(case_ids.values[count])
            count += 1

    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)


    patient_results = {}
    for i in range(len(all_case_ids)):
        case_id = all_case_ids[i]
        patient_results[case_id] = {
            "risk": all_risk_scores[i],
            "censor": all_censorships[i],
            "time": all_event_times[i],
            "logits": all_logits[i]
        }

    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores,
                                                          all_censorships, all_event_times, all_risk_by_bin_scores)

    return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss

def _save_results(cur, results_dict, args):
    filename = os.path.join(args.results_dir, "split_{}_results.pkl".format(cur))
    if os.path.exists(filename):
        os.remove(filename)
    print("Saving results...")
    _save_pkl(filename, results_dict)


def _step(args, cur, loss_fn, model, dataset_factory, optimizer, scheduler, train_loader, val_loader, log_file):
    all_survival = _extract_survival_metadata(dataset_factory)

    for epoch in range(args.max_epochs):
        _train_loop_survival(args, epoch, model, train_loader, optimizer, scheduler, loss_fn, log_file)
        results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, val_loss = _summary(args, dataset_factory, model, val_loader, loss_fn, all_survival)
        print(
            'Epoch:{} Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}'.format(
                epoch,
                val_cindex,
                val_cindex_ipcw,
                val_IBS,
                val_iauc
            ))
        log_file.write(
            'Epoch:{} Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}\n'.format(
                epoch,
                val_cindex,
                val_cindex_ipcw,
                val_IBS,
                val_iauc
            ))

        if val_cindex >= args.max_cindex:
            args.max_cindex = val_cindex
            args.max_cindex_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.results_dir, "model_best_s{}.pth".format(cur)))
            _save_results(cur, results_dict, args)

    # save the trained model
    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pth".format(cur)))

    results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, val_loss = _summary(args,
                                                                                                dataset_factory,
                                                                                                model,
                                                                                                val_loader, loss_fn,
                                                                                                all_survival)

    print(
        'Final Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}'.format(
            val_cindex,
            val_cindex_ipcw,
            val_IBS,
            val_iauc
        ))
    log_file.write(
        'Final Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}\n'.format(
            val_cindex,
            val_cindex_ipcw,
            val_IBS,
            val_iauc
        ))

    best_model = torch.load(os.path.join(args.results_dir, "model_best_s{}.pth".format(cur)))
    model.load_state_dict(best_model)
    _, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, val_loss = _summary(args, dataset_factory,
                                                                                     model, val_loader, loss_fn,
                                                                                     all_survival)
    print(
        'Best Val c-index: {:.4f} | Best Val c-index2: {:.4f} | Best Val IBS: {:.4f} | Best Val iauc: {:.4f}'.format(
            val_cindex,
            val_cindex_ipcw,
            val_IBS,
            val_iauc
        ))
    log_file.write(
        'Best Val c-index: {:.4f} | Best Val c-index2: {:.4f} | Best Val IBS: {:.4f} | Best Val iauc: {:.4f}\n'.format(
            val_cindex,
            val_cindex_ipcw,
            val_IBS,
            val_iauc
        ))

    return results_dict, (args.max_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, val_loss)


def _train_val(args, dataset_factory, cur, log_file):
    # ---> get the splits and summarize the data
    train_data, test_data, train_loader, test_loader = _get_split(args, dataset_factory, cur)
    # ---> init the model, loss function and optimizer
    model = _init_model(args, dataset_factory)
    loss_fn = _init_loss_function(args)
    optimizer = _init_optim(args, model)
    scheduler = _init_scheduler(args, optimizer)
    # ---> train and validate
    results_dict, metrics = _step(args, cur, loss_fn, model, dataset_factory, optimizer, scheduler, train_loader, test_loader, log_file)
    return results_dict, metrics