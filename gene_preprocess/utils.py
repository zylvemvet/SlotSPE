#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: preprocess_utils.py
@author:zyl
@contact:yilan.zhang@kaust.edu.sa
@time:12/18/24 4:34 PM
'''

import pandas as pd
import numpy as np
import os
import shutil
import tqdm
from sklearn.model_selection import KFold

def reorganize_rna_seq_data(data):
    data = data.drop(columns=["Entrez_Gene_Id"])
    # remove the NaN values
    data = data.dropna()
    data = data.reset_index(drop=True)
    data = data.set_index("Hugo_Symbol")
    data = data.rename_axis(None, axis=0)
    # remove th "01" in the end of the slide names
    data.columns = [name[:-3] for name in data.columns]
    # remove the duplicated columns
    data = data[~data.index.duplicated(keep='first')]

    return data

def load_csv_data(path):
    data = pd.read_csv(path, index_col=0)
    return data

def load_tsv_txt_data(path):
    data = pd.read_csv(path,sep='\t')
    return data

def series_intersection(s1, s2):
    r"""
    Return insersection of two sets

    Args:
        - s1 : set
        - s2 : set

    Returns:
        - pd.Series

    """
    return list(set(s1) & set(s2))


def reorganize_wsi_data(data_dir, target_dir):
    r"""
    Reorganize the wsi data into the target directory

    Args:
        - data_dir : str
            The directory of the data
        - target_dir : str
            The target directory to save the data

    Returns:
        - None

    """
    os.makedirs(target_dir, exist_ok=True)
    dirs = os.listdir(data_dir)
    for dir in tqdm.tqdm(dirs):
        # travel all the files in the sub_dir
        sub_dir = os.path.join(data_dir, dir)
        if not os.path.isdir(sub_dir):
            continue
        files = os.listdir(sub_dir)
        for file in files:
            if file.endswith(".svs"):
                # move the file to the data_dir
                shutil.move(os.path.join(sub_dir, file), os.path.join(target_dir, file))
                # remove the sub_dir
                # shutil.rmtree(sub_dir)
    # shutil.rmtree(data_dir)

def get_clinical_label(clinical, slides_path, status=None):
    col_names = ["Patient ID", "Disease Free (Months)", "Disease Free Status", "Months of disease-specific survival", "Disease-specific Survival status", "Overall Survival (Months)", "Overall Survival Status", "Progress Free Survival (Months)", "Progression Free Status"]
    clinical = clinical[col_names]
    if status:
        for col in status:
            print(f"the numbers of {col}: \n", clinical[col].value_counts())

    # change name: Disease Free (Months) -> survival_months_dfs
    # change name:Disease Free Status -> censorship_dfs, 1:Recurred/Progressed, 0: Disease Free -> 0, 1, NaN -> NaN
    clinical = clinical.rename(columns={"Disease Free (Months)": "survival_months_dfs", "Disease Free Status": "censorship_dfs"})
    # clinical["censorship_dfs"] = clinical["censorship_dfs"].apply(lambda x: 1 if x == "0:DiseaseFree" else 0) # 1 means censored
    for row in clinical.iterrows():
        if row[1]["censorship_dfs"] == "1:Recurred/Progressed":
            clinical.at[row[0], "censorship_dfs"] = 0
        elif row[1]["censorship_dfs"] == "0:DiseaseFree":
            clinical.at[row[0], "censorship_dfs"] = 1
        else:
            pass
    # change name: Months of disease-specific survival -> survival_months_dss
    # change name: Disease-specific Survival status -> censorship_dss, 1:DEAD WITH TUMOR, 0: LIVE OR DEAD TUMOR FREE-> 0, 1
    clinical = clinical.rename(columns={"Months of disease-specific survival": "survival_months_dss", "Disease-specific Survival status": "censorship_dss"})
    clinical["censorship_dss"] = clinical["censorship_dss"].apply(lambda x: 1 if x == "0:ALIVE OR DEAD TUMOR FREE" else 0) # 1 means censored
    # change name: Overall Survival (Months) -> survival_months_os
    # change name: Overall Survival Status -> censorship_os, 1:DECEASED , 0:LIVING -> 0, 1
    clinical = clinical.rename(columns={"Overall Survival (Months)": "survival_months_os", "Overall Survival Status": "censorship_os"})
    clinical["censorship_os"] = clinical["censorship_os"].apply(lambda x: 1 if x == "0:LIVING" else 0) # 1 means censored
    # change name: Progress Free Survival (Months) -> survival_months_pfs
    # change name: Progression Free Status -> censorship_pfs, 1:PROGRESSION, 0:CENSORED -> 0, 1
    clinical = clinical.rename(columns={"Progress Free Survival (Months)": "survival_months_pfs", "Progression Free Status": "censorship_pfs"})
    clinical["censorship_pfs"] = clinical["censorship_pfs"].apply(lambda x: 1 if x == "0:CENSORED" else 0) # 1 means censored

    clinical = clinical.rename(columns={"Patient ID": "case id"})

    #the case id in the clinical data is the same as the Case ID in the sheet
    case_dict = {}# case id: File Name
    slides = os.listdir(slides_path)
    for slide in slides:
        if slide.endswith(".pt"):
            # change it to .svs
            slide = slide[:-2] + "svs"
        case_id = slide.strip()[:12]
        # # update the case_dict
        # case_dict.update({case_id: slide})
        if case_id in case_dict:
            case_dict[case_id].append(slide)
        else:
            case_dict[case_id] = [slide]

    # change the list to string by using ', '.join(slide_list)
    for key, value in case_dict.items():
        case_dict[key] = ', '.join(value)

    # add a column to the clinical to save the file name, save all the slides of the case id
    clinical["wsi"] = clinical["case id"].map(case_dict)
    # delete the rows with nan values in the wsi column
    clinical = clinical.dropna(subset=["wsi"])

    return clinical


def get_intersection_between_rna_and_clinical(rna_data, clinical):
    case_ids = clinical["case id"].values
    intersection = set(case_ids).intersection(set(rna_data.columns))
    clinical_new = clinical[clinical["case id"].isin(intersection)]
    return clinical_new



def split_dataset(clinical, save_path, fold=5):
    kf = KFold(n_splits=fold, shuffle=True, random_state=42)
    for i, (train_index, val_index) in enumerate(kf.split(clinical)):
        # new dataframe to save the train and val case id for each fold
        train_val = pd.DataFrame(columns=["train", "val"])
        # get the train and val case id
        train_case_id = clinical.iloc[train_index]["case id"]
        val_case_id = clinical.iloc[val_index]["case id"]
        # reindex the train and val case id
        train_case_id = train_case_id.reset_index(drop=True)
        val_case_id = val_case_id.reset_index(drop=True)
        print(f"the numbers of train case id in fold {i}: ", train_case_id.shape[0])
        print(f"the numbers of val case id in fold {i}: ", val_case_id.shape[0])
        # save the train and val case id
        train_val["train"] = train_case_id
        train_val["val"] = val_case_id
        # save the train and val case id to the csv file
        train_val.to_csv(f"{save_path}/fold_{i}.csv")


def find_failed_pts(data_dir):
    import torch
    files = os.listdir(data_dir)
    for file in files:
        if file.endswith(".pt"):
            try:
                data = torch.load(os.path.join(data_dir, file))
            except:
                print(file)
                os.remove(os.path.join(data_dir, file))


if __name__ == "__main__":
    root_dir = "/Data/Pathology"
    study = "gbm"
    '''step 1: reorganize the rna data'''
    print("\n step 1: reorganize the rna data\n")
    rna_seq_path = f"{root_dir}/RNA/{study}_tcga/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt"
    rna_data = load_tsv_txt_data(rna_seq_path)
    rna_data = reorganize_rna_seq_data(rna_data)
    # save the reorganized data
    rna_data.to_csv(f"/home/zhany0x/Documents/projects/GeneralPIBD/survival/dataset_csv/raw_rna_data/{study}.csv")
    
    ''' step 2: get the gene embedding data and the intersection 
    
    (Note: This part of preprocessing is not included in the paper, but we want to share the preprcessing process for gene embedding.
    We will save the intersection data for future research. Only refer to the precessing precess in Notebook Genes-Preprocessing.ipynb)
    
    '''

    print("\n step 2: get the gene embedding data and the intersection \n")
    rna_seq_path = f"/home/zhany0x/Documents/projects/GeneralPIBD/survival/dataset_csv/raw_rna_data/{study}.csv"
    rna_data = load_csv_data(rna_seq_path)
    
    gene_embeding_path = f"{root_dir}/CTransPath-old/gene/genes_embedding_768.csv"
    gene_embedding = load_csv_data(gene_embeding_path)
    
    intersection = series_intersection(rna_data.index, gene_embedding.index)
    gene_embedding_intersection = gene_embedding.loc[intersection]
    rna_data_intersection = rna_data.loc[intersection]
    # save the intersection data
    
    print("rna_data shape: ", rna_data.shape)
    print("rna_data_intersection shape: ", rna_data_intersection.shape)
    print("gene_embedding_intersection shape: ", gene_embedding_intersection.shape)
    
    rna_data_intersection.to_csv(f"/home/zhany0x/Documents/projects/GeneralPIBD/survival/dataset_csv/raw_rna_data_inter/{study}_rna_inter.csv")
    gene_embedding_intersection.to_csv(f"/home/zhany0x/Documents/projects/GeneralPIBD/survival/dataset_csv/gene_embedding_inter/{study}_768_inter.csv")
    
    ''' step 3: process the clinical data '''
    # # 3.1 reorganize the wsi data
    # data_dir = f"/Data/Pathology/slides-new/{study}/"
    # target_dir = f"/Data/Pathology/slides-new/tcga_{study}/"
    # reorganize_wsi_data(data_dir, target_dir)
    
    # 3.2 get the clinical data
    print("\n step 3.2: get the clinical data \n")
    # slides_path = f"/home/zhany0x/Documents/data/Pathology/Slides/tcga_hnsc/"
    slides_path = f"{root_dir}/UNI/{study}/pt_files/"
    status = ["Disease Free Status", "Disease-specific Survival status", "Overall Survival Status", "Progression Free Status"]
    clinical_path = f"{root_dir}/labels/survival/{study}/{study}_tcga_pan_can_atlas_2018_clinical_data.tsv"
    clinical = load_tsv_txt_data(clinical_path)
    clinical = get_clinical_label(clinical, slides_path, status)
    clinical.to_csv(f"/home/zhany0x/Documents/projects/GeneralPIBD/survival/dataset_csv/clinical/all/{study}.csv")
    
    # 3.3 intersection between rna and clinical
    print("\n step 3.3: intersection between rna and clinical \n")
    clinical_path = f"/home/zhany0x/Documents/projects/GeneralPIBD/survival/dataset_csv/clinical/all/{study}.csv"
    clinical = load_csv_data(clinical_path)
    print("the numbers of clinical: ", clinical.shape[0])
    rna_path = f"/home/zhany0x/Documents/projects/GeneralPIBD/survival/dataset_csv/raw_rna_data_inter/{study}_rna_inter.csv"
    rna_data = load_csv_data(rna_path)
    print("the numbers of rna_data: ", rna_data.shape[1])
    
    clinical_new = get_intersection_between_rna_and_clinical(rna_data, clinical)
    print("the numbers of clinical_new: ", clinical_new.shape[0])
    # reset the index
    clinical_new = clinical_new.reset_index(drop=True)
    clinical_new.to_csv(f"/home/zhany0x/Documents/projects/GeneralPIBD/survival/dataset_csv/clinical/all/{study}.csv")
    
    
    # 3.4 split the dataset
    print("\n step 3.4: split the dataset \n")
    clinical_path = f"/home/zhany0x/Documents/projects/GeneralPIBD/survival/dataset_csv/clinical/all/{study}.csv"
    clinical = load_csv_data(clinical_path)
    save_path = f"/home/zhany0x/Documents/projects/GeneralPIBD/survival/dataset_csv/splits/5fold/{study}/"
    os.makedirs(save_path, exist_ok=True)
    split_dataset(clinical, save_path, fold=5)

    # 4.1 check the failed pts
    # data_dir = "/home/zhany0x/Documents/data/Pathology/UNI/brca/pt_files/"
    # find_failed_pts(data_dir)