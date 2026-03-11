#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: dataset_survival.py
@author:zyl
@contact:yilan.zhang@kaust.edu.sa
@time:12/20/24 4:17 PM
'''
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _unpack_data(data, device, omics_format):
    # [img, omic_data_list, label, event_time, c]
    data_wsi = data[0].to(device)

    if omics_format == "Pathways":
        data_omics = []  # TODO: check
        for idx, item in enumerate(data[1]):
            for idy, omic in enumerate(item):
                omic = omic.to(device)
                omic = omic.unsqueeze(0)
                if idx == 0:
                    data_omics.append(omic)
                else:
                    data_omics[idy] = torch.cat((data_omics[idy], omic), dim=0)
    else:
        data_omics = data[1].to(device)

    y_disc = data[2].to(device)
    event_time = data[3].to(device)
    c = data[4].to(device)

    return data_wsi, data_omics, y_disc, event_time, c

SIGNATURES = ["all", "six", "hallmarks", "combine", "xena"]
RNA_FORMATS = ["RNASeq", "Pathways", "GeneEmbedding"]

# six: 2979 genes,
# combine: 9076 genes,
# hallmarks: 6658 genes,
# xena: 2418
# all: 17518 genes

class SurvivalDatasetFactory:
    def __init__(self,
                 study,
                 data_path,
                 rna_format,
                 label_col,
                 signature="all",
                 n_bins=4,
                 eps=1e-6,
                 num_patches=4096,
                 num_genes=None):
        self.study = study
        self.data_path = data_path
        self.signature = signature
        if self.signature not in SIGNATURES:
            raise ValueError(f"Invalid signature: {self.signature}")
        self.rna_format = rna_format
        if self.rna_format not in RNA_FORMATS:
            raise ValueError(f"Invalid RNA format: {self.rna_format}")
        self.n_bins = n_bins
        self.num_patches = num_patches
        self.num_genes = num_genes
        self.eps = eps
        self.label_col = label_col

        if self.label_col == "survival_months_os":
            self.survival_endpoint = "OS"
            self.censorship_var = "censorship_os"
        elif self.label_col == "survival_months_pfi":
            self.survival_endpoint = "PFI"
            self.censorship_var = "censorship_pfi"
        elif self.label_col == "survival_months_dss":
            self.survival_endpoint = "DSS"
            self.censorship_var = "censorship_dss"
        elif self.label_col == "survival_months_dfs": #TODO: donot support this
            self.survival_endpoint = "DFS"
            self.censorship_var = "censorship_dfs"

        # ---> process gene expression data
        self._setup_gene_data()  # self.omics_names, self.omic_sizes,self.gene_embedding_df,self.gene_data_df

        # ---> process clinical data
        self._setup_clinical_data()  # self.clinical_df, self.bins

    def _setup_clinical_data(self):
        clinical_path = os.path.join(self.data_path, "clinical", "all", f"{self.study}.csv")
        clinical_df = pd.read_csv(clinical_path)
        clinical_df = clinical_df[["case id", self.label_col, self.censorship_var, "wsi"]]
        self.clinical_df = clinical_df.dropna()
        # reindex the clinical data
        self.clinical_df = self.clinical_df.reset_index(drop=True)

        # discretize the label
        uncensored_df = self._get_uncensored_data()
        self._disc_label(uncensored_df)

    def _get_uncensored_data(self):
        uncensored_df = self.clinical_df[self.clinical_df[self.censorship_var] < 1]
        return uncensored_df

    def _disc_label(self, uncensored_df):
        disc_labels, q_bins = pd.qcut(uncensored_df[self.label_col], q=self.n_bins, retbins=True, labels=False)
        q_bins[-1] = uncensored_df[self.label_col].max() + self.eps
        q_bins[0] = uncensored_df[self.label_col].min() - self.eps
        disc_labels, q_bins = pd.cut(self.clinical_df[self.label_col], bins=self.n_bins, retbins=True, labels=False,
                                     right=False, include_lowest=True)
        self.clinical_df.insert(2, 'label', disc_labels.values.astype(int))
        self.bins = q_bins

    def _setup_signatures(self, rna_data_df):
        if self.signature == "six":
            signature_path = os.path.join(self.data_path, "signatures", f"signatures.csv")
        elif self.signature == "all": #TODO: check this
            signature_path = os.path.join(self.data_path, "signatures", f"combine_signatures.csv")
        else:
            signature_path = os.path.join(self.data_path, "signatures", f"{self.signature}_signatures.csv")

        signature_df = pd.read_csv(signature_path)

        self.omic_names = []
        self.pathway_names = []  # only keep pathways with non-empty omic sets

        for col in signature_df.columns:
            omic = signature_df[col].dropna().unique()
            omic = sorted(set(omic).intersection(set(rna_data_df.index)))

            if len(omic) == 0:
                continue  # skip empty omics

            self.omic_names.append(omic)
            self.pathway_names.append(col)  # keep corresponding pathway name only

        self.omic_sizes = [len(omic) for omic in self.omic_names]
        print("pathway size: ", len(self.omic_sizes))


    def _setup_gene_embeddings(self):
        gane_embedding_path = os.path.join(self.data_path, "gene_embedding_inter", f"genes_embedding_768.csv")
        self.gene_embedding_df = pd.read_csv(gane_embedding_path, index_col=0)


    def _setup_gene_data(self):
        rna_file = os.path.join(self.data_path, "raw_rna_data_inter", f"{self.study}_rna_inter.csv")
        rna_data_df = pd.read_csv(rna_file, index_col=0)
        self.gene_data_df = rna_data_df
        self._setup_signatures(rna_data_df)
        if self.rna_format == "RNASeq":
            if self.signature != "all": # flatten the self.omic_names
                self.omic_names = [item for sublist in self.omic_names for item in sublist]
                self.gene_data_df = self.gene_data_df.loc[self.omic_names]
            self.omic_sizes = self.gene_data_df.shape[0]
            self.gene_embedding_df = None
        elif self.rna_format == "Pathways":
            self.gene_embedding_df = None
        elif self.rna_format == "GeneEmbedding":
            self._setup_gene_embeddings()
            if self.signature != "all": # flatten the self.omic_names
                self.omic_names = [item for sublist in self.omic_names for item in sublist]
                self.gene_data_df = self.gene_data_df.loc[self.omic_names]
                self.gene_embedding_df = self.gene_embedding_df.loc[self.omic_names]
            self.omic_sizes = self.gene_data_df.shape[0]
            print("gene embedding shape: ", self.gene_embedding_df.shape)
        else:
            raise ValueError(f"Invalid RNA format: {self.rna_format}")


    def _print_info(self):
        print("Study: ", self.study)
        print("Signature: ", self.signature)
        print("RNA format: ", self.rna_format)
        print("Label column: ", self.label_col)
        print("Number of bins: ", self.n_bins)
        print("Number of patches: ", self.num_patches)
        print("Number of genes: ", self.num_genes)
        print("Censorship variable: ", self.censorship_var)
        print("omic sizes: ", self.omic_sizes) # length of the genes
        # print("omic names: ", self.omic_names)
        if self.rna_format == "Pathways":
            print("pathway names: ", self.pathway_names)


class SurvivalDataset(Dataset):
    def __init__(self, dataset_factory, wsi_path, split_key: str = 'train', fold=None, encoding_dim=768):
        self.dataset_factory = dataset_factory
        self.wsi_path = wsi_path
        self.split_key = split_key
        self.fold = fold  # which fold to use
        self.encoding_dim = encoding_dim

        if split_key in ['train', 'val']:
            self.label_df = self._load_split()
        else:
            raise ValueError(f"Invalid split key: {split_key}")

    def _load_split(self):
        split_path = os.path.join(self.dataset_factory.data_path, "splits", "5fold", f"{self.dataset_factory.study}",
                                  f"fold_{self.fold}.csv")
        all_splits = pd.read_csv(split_path)
        split = self._get_split_from_df(all_splits, self.split_key)
        return split

    def _get_split_from_df(self, all_splits, split_key: str = 'train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        # change splits to list
        split = split.tolist()

        clinical_df_splits = self.dataset_factory.clinical_df[self.dataset_factory.clinical_df['case id'].isin(split)]

        # reset the index
        clinical_df_splits = clinical_df_splits.reset_index(drop=True)

        return clinical_df_splits

    def load_wsi(self, slides):
        if str(slides) == "nan":
            return torch.zeros((1))
        else:
            slide_ids = slides.split(", ")
            wsi = []
            for slide_id in slide_ids:
                wsi_path = os.path.join(self.wsi_path, '{}.pt'.format(slide_id.rstrip('.svs')))
                if os.path.exists(wsi_path):
                    wsi.append(torch.load(wsi_path))
                else:
                    wsi.append(torch.zeros((self.dataset_factory.num_patches, self.encoding_dim)))
                    print("missing file: ", slide_id)
            wsi = torch.cat(wsi, dim=0).type(torch.float32)  # TODO: check the torch.float32
            return wsi

    def load_genes(self, case_id):
        patient_genes = self.dataset_factory.gene_data_df[case_id]

        if self.dataset_factory.rna_format == "RNASeq":
            patient_genes = torch.from_numpy(patient_genes.values.astype(np.float32))
            return patient_genes
        elif self.dataset_factory.rna_format == "Pathways":
            omic_list = []
            for omic in self.dataset_factory.omic_names:
                omic_data = patient_genes[omic].values
                omic_data = torch.from_numpy(omic_data.astype(np.float32))
                # print(omic_data.size(0))
                # # pad the omic data to the shared size
                # if omic_data.size(0) < 195:
                #     omic_data = torch.cat([omic_data, torch.zeros(195 - omic_data.size(0))], dim=0)
                omic_list.append(omic_data)
            return omic_list
        elif self.dataset_factory.rna_format == "GeneEmbedding":
            patient_genes = torch.from_numpy(patient_genes.values.astype(np.float32))
            rna = patient_genes.unsqueeze(1)
            gene_data = self.dataset_factory.gene_embedding_df.values
            gene_data = torch.from_numpy(gene_data.astype(np.float32))
            gene_embedding = rna * gene_data
            return gene_embedding
        else:
            raise ValueError(f"Invalid RNA format: {self.dataset_factory.rna_format}")

    def get_label(self, case_id):
        label = self.label_df[self.label_df['case id'] == case_id]['label']
        event_time = self.label_df[self.label_df['case id'] == case_id][self.dataset_factory.label_col]
        censorship = self.label_df[self.label_df['case id'] == case_id][self.dataset_factory.censorship_var]
        # convert to tensor
        label = torch.tensor(label.values[0], dtype=torch.long)
        event_time = torch.tensor(event_time.values[0], dtype=torch.float32)
        censorship = torch.tensor(censorship.values[0], dtype=torch.float32)
        return label, event_time, censorship

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        case_id = self.label_df.loc[idx, 'case id']
        slides = self.label_df.loc[idx, 'wsi']
        label, event_time, censorship = self.get_label(case_id)
        wsi = self.load_wsi(slides)
        genes = self.load_genes(case_id)

        # sample from the patches
        if self.dataset_factory.num_patches is not None and self.split_key == 'train':
            n_samples = min(self.dataset_factory.num_patches, wsi.size(0))
            idx = np.sort(np.random.choice(wsi.size(0), n_samples, replace=False))
            wsi = wsi[idx, :]

            if n_samples < self.dataset_factory.num_patches:
                wsi = torch.cat([wsi, torch.zeros(self.dataset_factory.num_patches - n_samples, wsi.size(1))], dim=0)
        if self.dataset_factory.num_genes is not None and self.split_key == 'train':
            if self.dataset_factory.rna_format != "Pathways":
                n_genes = min(self.dataset_factory.num_genes, genes.size(0))
                idx = np.sort(np.random.choice(genes.size(0), n_genes, replace=False))
                genes = genes[idx]
                if n_genes < self.dataset_factory.num_genes:
                    genes = torch.cat([genes, torch.zeros(self.dataset_factory.num_genes - n_genes)], dim=0)

        return wsi, genes, label, event_time, censorship


def _collate_pathways(batch):

    img = torch.stack([item[0] for item in batch])

    omic_data_list = []
    for item in batch:
        omic_data_list.append(item[1])

    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])

    return [img, omic_data_list, label, event_time, c]

if __name__ == '__main__':
    from torch.utils.data import DataLoader, SubsetRandomSampler

    study = "blca"
    data_path = "./dataset_csv"
    rna_format = "Pathways"  # "RNASeq", "Pathways", "GeneEmbedding"
    label_col = "survival_months_dss"
    signature = "combine"

    dataset_factory = SurvivalDatasetFactory(study, data_path, rna_format, label_col, signature, num_genes=None)
    dataset_factory._print_info()

    wsi_path = f"/Data/Pathology/UNI/{study}/pt_files/"
    split_key = 'train'
    fold = 0
    dataset = SurvivalDataset(dataset_factory, wsi_path, split_key, fold)

    if rna_format == "Pathways":
        collate_fn = _collate_pathways
    else:
        collate_fn = None

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)
    # for i, (wsi, genes, label, event_time, censorship) in enumerate(train_loader):
    #     print(wsi.shape, label, event_time, censorship)
        # for gene in genes:
        #     print(gene.shape)

    for i, data in enumerate(train_loader):
        wsi, genes, label, event_time, censorship = _unpack_data(data, device="cpu", omics_format=rna_format)
        # print(wsi.shape, label, event_time, censorship)
        for gene in genes:
            print(gene.shape)

