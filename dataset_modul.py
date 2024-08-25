import numpy as np
import pandas as pd
import torch as torch
import torch.nn as nn
from torch.utils.data import Dataset

class Dataset(Dataset):

    def __init__(self, data, df, triplet_df, supp=8, train = True,seed=None):
        
        self.data = data
        self.df = df #sider
        self.triplet_df = triplet_df # cols: mol_id,target_id,label 
        self.supp = supp
        self.train = train
        self.seed = seed

        if seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        
        if not self.train:
            
            self.tasks = self.triplet_df['target_id'].unique()
            self.evalsets = { task : {'n_supp': None, 'p_supp': None} for task in self.tasks}
            
            for task in self.tasks:

                n_mask, p_mask = self.support_mask(task)
                
                # update
                self.evalsets[task]["n_supp"] = torch.from_numpy(data[n_mask]).float()
                self.evalsets[task]["p_supp"] = torch.from_numpy(data[p_mask]).float()
            
                masks = np.concatenate([n_mask, p_mask]).tolist()
                for idx in masks:
            
                    # Delete rows where mol_id == index in indexes_to_delete and task_id == current task_id
                    self.triplet_df = self.triplet_df[~((self.triplet_df['mol_id'] == idx) & (self.triplet_df['target_id'] == task))]

    def support_mask(self,task, mol_id = None, ex_mol = False):
    
            if ex_mol:
                # p == label with 1 / n == labels with 0 
                n_idx = self.df.loc[(self.df[self.df.columns[task]] == 0) & (self.df.index != mol_id)].index.to_numpy()
                p_idx = self.df.loc[(self.df[self.df.columns[task]] == 1) & (self.df.index != mol_id)].index.to_numpy()
                
                n_mask = np.random.choice(n_idx, size= self.supp, replace=False)
                p_mask = np.random.choice(p_idx, size= self.supp, replace=False)
                    
            else:

                # p == label with 1 / n == labels with 0 
                n_idx = self.df.loc[self.df[self.df.columns[task]] == 0].index.to_numpy()
                p_idx = self.df.loc[self.df[self.df.columns[task]] == 1].index.to_numpy()
                
                n_mask = np.random.choice(n_idx, size= self.supp, replace=False)
                p_mask = np.random.choice(p_idx, size= self.supp, replace=False)
                
            return n_mask, p_mask
        
    def __len__(self):
        return len(self.triplet_df) 

    def __getitem__(self, index):

        if self.train:
            mol_id = self.triplet_df.iloc[index]["mol_id"]
            label  = self.triplet_df.iloc[index]["label"]
            task   = self.triplet_df.iloc[index]["target_id"] # think of also returning the target id for evaluation
            query  = self.data[mol_id] # get descriptor
            
            n_mask, p_mask = self.support_mask(task, mol_id, ex_mol = True)
            
            return {
                'query_mol': torch.from_numpy(query).float(),
                'query_label': torch.tensor(label).float(),
                'n_supp': torch.from_numpy(self.data[n_mask]).float(),
                'p_supp': torch.from_numpy(self.data[p_mask]).float(),
                'task_id': torch.tensor(task).float()}

        else:

            mol_id = self.triplet_df.iloc[index]["mol_id"]
            label  = self.triplet_df.iloc[index]["label"]
            task   = self.triplet_df.iloc[index]["target_id"]
            query  = self.data[mol_id] # get descriptor

            return {
                'query_mol': torch.from_numpy(query).float(),
                'query_label': torch.tensor(label).float(),
                'n_supp': self.evalsets[task]["n_supp"],
                'p_supp': self.evalsets[task]["p_supp"],
                'task_id': torch.tensor(task).float()}