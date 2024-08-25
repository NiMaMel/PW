import numpy as np
import pandas as df
import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,accuracy_score, f1_score

# 1. Datasplit Methods

def data_split(df, seed):

    np.random.seed(seed)
    
    # Unique tasks
    unique_targets = df['target_id'].unique()
    
    # Shuffle the unique tasks 
    np.random.shuffle(unique_targets)

    # Define the proportions for each set
    train_prop, val_prop, test_prop = 0.6, 0.2, 0.2

    # Calculate the number of tasks for each set
    n_tasks = len(unique_targets)
    n_train = int(train_prop * n_tasks)
    n_val = int(val_prop * n_tasks)

    # Split the tasks into train, validation, and test sets
    train_targets = unique_targets[:n_train]
    val_targets = unique_targets[n_train:n_train+n_val]
    test_targets = unique_targets[n_train+n_val:]


    # Filter the DataFrame based on the selected tasks for each set
    train_triplet = df[(df['target_id'].isin(train_targets))]
    val_triplet = df[(df['target_id'].isin(val_targets))]
    test_triplet = df[(df['target_id'].isin(test_targets))]
    
    # Display the lengths of the sets
    print(f"Train set length: {len(train_triplet)}")
    print(f"Validation set length: {len(val_triplet)}")
    print(f"Test set length: {len(test_triplet)}")

    return train_triplet, val_triplet, test_triplet

# 2. Train Methods

def train_rf(df,data,triplet,seed,n_estimators,shuffle = False):
    
    """
    df     : org dataset (sider)
    data   : descriptors
    triplet: organized df
    """
    
    np.random.seed(seed)
    
    tasks = triplet['target_id'].unique()
    y_hats_proba = np.empty((data.shape[0]-16, len(tasks)))
    y_hats_class = np.empty_like(y_hats_proba)
    true_labels  = np.empty_like(y_hats_proba)
    
    for i,task in enumerate(tasks):
        
        # create mask for supportset (trainset)
        n_idx = df.loc[df[df.columns[task]] == 0].index.to_numpy()
        p_idx = df.loc[df[df.columns[task]] == 1].index.to_numpy()
        n_mask = np.random.choice(n_idx, size= 8, replace=False)
        p_mask = np.random.choice(p_idx, size= 8, replace=False)

        # masks for testset
        task_labels = df.iloc[:,task]
        test_mask = np.full(len(task_labels), True)
        test_mask[n_mask] = False
        test_mask[p_mask] = False
        
        # train test split )
        X_train = np.concatenate((data[n_mask], data[p_mask]), axis=0)
        y_train = np.concatenate((np.zeros(8),np.ones(8)), axis=0)

        X_test = data[test_mask]
        y_test = task_labels[test_mask]
        
        if shuffle:

            indices = np.random.permutation(X_train.shape[0])
            X_train = X_train[indices]
            y_train = y_train[indices]
            
        # train step
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
        rf_model.fit(X_train, y_train)

        # predict      
        y_hats_proba[:, i] = rf_model.predict_proba(X_test)[:, 1]
        y_hats_class[:, i] = rf_model.predict(X_test)
        true_labels[:, i]  = y_test
        
    return y_hats_proba, y_hats_class,true_labels

# 3. Eval Methods

def eval_dauprc(model,dataloader,device):
    
    model.train(False)
    predictions = []
    true_labels = []
    target_ids = []
    threshold = 0.5
    
    for batch, data_ in enumerate(dataloader):
        # compute output
        q, p, n, t = data_["query_mol"].to(device), data_["p_supp"].to(device), data_["n_supp"].to(device),data_["task_id"].to(device)
        preds = model(q, p, n, train=False)
        predictions.append(preds)
        #pred_labels = (torch.sigmoid(preds) >= threshold).float()
        #predictions.append(pred_labels)
        true_labels.append(data_['query_label'].to(device))
        target_ids.append(t)


    predictions, true_labels,target_ids = torch.stack(flatten(predictions)), torch.stack(flatten(true_labels)),torch.stack(flatten(target_ids))

    mean_dauprcs, dauprcs,target_id_list = compute_dauprc_score(predictions, true_labels, target_ids)

    return mean_dauprcs, dauprcs, target_id_list

def compute_dauprc_score(predictions, labels, target_ids):
    dauprcs = list()
    target_id_list = list()

    for target_idx in torch.unique(target_ids):
        
        rows = torch.where(target_ids == target_idx)
        preds = predictions[rows].detach()
        y = labels[rows].int()

        if torch.unique(y).shape[0] == 2:
            number_actives = y[y == 1].shape[0]
            number_inactives = y[y == 0].shape[0]
            number_total = number_actives + number_inactives

            random_clf_auprc = number_actives / number_total
            auprc = average_precision_score(
                y.cpu().numpy().flatten(), preds.cpu().numpy().flatten()
            )

            dauprc = auprc - random_clf_auprc
            dauprcs.append(dauprc)
            target_id_list.append(target_idx.item())
        else:
            dauprcs.append(np.nan)
            target_id_list.append(target_idx.item())

    return np.nanmean(dauprcs), dauprcs, target_id_list


def flatten(li):
    """
    Flattens a given list.
    used in acc_f1 and evaluate model
    """
    return [item for sublist in li for item in sublist]

def auc_pr(model, loader,device):
    """
    Return accuracy and F1 score of a given model and dataloader.
    To Do: Evaluate per Task and return mean
    """
    model.train(False)
    predictions = []
    targets = []
    threshold = 0.5
    for batch, data_ in enumerate(loader):
        # compute output
        q, p, n, t = data_["query_mol"].to(device), data_["p_supp"].to(device), data_["n_supp"].to(device),data_["task_id"].to(device)
        preds = model(q, p, n, train=False)
        pred_labels = (preds >= threshold).float()
        predictions.append(pred_labels.cpu())
        targets.append(data_['query_label'].cpu())

    # Flatten predictions and targets
    predictions, targets = flatten(predictions), flatten(targets)

    # auc pr
    aucPR = average_precision_score(targets, predictions)

    return aucPR

def acc_f1(model, loader,device):
    """
    Return accuracy and F1 score of a given model and dataloader.
    To Do: Evaluate per Task and return mean
    """
    model.train(False)
    predictions = []
    targets = []
    threshold = 0.5
    for batch, data_ in enumerate(loader):
        # compute output
        q, p, n, t = data_["query_mol"].to(device), data_["p_supp"].to(device), data_["n_supp"].to(device),data_["task_id"].to(device)
        preds = model(q, p, n, train=False)
        #pred_labels = (torch.sigmoid(preds) >= threshold).float()
        pred_labels = (preds >= threshold).float()
        predictions.append(pred_labels.cpu())
        targets.append(data_['query_label'].cpu())

    # Flatten predictions and targets
    predictions, targets = flatten(predictions), flatten(targets)

    # Accuracy
    correct = (torch.tensor(predictions) == torch.tensor(targets)).float().sum()
    accuracy = (correct / len(predictions)).item()

    # F1 Score
    f1 = f1_score(targets, predictions)

    del predictions, targets, preds

    return accuracy, f1

def acc(model, loader):
    """
    Return accuracy of a given model and dataloader.
    """
    model.train(False)
    predictions = []
    targets = []
    threshold = 0.5
    for batch, data_ in enumerate(loader):
        # compute output
        q, p, n, t = data_["query_mol"].to(device),data_["p_supp"].to(device),data_["n_supp"].to(device),data_["task_id"].to(device)
        preds = model(q,p,n, train=False)
        pred_labels = (torch.sigmoid(preds) >= threshold).float()
        predictions.append(pred_labels.cpu())
        targets.append(data_['query_label'].cpu())

    # accuracy
    predictions, targets = flatten(predictions), flatten(targets)
    correct = (torch.tensor(predictions) == torch.tensor(targets)).float().sum()
    accuracy = (correct / len(predictions)).item()
    
    del predictions, targets, preds
    
    return accuracy