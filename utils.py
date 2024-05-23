import sys
import numpy as np
import pandas as pd
import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score, f1_score
from collections import defaultdict

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

def train_model(config, writer, train_loader, val_loader, device, BATCH_SIZE):
    """ 
    Training procedure.
    """
    model = config['model']
    optimizer = config['optimizer']
    criterion = config['criterion']
    
    MAX_EPOCHS = config['max_epochs']
    PATIENCE = config['patience']
    log_interval = config['log_interval']
    save_path = config['save_path']
    val_interval = config['val_interval']  
    
    best_val_score = None
    auc= 0.
    daucPR = 0.
    pat_log = 0
    avg_valLoss = 0 
    avg_trainLoss = 0
    
    for epoch in range(MAX_EPOCHS):
        
        losses = []
        val_losses = []
        model.train(True)
        
        for batch, data_ in enumerate(train_loader):
            # reset gradients
            optimizer.zero_grad()

            # compute output
            q, p, n, t = data_["query_mol"].to(device),data_["p_supp"].to(device),data_["n_supp"].to(device),data_["task_id"].to(device)
            #preds = model(q,p,n, train=False)
            preds = model(q,p,n)
            # compute loss
            loss = criterion(preds, data_['query_label'].to(device)) 
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach())
            avg_trainLoss = np.mean(losses)
            
            # plotting
            if batch % log_interval == 0 or batch == BATCH_SIZE - 1:
                out = f'epoch:{epoch + 1}/{MAX_EPOCHS} batches:{batch:>04d}/{len(train_loader) - 1}'
                out += f' avg-train_loss:{avg_trainLoss:.8f}, avg-val_loss:{avg_valLoss:.8f}, val-auc:{auc:.8f}, val-dauc_pr:{daucPR:.8f}'
                
                # overwrite what's already been written
                sys.stdout.write('\r' + ' ' * 400)
                # write 'out' to stdout
                sys.stdout.write(f'\r{out}')
                sys.stdout.flush()
                
        # compute validation loss
        if (epoch + 1) % val_interval == 0:
            model.eval() 
            with torch.no_grad():
                for vdata_ in val_loader:
                    q, p, n, t = vdata_["query_mol"].to(device), vdata_["p_supp"].to(device), vdata_["n_supp"].to(device), vdata_["task_id"].to(device)
                    #preds = model(q, p, n)  
                    preds = model(q, p, n, train=False)  
                    val_loss = criterion(preds, vdata_['query_label'].to(device))
                    val_losses.append(val_loss.cpu().detach())
                    avg_valLoss = np.mean(val_losses)
 
        del preds, loss

        # eval step on val-set
        daucPR = dauc_pr(model, val_loader, device)
        auc = auc_score(model, val_loader, device)
 
        # logging tensorboard
        writer.add_scalar(f"{model.__class__.__name__} training avg-loss", np.mean(losses), epoch)
        if len(val_losses) > 0:
            writer.add_scalar(f"{model.__class__.__name__} validation avg-loss", np.mean(val_losses), epoch)
        writer.add_scalar(f"{model.__class__.__name__} validation auc", auc, epoch)
        writer.add_scalar(f"{model.__class__.__name__} validation dauc_pr", daucPR, epoch)

        # saving best model
        if best_val_score is None or best_val_score < auc: 
            best_val_score = auc
            torch.save(model.state_dict(), save_path)
            pat_log = 0
        else:
            pat_log += 1

        # early stopping
        if pat_log == PATIENCE:
            break

    print("Finished training...")


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

# do not flatten use mean of scores 

def flatten(li):
    """
    Flattens a given list.
    used in acc_f1 and evaluate model
    # not used anymore
    """
    return [item for sublist in li for item in sublist]

def dauc_pr(model, loader, device):
    """
    Return mean dAUC PR of a given model and dataloader, evaluated per task.
    """
    model.train(False)
    task_predictions = defaultdict(list)
    task_targets = defaultdict(list)
    #threshold = 0.5 no thresholding fopr auc and daucPR

    for batch, data_ in enumerate(loader):
        # compute output
        q, p, n, t = data_["query_mol"].to(device), data_["p_supp"].to(device), data_["n_supp"].to(device),data_["task_id"].to(device)
        predictions = model(q, p, n, train=False)

        # Append predictions and targets to the corresponding task lists
        for task_id, pred, target in zip(t.cpu().numpy(), predictions.detach().cpu().numpy(), data_['query_label'].cpu().numpy()):
            task_predictions[task_id].append(pred)
            task_targets[task_id].append(target)

    # Calculate dAUC PR for each task
    task_dauc_prs = []
    for task_id in task_predictions:
        preds = task_predictions[task_id] 
        targets = task_targets[task_id]

        # Get the number of actives and inactives
        _, counts = torch.unique(torch.tensor(targets), return_counts=True)
        n_actives = counts[1].numpy().astype(np.float64)
        n_inactives = counts[0].numpy().astype(np.float64)
        n_total = n_actives + n_inactives
        random_clf_auprc = n_actives / n_total
        
        # dAUC PR
        aucPR = average_precision_score(targets, preds)
        daucPR = aucPR - random_clf_auprc
        task_dauc_prs.append(daucPR)

    # Return the mean dAUC PR across all tasks
    mean_dauc_pr = np.mean(task_dauc_prs)
    return mean_dauc_pr

def auc_score(model, loader, device):
    """
    Return mean auc of a given model and dataloader, evaluated per task.
    """
    model.train(False)
    task_predictions = defaultdict(list)
    task_targets = defaultdict(list)
    #threshold = 0.5 no thresholding fopr auc and daucPR

    for batch, data_ in enumerate(loader):
        # compute output
        q, p, n, t = data_["query_mol"].to(device), data_["p_supp"].to(device), data_["n_supp"].to(device),data_["task_id"].to(device)
        predictions = model(q, p, n, train=False)

        # Append predictions and targets to the corresponding task lists
        for task_id, pred, target in zip(t.cpu().numpy(), predictions.detach().cpu().numpy(), data_['query_label'].cpu().numpy()):
            task_predictions[task_id].append(pred)
            task_targets[task_id].append(target)

    # Calculate auc for each task
    task_aucs = []
    for task_id in task_predictions:
        preds = task_predictions[task_id] 
        targets = task_targets[task_id]

        # auc score
        auc = roc_auc_score(targets, preds)
        task_aucs.append(auc)

    # Return the mean dAUC PR across all tasks
    mean_auc_pr = np.mean(task_aucs)
    return mean_auc_pr

def acc_f1(model, loader,device):
    """
    Return accuracy and F1 score of a given model and dataloader
    Evaluates per Task and return mean
    """
    model.train(False)
    task_predictions = defaultdict(list)
    task_targets = defaultdict(list)
    threshold = 0.5
    
    for batch, data_ in enumerate(loader):
        # compute output
        q, p, n, t = data_["query_mol"].to(device), data_["p_supp"].to(device), data_["n_supp"].to(device),data_["task_id"].to(device)
        preds = model(q, p, n, train=False)
        pred_labels = (preds >= threshold).float()
        
        # Append predictions and targets to the corresponding task lists
        for task_id, pred, target in zip(t.cpu().numpy(), pred_labels.cpu().numpy(), data_['query_label'].cpu().numpy()):
            task_predictions[task_id].append(pred)
            task_targets[task_id].append(target)

    # Calculate acc and f1 for each task
    task_accs = []
    task_f1s = []
    for task_id in task_predictions:
        preds = task_predictions[task_id]
        targets = task_targets[task_id]

        # acc  & f1 score
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds)
        task_accs.append(acc)
        task_f1s.append(f1)

    # Return the mean acc and f1 across all tasks
    mean_acc = np.mean(task_accs)
    mean_f1 = np.mean(task_f1s)
    
    return mean_acc, mean_f1     

def combined_metrics(model, loader, device):

    # call prev functions
    acc, f1 = acc_f1(model, loader,device)
    auc = auc_score(model, loader, device)
    daucPr = dauc_pr(model, loader, device)

    # store them in a pd df
    metrics_dict = {
    'Accuracy': acc,
    'F1 Score': f1,
    'AUC': auc,
    'D-AUC PR': daucPr
    }

    # Convert the dictionary to a pandas DataFrame
    return pd.DataFrame([metrics_dict])
    
def eval_rf(y_hat,y_true):

    # Initialize lists to store metrics for each task
    accuracies = []
    f1_scores = []
    auc_scores = []
    dauc_pr_scores = []
    
    # Iterate over each task (column)
    for task_id in range(y_hat.shape[1]):
        preds = y_hat[:, task_id]
        trues = y_true[:, task_id]
        
        # Compute metrics for the current task
        acc = accuracy_score(trues, preds) 
        f1 = f1_score(trues, preds)
        auc = roc_auc_score(trues, preds)
        aucPR = average_precision_score(trues, preds)

        # for daucPr compute random_clf_auprc
        _, counts = np.unique(trues, return_counts=True)
        n_inactives = counts[0]
        n_actives = counts[1]
        n_total = n_actives + n_inactives
        random_clf_auprc = n_actives / n_total

        # dAUC PR
        daucPR = aucPR - random_clf_auprc

        # Append metrics to the lists
        accuracies.append(acc)
        f1_scores.append(f1)
        auc_scores.append(auc)
        dauc_pr_scores.append(daucPR)
    
    # Create a DataFrame to store metrics for each task
    metrics_dict = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'AUC': np.mean(auc_scores),
        'D-AUC PR': np.mean(dauc_pr_scores)
        }
    
    # Convert the dictionary to a pandas DataFrame
    return pd.DataFrame([metrics_dict]) 
