import sys
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
    
    best_val_score = None
    accuracy = 0.
    f1 = 0.
    aucPR = 0.
    pat_log = 0
    
    for epoch in range(MAX_EPOCHS):
        losses = []
        model.train(True)
        for batch, data_ in enumerate(train_loader):
            # reset gradients
            optimizer.zero_grad()

            # compute output
            q, p, n, t = data_["query_mol"].to(device),data_["p_supp"].to(device),data_["n_supp"].to(device),data_["task_id"].to(device)
            preds = model(q,p,n, train=False)
            # compute loss
            loss = criterion(preds, data_['query_label'].to(device)) 
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach())

            # plotting
            if batch % log_interval == 0 or batch == BATCH_SIZE - 1:
                out = f'epoch:{epoch + 1}/{MAX_EPOCHS} batches:{batch:>04d}/{len(train_loader) - 1}'
                #out += f' avg-loss:{np.mean(losses)} acc-val:{accuracy}'
                out += f' avg-loss:{np.mean(losses)} val-auc_pr:{aucPR}'
                #out += f' avg-loss:{np.mean(losses)} acc-val:{f1}'

                # overwrite what's already been written
                sys.stdout.write('\r' + ' ' * 400)
                # write 'out' to stdout
                sys.stdout.write(f'\r{out}')
                sys.stdout.flush()

        del preds, loss

        # validation
        #accuracy , f1 = acc_f1(model, val_loader, device)
        aucPR = auc_pr(model, val_loader, device)

        # logging tensorboard
        writer.add_scalar(f"{model.__class__.__name__} training avg-loss", np.mean(losses), epoch)
        #writer.add_scalar(f"{model.__class__.__name__} validation acc", accuracy, epoch)
        writer.add_scalar(f"{model.__class__.__name__} validation auc_pr", aucPR, epoch)

        # saving best model
        #if best_val_score is None or best_val_score < accuracy:
        if best_val_score is None or best_val_score < aucPR: 
            #best_val_score = accuracy
            best_val_score = aucPR
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
