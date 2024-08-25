import os
import time
import numpy as np
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import optuna
from optuna.trial import TrialState

import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fs_model import Model
from dataset_modul import Dataset
from preprocessing import preprocessing
from utils import data_split, train_model, auc_score, dauc_pr

seed = 42

np.random.seed(seed)
torch.manual_seed(seed)

# cuda setting
if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    device = torch.device('cuda:%d' % device_id)
else:
    device = torch.device('cpu')
print(f"device set:{torch.cuda.get_device_name(device_id)}\n")

# 1. Load Dataset and create Tiplet-Df
sider = pd.read_csv("datasets/sider.csv")
triplets = [(i,j,row.iloc[j]) for i,row in sider.iterrows() for j in range(1,len(row))]
triplet_df = pd.DataFrame(triplets, columns=['mol_id', 'target_id', 'label'])

# 2. Preprocessing
data = preprocessing(sider)

def objective(trial):

    params = {
        "opt_lr": trial.suggest_float("opt_lr", 1e-7, 0.1,log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3,log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256,512]),
        "input_dim": 2248,
        "hidden_dim": trial.suggest_int("hidden_dim", 1124, 4096),
        "output_dim": trial.suggest_int("output_dim", 12, 1124),
        "num_layers": trial.suggest_int("num_layers", 1, 5),
        "p": trial.suggest_float("p", 0.0, 0.5)
        }

    # 3. Initialize Model

    model = Model(params["input_dim"], params["hidden_dim"],params["output_dim"], params["num_layers"], params["p"])
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=params["opt_lr"], weight_decay=params["weight_decay"])

    # 4. Train-split and Dataloader

    train_triplet, val_triplet, test_triplet = data_split(triplet_df, seed)

    train_set = Dataset(data, sider, train_triplet, supp=8)
    val_set = Dataset(data, sider, val_triplet, supp=8, train=False)
    test_set = Dataset(data, sider, test_triplet, supp=8, train=False)

    BATCH_SIZE = params["batch_size"]
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    # 5. Modeltraining

    train_config = {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "max_epochs": 50,
        "patience": 4,
        "log_interval": 100,
        "val_interval": 2,  # set higher to fasten up the process of training
        "save_path": f"{os.path.join('models', f'model_hps')}.mdl"
    }

    writer = SummaryWriter()
    train_model(train_config, writer, train_loader, val_loader, device, BATCH_SIZE)

    # 6. Evaluation

    # load best model
    model.load_state_dict(torch.load(train_config["save_path"]))

    auc = auc_score(model, val_loader, device)
    daucPr = dauc_pr(model, val_loader, device)

    return auc # daucPr

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")

    start_time = time.time()

    study.optimize(objective, n_trials=1000, timeout=36000)

    elapsed_time = time.time() - start_time

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))