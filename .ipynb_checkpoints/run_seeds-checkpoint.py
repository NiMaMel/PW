import os
import json
import numpy as np
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fs_model import Model
from dataset_modul import Dataset
from preprocessing import preprocessing
from utils import data_split, train_rf, train_model, combined_metrics, eval_rf, mean_scores

# Preparations

# set hardcoded seed to generate 10 random seeds for the experiments
np.random.seed(42)

# generate seeds for experiments
seeds = np.random.randint(0, 10000, size=10)

# cuda setting
if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    device = torch.device('cuda:%d' % device_id)
else:
    device = torch.device('cpu')
print(f"device set:{torch.cuda.get_device_name(device_id)}\n")

#default config
with open('default_config.json') as json_file:
    model_config = json.load(json_file)

#with open('test_config.json') as json_file:
#    model_config = json.load(json_file)

# 1. Load Dataset and create Tiplet-Df
sider = pd.read_csv("datasets/sider.csv")
triplets = [(i,j,row.iloc[j]) for i,row in sider.iterrows() for j in range(1,len(row))]
triplet_df = pd.DataFrame(triplets, columns=['mol_id', 'target_id', 'label'])

# 2. Preprocessing
data = preprocessing(sider)

# Experiments
val_scores = pd.DataFrame([])
test_scores = pd.DataFrame([])
rf_scores = pd.DataFrame([])
for i, seed in enumerate(seeds):
    print(f"Running experiment {i+1} with seed {seed}")

    # set current seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 3. Initialize Model

    model = Model(model_config["input_dim"], model_config["hidden_dim"],model_config["output_dim"], model_config["num_layers"], model_config["p"])
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=model_config["opt_lr"], weight_decay=model_config["weight_decay"])

    # 4. Train-split and Dataloader

    train_triplet, val_triplet, test_triplet = data_split(triplet_df, seed)

    train_set = Dataset(data, sider, train_triplet, supp=8)
    val_set = Dataset(data, sider, val_triplet, supp=8, train=False)
    test_set = Dataset(data, sider, test_triplet, supp=8, train=False)

    BATCH_SIZE = model_config["batch_size"]
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
        "val_interval": 1,  # set higher to fasten up the process of training
        "save_path": f"{os.path.join('models', f'model_{seed}_run{i+1}')}.mdl"
    }

    writer = SummaryWriter()
    train_model(train_config, writer, train_loader, val_loader, device, BATCH_SIZE)


    # 6. Evaluation

    # load best model
    model.load_state_dict(torch.load(train_config["save_path"]))

    val_score = combined_metrics(model, val_loader, device)
    test_score= combined_metrics(model, test_loader, device)

    val_score.insert(0, "Seed", seed)
    test_score.insert(0, "Seed", seed)

    # 7. Compare to RF Baseline
    y_hats_proba, y_hats_class, true_labels = train_rf(sider, data, test_triplet, seed, 1000, shuffle=True)
    rf_score = eval_rf(y_hats_class, true_labels)
    rf_score.insert(0, "Seed", seed)

    # update Score-Df's
    val_scores = pd.concat([val_scores, val_score])
    test_scores = pd.concat([test_scores, test_score])
    rf_scores = pd.concat([rf_scores, rf_score])

    print(f"Current Evaluations: Test-AUC:{test_score['AUC']}, Test-D-AUC PR:{test_score['D-AUC PR']}")
    print(f"Experiment {i+1} finished!\n")

print("Experiment done!\n")

avg_val_scores, avg_test_scores, avg_rf_scores = mean_scores(val_scores, test_scores, rf_scores)
print(f"{avg_val_scores=}\n{avg_test_scores=}\n{avg_rf_scores=}")