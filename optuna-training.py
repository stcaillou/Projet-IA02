import torch
import optuna
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn as nn

data = torchaudio.datasets.SPEECHCOMMANDS(download=True, root="./")
data_train = torchaudio.datasets.SPEECHCOMMANDS(download=True, root="./", subset="training")
data_validation = torchaudio.datasets.SPEECHCOMMANDS(download=True, root="./", subset="validation")
data_testing = torchaudio.datasets.SPEECHCOMMANDS(download=True, root="./", subset="testing")
labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

label_to_idx = {l: i for i, l in enumerate(labels)}

def collate_fn(batch):
    target_len = 16000
    waves, labs = [], []

    for waveform, sr, label_str, _, _ in batch:

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.shape[1] < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_len]

        label = label_to_idx.get(label_str)
        if label is None:
            continue

        waves.append(waveform)
        labs.append(label)

    if len(waves) == 0:
        return torch.empty(0), torch.empty(0)

    return torch.stack(waves), torch.tensor(labs)


def get_loaders(batch_size):

    train_loader = DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,         
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        data_validation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, val_loader



device = "cuda"

def objective(trial, modelClass):

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    #Décrit comme une régularisation pour l'optimizer
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)

    params = modelClass.optimize(trial)

    # Initialisation du modèle
    model = modelClass(params).to(device)

    train_loader, val_loader = get_loaders(batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0
    patience = 2
    wait = 0

    for epoch in range(3):

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total

        trial.report(acc, epoch)

        #Permet de faire un élagage
        if trial.should_prune():
            raise optuna.TrialPruned()

        #EarlyStopping
        if acc > best_acc:
            best_acc = acc
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    return best_acc


from cnn_one_fpool3 import CNNOneFPool3
from functools import partial

if __name__ == "__main__":
    #Permet le multi-processing
    mp.set_start_method("spawn", force=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, modelClass=CNNOneFPool3), n_trials=50)

    print("Meilleur paramètres :", study.best_params)
    print("Meilleur accuracy :", study.best_value)