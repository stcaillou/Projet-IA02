import torchaudio
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torchaudio.transforms as T
import torch.multiprocessing as mp
import time
import torch.nn as nn

data = torchaudio.datasets.SPEECHCOMMANDS(download=True, root="./")
data_train = torchaudio.datasets.SPEECHCOMMANDS(download=True, root="./", subset="training")
data_validation = torchaudio.datasets.SPEECHCOMMANDS(download=True, root="./", subset="validation")
data_testing = torchaudio.datasets.SPEECHCOMMANDS(download=True, root="./", subset="testing")
labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']


label_to_idx = {l: i for i, l in enumerate(labels)}

def pre_process_batch(batch):
    target_len = 16000
    new_batch = []
    
    for waveform, sample_rate, label_str, speaker_id, utterance_number in batch:

        current_len = waveform.shape[1]
        if current_len < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - current_len))

        label = label_to_idx.get(label_str, None)
        if label is None:
            continue

        new_batch.append((waveform, label))

    waveforms = torch.stack([item[0] for item in new_batch])
    labels_tensor = torch.tensor([item[1] for item in new_batch])
    
    return waveforms, labels_tensor
 


from ... import ...
lr =   ...  
wd =  ...
batch_size = ...

train_loader = DataLoader(
    data_train, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=pre_process_batch 
)

val_loader = DataLoader(
    data_validation, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=pre_process_batch 
)


##Boucle d'entrainement
if __name__ == "__main__":
    ##Permet de faire du parallélisme
    mp.set_start_method("spawn", force=True)
    device = "cuda"

    start_time = time.perf_counter()

    model = ...()
    model.to("cuda")
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train() 

    epochs = 20
    best_loss = float('inf')
    patience = 5
    trigger_times = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (waveforms, labels_batch) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for waveforms, labels_batch in val_loader:
                waveforms = waveforms.to(device)
                labels_batch = labels_batch.to(device)
                outputs = model(waveforms)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        #EarlyStopping
        if val_loss < best_loss:
            best_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), f'best_model_{model.__class__.__name__}.pth')
            print(f"Nouveau meilleur modèle sauvegardé (Validation Loss: {best_loss:.4f})")
        else:
            trigger_times += 1
            print(f"Pas d'amélioration.")
            if trigger_times >= patience:
                print(f"Early Stopping déclenché. Epoque : {epoch+1}")
                break
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"\nTemps total d'entraînement : {total_time:.2f} secondes")
    exit()