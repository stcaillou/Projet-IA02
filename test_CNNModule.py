import torch
import torchaudio
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import time

start_time = time.time()


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

test_loader = DataLoader(
    data_testing,
    batch_size=32,
    shuffle=False, 
    collate_fn=pre_process_batch
)

from ... import ...

device = "cuda"
model = ...()
model.to(device)
model.load_state_dict(torch.load(f"best_model_{model.__class__.__name__}.pth", weights_only=True))
model.eval()

criterion = torch.nn.CrossEntropyLoss()
all_labels = []
all_predictions = []
test_loss = 0.0

with torch.no_grad():
    for waveforms, labels_batch in test_loader:
        waveforms = waveforms.to(device)
        labels_batch = labels_batch.to(device)

        outputs = model(waveforms)
        loss = criterion(outputs, labels_batch)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels_batch.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())


test_loss /= len(test_loader)
accuracy = accuracy_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions, average="weighted")

print("\n--- Métriques sur le jeu de test ---")
print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F1-Score (weighted): {f1:.4f}")

end_time = time.time()
elapsed_time = end_time - start_time

print("\n--- Temps de calcul ---")
print(f"Temps écoulé : {elapsed_time:.2f} secondes")