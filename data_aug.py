import random
import torch
import torchaudio
from torch.utils.data import Dataset


class AugmentedSpeechCommands(Dataset):

    def __init__(self,
                 root="./",
                 subset="training"):

        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=root,
            download=True,
            subset=subset
        )

    def add_noise(self, waveform):
        noise_level = random.uniform(0.001, 0.01)
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise

    def time_shift(self, waveform):
        shift = random.randint(
            -int(0.1 * waveform.shape[1]),
             int(0.1 * waveform.shape[1])
        )

        shifted = torch.zeros_like(waveform)

        if shift > 0:
            shifted[:, shift:] = waveform[:, :-shift]
        elif shift < 0:
            shifted[:, :shift] = waveform[:, -shift:]
        else:
            shifted = waveform

        return shifted

    def gain(self, waveform):
        factor = random.uniform(0.8, 1.2)
        return torch.clamp(waveform * factor, -1.0, 1.0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        waveform, sample_rate, label, speaker_id, utterance_number = \
            self.dataset[idx]

        waveform = waveform.clone()

        augmentation = random.choice([
            "noise",
            "shift",
            "gain",
            "nothing"
        ])

        if augmentation == "noise":
            waveform = self.add_noise(waveform)

        elif augmentation == "shift":
            waveform = self.time_shift(waveform)

        elif augmentation == "gain":
            waveform = self.gain(waveform)

        return (
            waveform,
            sample_rate,
            label,
            speaker_id,
            utterance_number
        )