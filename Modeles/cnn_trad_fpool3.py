import torch
import torchaudio.transforms as T

class CNNTrad(torch.nn.Module):
    def __init__(self, params=None):
        super().__init__()

        if params is None:
            params = self.optimize_defaults()

        self.spec = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=40,
            win_length=400
        )

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=54,
                kernel_size=(20, 8),
                stride=(1, 1),
                padding=0
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            torch.nn.Conv2d(
                in_channels=54,
                out_channels=64,
                kernel_size=(10, 4),
                stride=(1, 1),
                padding=0
            ),
            torch.nn.ReLU()
        )

        self.flatten = torch.nn.Flatten()

        with torch.no_grad():
            dummy_input = torch.zeros(1, 16000)
            dummy_spec = self.spec(dummy_input)
            dummy_spec = torch.log(dummy_spec + 1e-10)
            dummy_spec = dummy_spec.unsqueeze(0)
            dummy_cnn = self.cnn(dummy_spec)
            flattened_size = dummy_cnn.numel() // dummy_cnn.shape[0]  

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(flattened_size, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(params["dropout"]),
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(params["dropout"]),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(params["dropout"]),
            torch.nn.Linear(128, params["fc_output_size"])
        )

    def forward(self, x):
        x = self.spec(x)
        x = torch.log(x + 1e-10)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = self.cnn(x)
        x = self.flatten(x)
        return self.dense(x)

    @staticmethod
    def optimize(trial):
        params = {
            "n_mels": 40,
            "n_fft": 1024,
            "hop_length": 256,
            "cnn1_out_channels": 54,
            "cnn2_out_channels": 64,
            "kernel_size": (20, 8),
            "pooling_kernel": (1, 3),
            "fc_output_size": 35,
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        }
        return params

    @staticmethod
    def optimize_defaults():
        return {
            "n_mels": 40,
            "n_fft": 1024,
            "hop_length": 256,
            "cnn1_out_channels": 54,
            "cnn2_out_channels": 64,
            "kernel_size": (20, 8),
            "pooling_kernel": (1, 3),
            "adaptive_pool_output": None,
            "fc_output_size": 35,
            "dropout": 0.0,
        }