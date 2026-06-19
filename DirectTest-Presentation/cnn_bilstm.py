import torch
import torchaudio.transforms as T

class CNNBiLSTM(torch.nn.Module):
    def __init__(self, params=None):
        super().__init__()

        if params is None:
            params = self.optimize_defaults()

        self.spec = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=params["n_fft"],
            hop_length=params["hop_length"],
            n_mels=params["n_mels"]
        )

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=params["cnn1_out_channels"],
                kernel_size=params["kernel_size"],
                padding=params["kernel_size"] // 2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(params["pooling_kernel"]),
            torch.nn.Dropout2d(params["dropout"]),

            torch.nn.Conv2d(
                in_channels=params["cnn1_out_channels"],
                out_channels=params["cnn2_out_channels"],
                kernel_size=params["kernel_size"],
                padding=params["kernel_size"] // 2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(params["pooling_kernel"]),
            torch.nn.Dropout2d(params["dropout"])
        )

        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d(params["adaptive_pool_output"])

        pooled_height, pooled_width = params["adaptive_pool_output"]
        cnn_output_features = params["cnn2_out_channels"] * pooled_width
        self.lstm = torch.nn.LSTM(
            input_size=cnn_output_features, 
            hidden_size=params["lstm_hidden_size"],
            num_layers=params["lstm_num_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=params["dropout"] if params["lstm_num_layers"] > 1 else 0.0
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(params["lstm_hidden_size"] * 2, params["fc_hidden_size"]),
            torch.nn.ReLU(),
            torch.nn.Dropout(params["dropout"]),
            torch.nn.Linear(params["fc_hidden_size"], params["fc_output_size"])
        )

    def forward(self, x):
        x = self.spec(x)
        x = torch.log(x + 1e-10)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = self.cnn(x)
        x = self.adaptive_pool(x)

        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, x.size(1), -1)

        x, (h_n, c_n) = self.lstm(x)

        forward_hidden = h_n[-2] 
        backward_hidden = h_n[-1] 
        x = torch.cat([forward_hidden, backward_hidden], dim=1)

        return self.fc(x)

    @staticmethod
    def optimize(trial):
        params = {
            "n_mels": trial.suggest_int("n_mels", 32, 128, step=16),
            "n_fft": trial.suggest_int("n_fft", 512, 2048, step=256),
            "hop_length": trial.suggest_int("hop_length", 256, 1024, step=64),

            "cnn1_out_channels": trial.suggest_int("cnn1_out_channels", 16, 128, step=16),
            "cnn2_out_channels": trial.suggest_int("cnn2_out_channels", 32, 256, step=16),
            "kernel_size": trial.suggest_int("kernel_size", 3, 7, step=2),
            "pooling_kernel": trial.suggest_int("pooling_kernel", 2, 4),
            "adaptive_pool_output": trial.suggest_categorical(
                "adaptive_pool_output", [(4, 4), (8, 8), (2, 2)]
            ),

            "lstm_hidden_size": trial.suggest_int("lstm_hidden_size", 64, 256, step=32),
            "lstm_num_layers": trial.suggest_int("lstm_num_layers", 1, 3),

            "fc_hidden_size": trial.suggest_int("fc_hidden_size", 64, 512, step=64),
            "fc_output_size": 35,
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        }
        return params

    @staticmethod
    def optimize_defaults():
        return {
            "n_mels": 64,
            "n_fft": 1536,
            "hop_length": 448,

            "cnn1_out_channels": 48,
            "cnn2_out_channels": 224,
            "kernel_size": 7,
            "pooling_kernel": 2,
            "adaptive_pool_output": (8, 8),

            "lstm_hidden_size": 192,
            "lstm_num_layers": 1,
            "fc_hidden_size": 320,
            "fc_output_size": 35,
            "dropout": 0.1,
        }

if __name__ == "__main__":
    model = CNNBiLSTM()
    output = model(torch.zeros(1, 16000))
    print(output)