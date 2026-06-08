import torch
import torchaudio.transforms as T

class DoubleCNNModule(torch.nn.Module):

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

        self.cnn1D = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=params["cnn1d_out_channels_1"],
                kernel_size=params["kernel_size_1d"],
                padding=params["kernel_size_1d"] // 2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(params["pooling_kernel_1d"]),
            torch.nn.Dropout(params["dropout"]),

            torch.nn.Conv1d(
                in_channels=params["cnn1d_out_channels_1"],
                out_channels=params["cnn1d_out_channels_2"],
                kernel_size=params["kernel_size_1d"],
                padding=params["kernel_size_1d"] // 2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(params["pooling_kernel_1d"]),
            torch.nn.Dropout(params["dropout"])
        )
        self.pool1D = torch.nn.AdaptiveAvgPool1d(params["adaptive_pool_1d_output"])

        self.cnn2D = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=params["cnn2d_out_channels_1"],
                kernel_size=params["kernel_size_2d"],
                padding=params["kernel_size_2d"] // 2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(params["pooling_kernel_2d"]),
            torch.nn.Dropout2d(params["dropout"]),

            torch.nn.Conv2d(
                in_channels=params["cnn2d_out_channels_1"],
                out_channels=params["cnn2d_out_channels_2"],
                kernel_size=params["kernel_size_2d"],
                padding=params["kernel_size_2d"] // 2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(params["pooling_kernel_2d"]),
            torch.nn.Dropout2d(params["dropout"])
        )
        self.pool2D = torch.nn.AdaptiveAvgPool2d(params["adaptive_pool_2d_output"])

        self.flatten = torch.nn.Flatten()

        pooled_1d_size = params["cnn1d_out_channels_2"] * params["adaptive_pool_1d_output"]
        pooled_2d_height, pooled_2d_width = params["adaptive_pool_2d_output"]
        pooled_2d_size = params["cnn2d_out_channels_2"] * pooled_2d_height * pooled_2d_width

        total_fc_input = pooled_1d_size + pooled_2d_size

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(total_fc_input, params["fc_hidden_size"]),
            torch.nn.ReLU(),
            torch.nn.Dropout(params["dropout"]),
            torch.nn.Linear(params["fc_hidden_size"], params["fc_output_size"])
        )

    def forward(self, x):

        x_1D = self.cnn1D(x)
        x_1D = self.pool1D(x_1D)
        x_1D = self.flatten(x_1D)

        x_2D = self.spec(x)
        x_2D = torch.log(x_2D + 1e-10)
        if x_2D.dim() == 3:
            x_2D = x_2D.unsqueeze(0)
        x_2D = self.cnn2D(x_2D)
        x_2D = self.pool2D(x_2D)
        x_2D = self.flatten(x_2D)

        x = torch.cat([x_1D, x_2D], dim=1)

        return self.dense(x)

    @staticmethod
    def optimize(trial):
        params = {

            "n_mels": trial.suggest_int("n_mels", 32, 128, step=16),
            "n_fft": trial.suggest_int("n_fft", 512, 2048, step=256),
            "hop_length": trial.suggest_int("hop_length", 256, 1024, step=64),

            "cnn1d_out_channels_1": trial.suggest_int("cnn1d_out_channels_1", 16, 128, step=16),
            "cnn1d_out_channels_2": trial.suggest_int("cnn1d_out_channels_2", 32, 256, step=16),
            "kernel_size_1d": trial.suggest_int("kernel_size_1d", 3, 7, step=2),
            "pooling_kernel_1d": trial.suggest_int("pooling_kernel_1d", 2, 4),
            "adaptive_pool_1d_output": trial.suggest_int("adaptive_pool_1d_output", 8, 32, step=4),

            "cnn2d_out_channels_1": trial.suggest_int("cnn2d_out_channels_1", 16, 128, step=16),
            "cnn2d_out_channels_2": trial.suggest_int("cnn2d_out_channels_2", 32, 256, step=16),
            "kernel_size_2d": trial.suggest_int("kernel_size_2d", 3, 7, step=2),
            "pooling_kernel_2d": trial.suggest_int("pooling_kernel_2d", 2, 4),
            "adaptive_pool_2d_output": trial.suggest_categorical(
                "adaptive_pool_2d_output", [(4, 4), (8, 8), (2, 2)]
            ),

            "fc_hidden_size": trial.suggest_int("fc_hidden_size", 64, 512, step=64),
            "fc_output_size": 35, 
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        }
        return params

    @staticmethod
    def optimize_defaults():
        return {
            "n_mels": 32,
            "n_fft": 1536,
            "hop_length": 320,
            "cnn1d_out_channels_1": 32,
            "cnn1d_out_channels_2": 64,
            "kernel_size_1d": 3,
            "pooling_kernel_1d": 2,
            "adaptive_pool_1d_output": 16,
            "cnn2d_out_channels_1": 32,
            "cnn2d_out_channels_2": 64,
            "kernel_size_2d": 3,
            "pooling_kernel_2d": 2,
            "adaptive_pool_2d_output": (4, 4),
            "fc_hidden_size": 128,
            "fc_output_size": 35,
            "dropout": 0.2,
        }