import torch
import torchaudio.transforms as T

class CNNModule(torch.nn.Module):
    def __init__(self,params=None):
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
            torch.nn.Conv2d(1, params["cnn1_out_channels"], kernel_size=params["kernel_size"], padding=params["kernel_size"] // 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(params["pooling_kernel"]),
            torch.nn.Dropout2d(params["dropout"]),
            torch.nn.Conv2d(params["cnn1_out_channels"], params["cnn2_out_channels"], kernel_size=params["kernel_size"], padding=params["kernel_size"] // 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(params["pooling_kernel"]),
            torch.nn.Dropout2d(params["dropout"])
        )

        self.pool = torch.nn.AdaptiveAvgPool2d(params["adaptive_pool_output"])

        self.flatten = torch.nn.Flatten()

        pooled_height, pooled_width = params["adaptive_pool_output"]

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(params["cnn2_out_channels"] * pooled_height * pooled_width, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(params["dropout"]),
            torch.nn.Linear(128, params["fc_output_size"])
        )

    def forward(self, x):
        x = self.spec(x)
        x = torch.log(x + 1e-10)

        #Permet de passer un seul élément pendant la phase de production
        if len(x.shape) < 3:
            x = x.unsqueeze(0)

        x = self.cnn(x)

        x = self.pool(x)
        x = self.flatten(x)
        return self.dense(x)
    
    @staticmethod
    def optimize(trial):
        #Dans l'idée c'est pour optimiser le pipeline de optuna
        params = {
            "n_mels": trial.suggest_int("n_mels", 32, 128, step=16),
            "n_fft": trial.suggest_int("n_fft", 512, 2048, step=256),
            "hop_length": trial.suggest_int("hop_length", 256, 1024, step=64),


            "cnn1_out_channels": trial.suggest_int("cnn1_out_channels", 16, 128, step=16),
            "cnn2_out_channels": trial.suggest_int("cnn2_out_channels", 32, 256, step=16),
            "kernel_size": trial.suggest_int("kernel_size", 3, 7, step=2),
            "pooling_kernel": trial.suggest_int("pooling_kernel", 2, 4),

            "adaptive_pool_output": trial.suggest_categorical(
                "adaptive_pool_output",
                [(4, 4), (8, 8), (2, 2)]
            ),

            "fc_output_size": 35, 
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        }
        return params
    
    @staticmethod
    def optimize_defaults():
        #Ces paramètres ont était déterminées avec optuna
        return {
            "n_mels": 48,
            "n_fft": 1536,
            "hop_length": 384,

            "cnn1_out_channels": 48,
            "cnn2_out_channels": 208,
            "kernel_size": 7,
            "pooling_kernel": 2,

            "adaptive_pool_output": (8, 8),

            "fc_output_size": 35, 
            "dropout": 0.2,
        }