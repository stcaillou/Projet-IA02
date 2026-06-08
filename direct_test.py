import torch
import sounddevice as sd
from scipy.io.wavfile import write

"""
print(sd.query_devices())
device_info = sd.query_devices(10)
print(device_info)
"""

model = ...()

model.load_state_dict(torch.load(f"best_model_{model.__class__.__name__}.pth", weights_only=True))


device_index = 10

sd.default.device = device_index

duration = 1 
fs = int(sd.query_devices(device_index)["default_samplerate"])
filename = "capture.wav"

print("Enregistrement en cours...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait() 
print("Enregistrement terminé.")

audio = torch.from_numpy(audio.T).float()

if fs != 16000 : 
    resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
    audio = resampler(audio)


model.eval()

softmax_eval_live = torch.nn.Softmax(dim=1)

plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
plt.title("Distribution")
plt.bar(labels,softmax_eval_live(model(audio))[0].detach().numpy())

write(filename, 16000, audio.T.numpy())
print(f"Audio sauvegardé dans {filename}")

