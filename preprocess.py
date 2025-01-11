import torchaudio
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import argparse

def preprocess_yesno(root_path, output_path, sample_rate=8000):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    audio_files = sorted([x for x in os.listdir(root_path) if x.endswith('.wav')])
    if not audio_files:
        raise FileNotFoundError(f'No hay archivos en {root_path}')
    resampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=sample_rate)
    processed_audios = []
    labels = []
    for file in audio_files:
        waveform, sr = torchaudio.load(os.path.join(root_path, file))
        if sr != sample_rate:
            waveform = resampler(waveform)
        processed_audios.append(waveform)
        label = [int(yn) for yn in file[:-4].split("_")]
        labels.append(label)
    max_length = max(waveform.shape[1] for waveform in processed_audios)
    padded_audios = [F.pad(waveform, (0, max_length - waveform.shape[1])) for waveform in processed_audios]
    audio_tensor = torch.stack(padded_audios)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    combined_tensor = torch.cat([audio_tensor.view(audio_tensor.size(0), -1), labels_tensor], dim=1)
    output_file = os.path.join(output_path, "yesno_tensor.pt")
    torch.save(combined_tensor, output_file)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True, help="Ruta de los audios originales")
    parser.add_argument("--output_path", type=str, required=True, help="Ruta para guardar el tensor combinado")
    parser.add_argument("--sample_rate", type=int, default=8000, help="Frecuencia de muestreo deseada")
    args = parser.parse_args()
    preprocess_yesno(args.root_path, args.output_path, args.sample_rate)