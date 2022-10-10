import os
import torch
import torchaudio
from torchvision.transforms.functional import resize
from custom_denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torch.utils.data import Dataset
from dataset import load_audio
from sklearn.preprocessing import MinMaxScaler

artist = 'ACDC'
USE_MELS = False

class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms):
        super().__init__()
        self.spectrograms = spectrograms
    
    def __getitem__(self, idx):
        return self.spectrograms[idx]

    def __len__(self):
        return len(self.spectrograms)


def spectrogram2audio(data, save=True, mels=True):
    data = torchaudio.functional.DB_to_amplitude(data)
    transforms = [torchaudio.transforms.InverseSpectrogram(n_fft=2048, hop_length=512)]
    if mels:
        transforms += [torchaudio.transforms.InverseMelScale(n_stft=2048, sample_rate=44100)]
        transforms.reverse()
    for transform in transforms:
        data = transform(data)
    saved = False
    idx = 0
    while not saved:
        f = f'./generated_{artist}_audio_{idx}.mp3'
        if os.path.exists(f):
            idx += 1
            continue
        torchaudio.save(f, data, sample_rate=44100)
        saved = True
    return data

def train_diffusion_model(model, training_steps=100000):
    ...

if __name__ == '__main__':
    music_dir = f'/home/user/Projects/spotify-machine-learning/all_artists/{artist}'
    training_data = load_audio(music_dir, mel=USE_MELS)
    training_data = [data.view(-1, *data.size()) for data in training_data]
    training_data = torch.concat(training_data)

    min_val_ = training_data.min()
    val_range_ = training_data.max() - training_data.min()
    training_data = (training_data - min_val_) / val_range_

    old_dims = training_data.size()[-2:]
    max_dim = max(old_dims)
    training_data = resize(training_data, (max_dim, max_dim))

    training_dataset = SpectrogramDataset(training_data)

    model = Unet(
        dim=8,
        channels=2,
        dim_mults=(1,2)
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=max_dim,
        timesteps=1000,
        loss_type='l1'
    ).cuda()

    trainer = Trainer(diffusion, dataset=training_dataset, train_batch_size=4)
    trainer.train()

    samples = diffusion.sample(batch_size=5)
    samples = resize(samples, old_dims)
    samples = samples * val_range_ + min_val_
    generated_audio = [spectrogram2audio(samples[idx] for idx in range(len(samples)))]
