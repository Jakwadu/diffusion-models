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
    if save:
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

class AudioClipGenerator:
    def __init__(self, 
                 audio_dir, 
                 model, 
                 training_batch_size=16,
                 diffusion_timesteps=1000,
                 training_steps=100000):
        
        self.configure_training(audio_dir, 
                                training_batch_size, 
                                diffusion_timesteps,
                                training_steps)
        self.model = model

    def configure_training(self, dir, training_batch_size, diffusion_timesteps, training_steps):
        training_data = load_audio(dir, mel=USE_MELS)
        training_data = [data.view(-1, *data.size()) for data in training_data]
        training_data = torch.concat(training_data)

        self.min_val_ = training_data.min()
        self.val_range_ = training_data.max() - training_data.min()
        training_data = (training_data - self.min_val_) / self.val_range_

        self.old_dims = training_data.size()[-2:]
        self.max_dim = max(self.old_dims)
        training_data = resize(training_data, (self.max_dim, self.max_dim))

        self.dataset = SpectrogramDataset(training_data)

        self.diffusion = GaussianDiffusion(
            self.model,
            image_size=self.max_dim,
            timesteps=diffusion_timesteps,
            loss_type='l1'
        ).cuda()

        self.trainer = Trainer(self.diffusion, 
                               dataset=self.dataset, 
                               train_batch_size=training_batch_size,
                               train_num_steps=training_steps)

    def train(self):
        self.trainer.train()

    def sample(self, batch_size=5, save_audio=True):
        samples = self.diffusion.sample(batch_size=batch_size)
        samples = resize(samples, self.old_dims)
        samples = samples * self.val_range_ + self.min_val_
        generated_audio = [spectrogram2audio(samples[idx], save=save_audio) for idx in range(len(samples))]
        return generated_audio


if __name__ == '__main__':
    music_dir = f'/home/user/Projects/spotify-machine-learning/all_artists/{artist}'

    model = Unet(
        dim=8,
        channels=2,
        dim_mults=(1,2)
    ).cuda()

    audio_generator = AudioClipGenerator(music_dir, model)

    samples = audio_generator.sample(batch_size=5)
