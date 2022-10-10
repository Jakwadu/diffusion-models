import os
import torch
import torchaudio
import numpy as np
from matplotlib import pyplot as plt

def load_audio(dir, spectrograms=True, mel=True, standardise=False, snip_len=3):
    audio_data = []
    
    for f in os.listdir(dir):
        
        if '.mp3' in f:
            audio, sr = torchaudio.load(os.path.join(dir,f), normalize=True)
            n_snip_samples = sr*snip_len
            audio = audio[:,audio.size()[1]%n_snip_samples:]
            audio_snips = [audio[:,idx:idx+n_snip_samples] for idx in range(0, audio.size()[1], n_snip_samples)]
            
            for snip in audio_snips:
                if spectrograms:
                    if mel:
                        transforms = [torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=2048, hop_length=512), 
                                      torchaudio.transforms.AmplitudeToDB()]                    
                    else:
                        transforms = [torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=512), 
                                      torchaudio.transforms.AmplitudeToDB()]  
                    
                    for transform in transforms:
                        snip = transform(snip)
                    
                    if standardise:
                        snip = (snip - torch.mean(snip)) / torch.std(snip)
                
                audio_data.append(snip)
    
    return audio_data