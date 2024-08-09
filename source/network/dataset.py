import pickle
import os
import torch
from torch.utils.data import Dataset
from pedalboard.io import AudioFile
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, file_path):
        if file_path.endswith('.pt'):
            data = self.load_from_pt(file_path)
        elif file_path.endswith('.pkl'):
            data = self.load_from_pickle(file_path)

        self.sample_rate = data['sample_rate']
        self.dry_wet_pairs = self.process_audio_pairs(data['dry_wet_pairs'])

        print(f"Sample rate: {self.sample_rate} Hz")

    def __len__(self):
        return len(self.dry_wet_pairs)

    def __getitem__(self, idx):
        dry_audio, wet_audio = self.dry_wet_pairs[idx]
        return torch.tensor(dry_audio, dtype=torch.float32), torch.tensor(wet_audio, dtype=torch.float32)

    def get_sample_rate(self):
        return self.sample_rate
    
    @staticmethod
    def separate_channels(audio):
        # Convert to numpy array if it's not already
        audio = np.array(audio)
        
        # If it's a 1D array, it's already a single channel
        if audio.ndim == 1:
            return [audio]
        
        # If it's a 2D array, separate channels
        elif audio.ndim == 2:
            # Check if channels are the first or second dimension
            if audio.shape[0] == 1 or audio.shape[0] == 2:
                return [channel for channel in audio]
            else:
                return [audio.T[0], audio.T[1]] if audio.shape[1] == 2 else [audio.T[0]]
        
        else:
            raise ValueError(f"Unexpected audio shape: {audio.shape}. Expected 1D or 2D array.")

    @staticmethod
    def process_audio_pairs(pairs):
        processed_pairs = []
        for dry, wet in pairs:
            dry_channels = AudioDataset.separate_channels(dry)
            wet_channels = AudioDataset.separate_channels(wet)
            
            # Ensure dry and wet have the same number of channels
            min_channels = min(len(dry_channels), len(wet_channels))
            
            # Ensure each channel is 1D
            dry_channels = [channel.squeeze() for channel in dry_channels[:min_channels]]
            wet_channels = [channel.squeeze() for channel in wet_channels[:min_channels]]
            
            processed_pairs.extend(zip(dry_channels, wet_channels))
        return processed_pairs

    @staticmethod
    def save_to_pickle(dry_audio_files, wet_audio_files, filename):
        def process_audio(file_path):
            with AudioFile(file_path) as f:
                audio = f.read(int(f.samplerate * f.duration))
                return audio, f.samplerate
            
        _, sample_rate = process_audio(dry_audio_files[0])
        dry_wet_pairs = [(process_audio(dry)[0], process_audio(wet)[0]) 
                         for dry, wet in zip(dry_audio_files, wet_audio_files)]

        data = {
            'sample_rate': sample_rate,
            'dry_wet_pairs': dry_wet_pairs
        }

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def save_to_pt(dry_audio_files, wet_audio_files, filename):
        def process_audio(file_path):
            with AudioFile(file_path) as f:
                audio = f.read(int(f.samplerate * f.duration))
                return audio, f.samplerate

        _, sample_rate = process_audio(dry_audio_files[0])
        dry_wet_pairs = [(process_audio(dry)[0], process_audio(wet)[0]) 
                         for dry, wet in zip(dry_audio_files, wet_audio_files)]

        data = {
            'sample_rate': sample_rate,
            'dry_wet_pairs': dry_wet_pairs
        }

        torch.save(data, filename)

    @staticmethod
    def load_from_pickle(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    
    @staticmethod
    def load_from_pt(filename):
        data = torch.load(filename, weights_only=False)
        return data