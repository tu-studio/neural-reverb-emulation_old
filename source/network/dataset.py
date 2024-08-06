import pickle
import os
import torch
from torch.utils.data import Dataset
from pedalboard.io import AudioFile
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, file_path):

        # Load the data from the file, either pickle or pt
        if file_path.endswith('.pt'):
            data = self.load_from_pt(file_path)
        elif file_path.endswith('.pkl'):
            data = self.load_from_pickle(file_path)

        # Compute sample rate and extract batches
        self.sample_rate = data['sample_rate']
        self.batches = [(dry, wet) for dry, wet in data['batches'] if dry.shape[0] != 1 and wet.shape[0] != 1]

        print(f"Removed {len(data['batches']) - len(self.batches)} samples from batches")
        print(f"Sample rate: {self.sample_rate} Hz")

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        dry_audio, wet_audio = self.batches[idx]

        return torch.tensor(dry_audio), torch.tensor(wet_audio)

    def get_sample_rate(self):
        return self.sample_rate
    
    @staticmethod
    def save_to_pickle(dry_audio_files, wet_audio_files, filename):
        def process_audio(file_path):
            with AudioFile(file_path) as f:
                audio = f.read(int(f.samplerate * f.duration))
                return audio, f.samplerate
            
        # Process the first file to get the sample rate
        _, sample_rate = process_audio(dry_audio_files[0])

        # Full audio files (dry and wet)
        batches = [(process_audio(dry)[0], process_audio(wet)[0]) for dry, wet in zip(dry_audio_files, wet_audio_files)]

        data = {
            'sample_rate': sample_rate,
            'batches': batches
        }

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def save_to_pt(dry_audio_files, wet_audio_files, filename):
        def process_audio(file_path):
            with AudioFile(file_path) as f:
                audio = f.read(int(f.samplerate * f.duration))
                return audio, f.samplerate

        # Process the first file to get the sample rate
        _, sample_rate = process_audio(dry_audio_files[0])

        # Full audio files (dry and wet)
        batches = [(process_audio(dry)[0], process_audio(wet)[0]) for dry, wet in zip(dry_audio_files, wet_audio_files)]

        data = {
            'sample_rate': sample_rate,
            'batches': batches
        }

        torch.save(data, filename)

    @staticmethod
    def load_from_pickle(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    
    @staticmethod
    def load_from_pt(filename):
        data = torch.load(filename)
        return data