import torch
import torchaudio
import matplotlib.pyplot as plt
from source.network.ravepqmf import PQMF
from source.network.metrics import multiscale_stft, plot_spectrums_tensorboard, plot_distance_spectrums_tensorboard
from torch.utils.tensorboard import SummaryWriter

# Helper function to plot spectrogram
def plot_spectrogram(stft, title):
    plt.figure(figsize=(10, 4))
    plt.imshow(stft.squeeze().numpy(), aspect='auto', origin='lower')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

# Helper function to plot distance spectrums
def plot_distance_spectrums(x, y, scales):
    x_stfts = multiscale_stft(x, scales, 0.75)
    y_stfts = multiscale_stft(y, scales, 0.75)
    
    for i, scale in enumerate(scales):
        lin_dist = (x_stfts[i] - y_stfts[i]).abs()[0]
        log_dist = (torch.log(x_stfts[i] + 1e-7) - torch.log(y_stfts[i] + 1e-7)).abs()[0]
        
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.imshow(lin_dist.numpy(), aspect='auto', origin='lower')
        plt.title(f'Linear Distance (scale {scale})')
        plt.colorbar()
        
        plt.subplot(122)
        plt.imshow(log_dist.numpy(), aspect='auto', origin='lower')
        plt.title(f'Log Distance (scale {scale})')
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()

# Load an audio file
audio_path = "80s Beat 90 bpm_dry.wav"
waveform, sample_rate = torchaudio.load(audio_path)
waveform = waveform.unsqueeze(0)  # Add batch dimension

# Initialize PQMF
n_bands = 64
pqmf = PQMF(100, n_bands)

# 1. PQMF Decomposition
decomposed = pqmf(waveform)
print(f"Original shape: {waveform.shape}")
print(f"Decomposed shape: {decomposed.shape}")

# 2. Listen to audio after decomposition
# Note: This will sound distorted as it's decomposed
torchaudio.save("decomposed_audio.wav", decomposed.squeeze().cpu(), sample_rate)
print("Saved decomposed audio as 'decomposed_audio.wav'")

# 3. PQMF Reconstruction
reconstructed = pqmf.inverse(decomposed)
print(f"Reconstructed shape: {reconstructed.shape}")

# Save reconstructed audio
torchaudio.save("reconstructed_audio.wav", reconstructed.squeeze().cpu(), sample_rate)
print("Saved reconstructed audio as 'reconstructed_audio.wav'")

# 4. Spectral analysis
scales = [2048, 1024, 512, 256, 128]
original_stft = multiscale_stft(waveform, scales, 0.75)
reconstructed_stft = multiscale_stft(reconstructed, scales, 0.75)

for i, scale in enumerate(scales):
    plot_spectrogram(original_stft[i][0], f"Original STFT (scale {scale})")
    plot_spectrogram(reconstructed_stft[i][0], f"Reconstructed STFT (scale {scale})")

# Plot distance spectrums
plot_distance_spectrums(waveform, reconstructed, scales)

# Compare waveforms
plt.figure(figsize=(12, 4))
plt.plot(waveform.squeeze().numpy(), label='Original')
plt.plot(reconstructed.squeeze().detach().numpy(), label='Reconstructed')
plt.legend()
plt.title("Original vs Reconstructed Waveform")
plt.show()

# Compute difference
diff = waveform - reconstructed
print(f"Max absolute difference: {diff.abs().max().item()}")
print(f"Mean absolute difference: {diff.abs().mean().item()}")

# Plot difference
plt.figure(figsize=(12, 4))
plt.plot(diff.squeeze().detach().numpy())
plt.title("Difference between Original and Reconstructed Waveform")
plt.show()