import torch
import torch.nn as nn
from scipy.signal import kaiserord, firwin
from scipy.optimize import fmin
import math
import numpy as np
from einops import rearrange

# Helper Functions

def reverse_alternate_subbands(input_tensor):
    """
    Reverses the sign of alternate subbands in the input tensor.
    
    Args:
    input_tensor (torch.Tensor): Input tensor to be modified.
    
    Returns:
    torch.Tensor: Modified tensor with alternating subbands reversed.
    """
    mask = torch.ones_like(input_tensor)
    mask[..., 1::2, ::2] = -1
    return input_tensor * mask

def pad_to_next_power_of_two(input_tensor):
    """
    Pads the input tensor to the next power of two in the last dimension.
    
    Args:
    input_tensor (torch.Tensor): Input tensor to be padded.
    
    Returns:
    torch.Tensor: Padded tensor with last dimension as the next power of two.
    """
    current_size = input_tensor.shape[-1]
    next_power_of_two = 2**math.ceil(math.log2(current_size))
    padding_size = next_power_of_two - current_size
    return nn.functional.pad(input_tensor, (padding_size // 2, padding_size // 2 + int(padding_size % 2)))

def ensure_odd_length(input_tensor):
    """
    Ensures the input tensor has an odd length in the last dimension.
    
    Args:
    input_tensor (torch.Tensor): Input tensor to be modified.
    
    Returns:
    torch.Tensor: Tensor with odd length in the last dimension.
    """
    if input_tensor.shape[-1] % 2 == 0:
        return nn.functional.pad(input_tensor, (0, 1))
    return input_tensor

def create_cosine_modulated_filterbank(prototype_filter, num_subbands):
    """
    Creates a bank of cosine modulated filters from a prototype filter.
    
    Args:
    prototype_filter (torch.Tensor): Prototype lowpass filter.
    num_subbands (int): Number of subbands in the filterbank.
    
    Returns:
    torch.Tensor: Cosine modulated filterbank.
    """
    subband_indices = torch.arange(num_subbands).reshape(-1, 1)
    filter_length = prototype_filter.shape[-1]
    time_indices = torch.arange(-(filter_length // 2), filter_length // 2 + 1)
    
    phase_shift = (-1)**subband_indices * math.pi / 4
    modulation = torch.cos((2 * subband_indices + 1) * math.pi / (2 * num_subbands) * time_indices + phase_shift)
    
    return 2 * prototype_filter * modulation

def design_kaiser_lowpass(cutoff_freq, attenuation, num_taps=None):
    """
    Designs a Kaiser lowpass filter.
    
    Args:
    cutoff_freq (float): Normalized cutoff frequency (0 to 1).
    attenuation (float): Desired stopband attenuation in dB.
    num_taps (int, optional): Number of filter taps. If None, it's automatically determined.
    
    Returns:
    numpy.ndarray: Designed Kaiser lowpass filter coefficients.
    """
    if num_taps is None:
        num_taps, beta = kaiserord(attenuation, cutoff_freq / np.pi)
        num_taps = 2 * (num_taps // 2) + 1  # Ensure odd length
    else:
        _, beta = kaiserord(attenuation, cutoff_freq / np.pi)
    
    return firwin(num_taps, cutoff_freq / np.pi, window=('kaiser', beta), scale=False)

def compute_stopband_attenuation(cutoff_freq, attenuation, num_subbands, num_taps):
    """
    Computes the stopband attenuation for filter optimization.
    
    Args:
    cutoff_freq (float): Normalized cutoff frequency.
    attenuation (float): Desired stopband attenuation in dB.
    num_subbands (int): Number of subbands.
    num_taps (int): Number of filter taps.
    
    Returns:
    float: Maximum stopband attenuation.
    """
    prototype_filter = design_kaiser_lowpass(cutoff_freq, attenuation, num_taps)
    conv_result = np.convolve(prototype_filter, prototype_filter[::-1], "full")
    downsampled_result = abs(conv_result[conv_result.shape[-1] // 2::2 * num_subbands][1:])
    return np.max(downsampled_result)

def optimize_prototype_filter(attenuation, num_subbands, num_taps=None):
    """
    Optimizes the prototype lowpass filter for the PQMF.
    
    Args:
    attenuation (float): Desired stopband attenuation in dB.
    num_subbands (int): Number of subbands.
    num_taps (int, optional): Number of filter taps. If None, it's automatically determined.
    
    Returns:
    numpy.ndarray: Optimized prototype filter coefficients.
    """
    optimal_cutoff = fmin(lambda w: compute_stopband_attenuation(w, attenuation, num_subbands, num_taps), 1 / num_subbands, disp=0)[0]
    return design_kaiser_lowpass(optimal_cutoff, attenuation, num_taps)

# PQMF Analysis and Synthesis Functions

def pqmf_analysis(input_signal, filter_bank, use_polyphase=True):
    """
    Performs the analysis (decomposition) part of PQMF.
    
    Args:
    input_signal (torch.Tensor): Input signal to analyze (B x 1 x T).
    filter_bank (torch.Tensor): Filter bank coefficients (M x T).
    use_polyphase (bool): Whether to use polyphase implementation.
    
    Returns:
    torch.Tensor: Analyzed signal.
    """
    if use_polyphase:
        input_signal = rearrange(input_signal, "b c (t m) -> b (c m) t", m=filter_bank.shape[0])
        filter_bank = rearrange(filter_bank, "c (t m) -> c m t", m=filter_bank.shape[0])
    
    analyzed_signal = nn.functional.conv1d(input_signal, filter_bank, padding=filter_bank.shape[-1] // 2)[..., :-1]
    return analyzed_signal

def pqmf_synthesis(input_signal, filter_bank, use_polyphase=True):
    """
    Performs the synthesis (reconstruction) part of PQMF.
    
    Args:
    input_signal (torch.Tensor): Input signal to synthesize (B x M x T).
    filter_bank (torch.Tensor): Filter bank coefficients (M x T).
    use_polyphase (bool): Whether to use polyphase implementation.
    
    Returns:
    torch.Tensor: Synthesized signal.
    """
    num_subbands = filter_bank.shape[0]
    
    if use_polyphase:
        filter_bank = filter_bank.flip(-1)
        filter_bank = rearrange(filter_bank, "c (t m) -> m c t", m=num_subbands)
    
    padding = filter_bank.shape[-1] // 2 + 1
    synthesized_signal = nn.functional.conv1d(input_signal, filter_bank, padding=int(padding))[..., :-1] * num_subbands
    
    synthesized_signal = synthesized_signal.flip(1)
    synthesized_signal = rearrange(synthesized_signal, "b (c m) t -> b c (t m)", m=num_subbands)
    synthesized_signal = synthesized_signal[..., 2 * filter_bank.shape[1]:]
    
    return synthesized_signal

# Main PQMF Class

class PQMF(nn.Module):
    """
    Pseudo Quadrature Mirror Filter (PQMF) for multiband decomposition and reconstruction.
    
    Args:
    attenuation (int): Attenuation of the rejected bands in dB (typically 80-120).
    num_subbands (int): Number of subbands (must be a power of 2 for polyphase implementation).
    use_polyphase (bool): Whether to use the polyphase algorithm for faster computation.
    """
    
    def __init__(self, attenuation, num_subbands, use_polyphase=True):
        super().__init__()
        
        if use_polyphase:
            assert math.log2(num_subbands).is_integer(), "For polyphase algorithm, num_subbands must be a power of 2"
        
        prototype_filter = torch.from_numpy(optimize_prototype_filter(attenuation, num_subbands)).float()
        filter_bank = create_cosine_modulated_filterbank(prototype_filter, num_subbands)
        filter_bank = pad_to_next_power_of_two(filter_bank)
        
        self.register_buffer("filter_bank", filter_bank)
        self.register_buffer("prototype_filter", prototype_filter)
        self.num_subbands = num_subbands
        self.use_polyphase = use_polyphase
    
    def forward(self, input_signal):
        """
        Decomposes the input signal into subbands.
        
        Args:
        input_signal (torch.Tensor): Input signal to decompose.
        
        Returns:
        torch.Tensor: Decomposed signal.
        """
        if self.num_subbands == 1:
            return input_signal
        
        if self.use_polyphase:
            decomposed_signal = pqmf_analysis(input_signal, self.filter_bank)
        else:
            decomposed_signal = nn.functional.conv1d(
                input_signal,
                self.filter_bank.unsqueeze(1),
                stride=self.num_subbands,
                padding=self.filter_bank.shape[-1] // 2,
            )[..., :-1]
        
        return reverse_alternate_subbands(decomposed_signal)
    
    def inverse(self, input_signal):
        """
        Reconstructs the original signal from subbands.
        
        Args:
        input_signal (torch.Tensor): Input signal in subbands to reconstruct.
        
        Returns:
        torch.Tensor: Reconstructed signal.
        """
        if self.num_subbands == 1:
            return input_signal
        
        input_signal = reverse_alternate_subbands(input_signal)
        
        if self.use_polyphase:
            return pqmf_synthesis(input_signal, self.filter_bank)
        else:
            self.filter_bank = self.filter_bank.flip(-1)
            temp_signal = torch.zeros(*input_signal.shape[:2], self.num_subbands * input_signal.shape[-1]).to(input_signal)
            temp_signal[..., ::self.num_subbands] = input_signal * self.num_subbands
            reconstructed_signal = nn.functional.conv1d(
                temp_signal,
                self.filter_bank.unsqueeze(0),
                padding=self.filter_bank.shape[-1] // 2,
            )[..., 1:]
            return reconstructed_signal