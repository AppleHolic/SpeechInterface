import librosa
import torch
import torch.nn as nn
from typing import Tuple


class MelSpectrogram(nn.Module):
    """
    First mel-spectrogram module for neural speech synthesis. It will be a encoder for Hifi-GAN and WaveGlow.
    """
    def __init__(self, sample_rate: int = 22050, n_fft: int = 1024, window_size: int = 1024, hop_size: int = 256,
                 num_mels: int = 80, fmin: float = 0., fmax: float = 8000., is_center: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.window_size = window_size
        self.pad_size = (self.n_fft - self.hop_size) // 2
        self.is_center = is_center

        mel_filter_tensor = torch.FloatTensor(librosa.filters.mel(sample_rate, n_fft, num_mels, fmin, fmax))
        self.register_buffer('mel_filter', mel_filter_tensor)
        self.register_buffer('window', torch.hann_window(window_size))

    def stft(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # stft
        stft = torch.stft(wav, self.n_fft, hop_length=self.hop_size, win_length=self.window_size,
                          window=self.window, center=self.is_center, pad_mode='reflect',
                          normalized=False, onesided=True)
        real_part, img_part = [x.squeeze(3) for x in stft.chunk(2, 3)]
        return torch.sqrt(real_part ** 2 + img_part ** 2 + 1e-9), torch.atan2(img_part, real_part)

    def istft(self, mag: torch.Tensor, phase: torch.Tensor, is_center: bool = True) -> torch.Tensor:
        magnitude, phase = mag.unsqueeze(3), phase.unsqueeze(3)
        stft = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=3)
        return torch.istft(stft, self.n_fft, hop_length=self.hop_size, win_length=self.window_size,
                           window=self.window, center=is_center, normalized=False, onesided=True)

    def to_mel(self, mag: torch.Tensor, eps: float = 1e-5, norm_ratio: float = 1.) -> torch.Tensor:
        # mel
        spec = torch.matmul(self.mel_filter, mag)

        # spectral normalize
        spec = torch.log(torch.clamp(spec, min=eps) * norm_ratio)

        return spec

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, is_pad: bool = False) -> torch.Tensor:
        """
        Convert raw waveform to log-mel spectrogram.
        :param wav: raw waveform tensor (N, Tw)
        :param is_pad: Pad or not before calculating stft function.
        :return: log-mel spectrogram
        """
        if is_pad:
            pads = ((self.n_fft - self.hop_size) // 2, (self.n_fft - self.hop_size) // 2)
            wav = torch.nn.functional.pad(wav.unsqueeze(1), pads, mode='reflect').squeeze(1)
        return self.to_mel(self.stft(wav)[0])
