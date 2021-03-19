# These line is inserted for importing glow.py source.
import sys
import os
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

import torch
from typing import List, Dict
from speech_interface.encoders import MelSpectrogram
from speech_interface.interfaces import Interface
from speech_interface.utils import download_and_get_chkpath


# Hyperparameters for waveglow
PARAMS = {
    'audio': {
        'sample_rate': 22050,
        'n_fft': 1024,
        'window_size': 1024,
        'hop_size': 256,
        'num_mels': 80,
        'fmin': 0.,
        'fmax': 8000.,
        'is_center': True
    },
    'models': {
        # universal model has an import error with glow code.
        # 'waveglow_universal': 'https://drive.google.com/uc?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF',
        'waveglow_lj': 'https://drive.google.com/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx',
    },
    'is_gdrive': True
}


class InterfaceWaveGlow(Interface):
    """
    An interface between raw waveform and feature based with WaveGlow.
    """

    def __init__(self, model_name: str = 'hifi_gan_v1_universal', device='cuda'):
        assert model_name in PARAMS['models'], \
            'Model name {} is not valid! choose in {}'.format(
                model_name, str(PARAMS['models'].keys()))
        self.device = device

        # encoder
        self.encoder = MelSpectrogram(**PARAMS['audio']).to(device)

        # decoder
        self.waveglow = self.load_model('waveglow', model_name)

    def load_model(self, vocoder_name: str, model_name: str):
        chkpt_path = download_and_get_chkpath(
            vocoder_name, model_name, PARAMS['models'][model_name], is_gdrive=PARAMS['is_gdrive']
        )
        waveglow = torch.load(chkpt_path)['model']
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow.to(self.device).eval()
        return waveglow

    @staticmethod
    def available_models() -> List[str]:
        """
        :return: available model names
        """
        return list(PARAMS['models'].keys())

    @staticmethod
    def audio_params() -> Dict:
        return PARAMS['audio']

    @torch.no_grad()
    def encode(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert raw waveform into mel-spectrogram.
        :param wav_tensor: raw waveform tensor. (N, Tw)
        :return: mel-spectrogram tensor. (N, Cm, Tm)
        """
        assert wav_tensor.ndim == 2, '2D tensor (N, T) is needed'
        return self.encoder(wav_tensor)

    @torch.no_grad()
    def decode(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct raw waveform from mel-spectrogram.
        :param mel_tensor: mel-spectrogram tensor. (N, Cm, Tm)
        :return: raw waveform tensor. (N, Tw)
        """
        assert mel_tensor.ndim == 3, '3D tensor (N, C, T) is needed'
        return self.waveglow.infer(mel_tensor, sigma=1.)
