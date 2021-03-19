from typing import List, Dict

import torch
from speech_interface.interfaces import Interface


PARAMS = {
    'audio': {
        'sample_rate': 22050,
        'n_fft': 1024,
        'window_size': 1024,
        'hop_size': 256,
        'num_mels': 80,
        'fmin': 0.,
        'fmax': None,
        'is_center': False
    },
    'models': {
        'mel_gan_lj': '',
        'mel_gan_multi': ''
    }
}


class InterfaceMelGAN(Interface):

    def __init__(self, model_name: str = 'mel_gan_multi', device='cpu'):
        super().__init__()
        assert model_name in PARAMS['models'], \
            'Model name {} is not valid! choose in {}'.format(
                model_name, str(PARAMS['models'].keys()))

        model_name_mapping = {
            'mel_gan_lj': 'linda_johnson',
            'mel_gan_multi': 'multi_speaker'
        }

        self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan',
                                      model_name=model_name_mapping[model_name])

        self.device = device
        self.vocoder.fft = self.vocoder.fft.to(device)
        self.vocoder.mel2wav = self.vocoder.mel2wav.to(device)
        # print params
        print('Total Model {} params.'.format(self.num_params(self.vocoder.mel2wav)))

    def num_params(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    def encode(self, wav_tensor):
        """
        Convert raw waveform into mel-spectrogram.
        :param wav_tensor: raw waveform tensor. (N, 1, Tw)
        :return: mel-spectrogram tensor. (N, Cm, Tm)
        """
        if wav_tensor.ndim == 2:
            wav_tensor = wav_tensor.unsqueeze(1)
        return self.vocoder.fft(wav_tensor.to(self.device))

    @torch.no_grad()
    def decode(self, mel_spectrogram):
        """
        Reconstruct raw waveform from mel-spectrogram.
        :param mel_tensor: mel-spectrogram tensor. (N, Cm, Tm)
        :return: raw waveform tensor. (N, Tw)
        """
        return self.vocoder.mel2wav(mel_spectrogram).squeeze(1)
