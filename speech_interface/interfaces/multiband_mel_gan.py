import torch
from typing import List, Dict
from speech_interface.encoders import MelSpectrogram
from speech_interface.interfaces import Interface
from speech_interface.interfaces.stats import MULTI_BAND_MEL_GAN_PARAMS
from parallel_wavegan.utils import download_pretrained_model
from parallel_wavegan.utils import load_model


PARAMS = {
    'audio': {
        'multiband_mel_gan_lj': {
            'sample_rate': 22050,
            'n_fft': 1024,
            'window_size': 1024,
            'hop_size': 256,
            'num_mels': 80,
            'fmin': 80.,
            'fmax': 7600.,
            'is_center': True
        },
        'multiband_mel_gan_vctk': {
            'sample_rate': 24000,
            'n_fft': 2048,
            'hop_size': 300,
            'window_size': 1200,
            'num_mels': 80,
            'fmin': 80.,
            'fmax': 7600.,
            'is_center': True
        }
    },
    'models': {
        'multiband_mel_gan_lj': 'ljspeech_multi_band_melgan.v2',
        'multiband_mel_gan_vctk': 'vctk_multi_band_melgan.v2'
    }
}


class InterfaceMultibandMelGAN(Interface):

    def __init__(self, model_name: str = 'multiband_mel_gan_vctk', device='cpu'):
        super().__init__()
        assert model_name in PARAMS['models'], \
            'Model name {} is not valid! choose in {}'.format(
                model_name, str(PARAMS['models'].keys()))

        model_name_mapping = PARAMS['models'][model_name]

        self.device = device
        self.encoder = MelSpectrogram(**PARAMS['audio'][model_name])
        self.vocoder = load_model(download_pretrained_model(model_name_mapping)).to(device).eval()
        self.vocoder.remove_weight_norm()

        # make stat tensors
        param_key = 'vctk' if 'vctk' in model_name else 'lj'
        stats = MULTI_BAND_MEL_GAN_PARAMS[param_key]
        self.mean = torch.FloatTensor(stats['mean']).unsqueeze(0).unsqueeze(-1)
        self.scale = torch.FloatTensor(stats['scale']).unsqueeze(0).unsqueeze(-1)

        # print params
        print('Total Model {} params.'.format(self.num_params(self.vocoder)))

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
        :param wav_tensor: raw waveform tensor. (N, Tw)
        :return: mel-spectrogram tensor. (N, Cm, Tm)
        """
        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        # magnitude
        mag, _ = self.encoder.stft(wav_tensor)
        # mel
        mel = self.encoder.to_mel(mag)
        # to log 10 and normalize
        normlogmel = (torch.log10(mel) - self.mean) / self.scale

        return normlogmel.to(self.device)

    @torch.no_grad()
    def decode(self, mel_spectrogram):
        """
        Reconstruct raw waveform from mel-spectrogram.
        :param mel_tensor: mel-spectrogram tensor. (N, Cm, Tm)
        :return: raw waveform tensor. (N, Tw)
        """
        subbands = self.vocoder(mel_spectrogram)
        pred = self.vocoder.pqmf.synthesis(subbands)
        return pred.squeeze(1)
