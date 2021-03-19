import torch
from typing import List, Dict
from speech_interface.encoders import MelSpectrogram
from speech_interface.interfaces import Interface
from speech_interface.models import hifi_gan
from speech_interface.utils import download_and_get_chkpath


# Hyperparameters for Hifi-GAN
PARAMS = {
    'audio': {
        'sample_rate': 22050,
        'n_fft': 1024,
        'window_size': 1024,
        'hop_size': 256,
        'num_mels': 80,
        'fmin': 0.,
        'fmax': 8000.,
        'is_center': False
    },
    'models': {
        'hifi_gan_v1_universal': 'https://drive.google.com/uc?id=1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW',
        'hifi_gan_v1_vctk': 'https://drive.google.com/uc?id=1RkZ8reW0WjR9lE_ztTnN1qFVx24JZhhy',
        'hifi_gan_v1_lj': 'https://drive.google.com/uc?id=14NENd4equCBLyyCSke114Mv6YR_j_uFs',
        'hifi_gan_v2_vctk': 'https://drive.google.com/uc?id=1CEUIeAupaUrBUSJCtP3PV4Zg7W7GQUES',
        'hifi_gan_v2_lj': 'https://drive.google.com/uc?id=1gfouaWecMbmfqIdWYs-KtsULIdYCveYW',
        'hifi_gan_v3_vctk': 'https://drive.google.com/uc?id=1wDJD4YwEVOvvfgKd6ayZYJXJx5I7mahR',
        'hifi_gan_v3_lj': 'https://drive.google.com/uc?id=18TNnHbr4IlduAWdLrKcZrqmbfPOed1pS'
    },
    'is_gdrive': True
}


class InterfaceHifiGAN(Interface):
    """
    An interface between raw waveform and feature based with Hifi-GAN.
    Examples::
        import librosa
        import torch

        # Make an interface
        model_name = 'hifi_gan_v1_universal'
        device = 'cuda'
        interface = InterfaceHifiGAN(model_name=model_name, device=device)

        wav, sr = librosa.load('/your/wav/form/file/path')

        # to pytorch tensor
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)  # (1, Tw)

        # encode waveform tensor
        features = interface.encode(wav_tensor)

        # your speech synthesis process ...
        # ...

        # reconstruct waveform
        pred_wav_tensor = interface.decode(features)
    """

    def __init__(self, model_name: str = 'hifi_gan_v1_universal', device='cpu'):
        assert model_name in PARAMS['models'], \
            'Model name {} is not valid! choose in {}'.format(
                model_name, str(PARAMS['models'].keys()))

        # encoder
        self.encoder = MelSpectrogram(**PARAMS['audio']).to(device)

        # decoder
        model_config = getattr(hifi_gan, '_'.join(model_name.split('_')[:-1]))
        self.decoder = hifi_gan.Generator(**model_config())

        # load pretrained chkpt
        self.load_pretrained_chkpt('hifi_gan', model_name)

    def load_pretrained_chkpt(self, vocoder_name: str, model_name: str):
        chkpt_path = download_and_get_chkpath(
            vocoder_name, model_name, PARAMS['models'][model_name], is_gdrive=PARAMS['is_gdrive']
        )
        chkpt = torch.load(chkpt_path)
        self.decoder.load_state_dict(chkpt['generator'])
        self.decoder.remove_weight_norm()

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
        return self.encoder(wav_tensor, is_pad=True)

    @torch.no_grad()
    def decode(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct raw waveform from mel-spectrogram.
        :param mel_tensor: mel-spectrogram tensor. (N, Cm, Tm)
        :return: raw waveform tensor. (N, Tw)
        """
        assert mel_tensor.ndim == 3, '3D tensor (N, C, T) is needed'
        return self.decoder(mel_tensor)


if __name__ == '__main__':
    import librosa
    import torch

    # Make an interface
    model_name = 'hifi_gan_v1_universal'
    device = 'cuda'
    interface = InterfaceHifiGAN(model_name=model_name, device=device)

    wav, sr = librosa.load('/your/wav/form/file/path')

    # to pytorch tensor
    wav_tensor = torch.from_numpy(wav).unsqueeze(0)  # (1, Tw)

    # encode waveform tensor
    features = interface.encode(wav_tensor)

    # your speech synthesis process ...
    # ...

    # reconstruct waveform
    pred_wav_tensor = interface.decode(features)
