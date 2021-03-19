# SpeechInterface

A Speech Interface Toolkit for Neural Speech Synthesis with Pytorch

This repository is made for deploying your neural speech synthesis experiments efficiently. 
The main feature is defined as:

> - Matching audio feature parameters and their source codes for using major neural vocoders
>
> - They called an interface, which has encode and decode function.
>
>   - Encode: Convert raw waveform to audio features. (e.g. mel-spectrogram, mfcc ...)
>
>   - Decode: Reconstruct audio features to raw waveform. (i.e. neural vocoder)  
>

- Usage Examples
  - Compare experimental results of neural vocoder with others
  - Use directly audio features and neural vocoders for neural speech synthesis models


## Available neural vocoders

1. Hifi-GAN (Universal v1, VCTK, LJSpeech)
2. WaveGlow, MelGAN (WIP)
3. Multi-band MelGAN, Parallel WaveGAN (WIP)


## Example

- Use an interface

```python
import librosa
import torch
from speech_interface.interfaces.hifi_gan import InterfaceHifiGAN

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
```

- Checkout available models and audio parameters

```python
from speech_interface.interfaces.hifi_gan import InterfaceHifiGAN

# available models
print(InterfaceHifiGAN.available_models())

# audio parameters
print(InterfaceHifiGAN.audio_params())
```

## Reference

- Hifi-GAN : https://github.com/jik876/hifi-gan


## License

This repository is under MIT license.
