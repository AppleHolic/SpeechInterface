import abc


class Interface:
    """
    Defines the interface between 'wav' and 'model'
    """

    @abc.abstractmethod
    def load_pretrained_chkpt(self, vocoder_name: str, model_name: str):
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def available_models():
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def audio_params():
        raise NotImplementedError()

    @abc.abstractmethod
    def encode(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def decode(self, *args, **kwargs):
        raise NotImplementedError()
