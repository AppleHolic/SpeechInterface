import os
import gdown
import requests


def get_cache_dir() -> str:
    """
    :return: master cache directory
    """
    home = os.path.expanduser("~")
    cache_dir = os.path.join(home, '.cache', 'speech_interface')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_chkpt_path(vocoder_name: str, model_name: str):
    """
    File path for storing checkpoints.
    :param vocoder_name: the name of vocoder
    :param model_name: the sub-name of vocoder
    :return: stored file path of a checkpoint
    """
    # get cache dir
    master_dir = get_cache_dir()
    # make dir
    sub_dir = os.path.join(master_dir, vocoder_name)
    os.makedirs(sub_dir, exist_ok=True)
    # chkpt path
    chkpt_path = os.path.join(sub_dir, f'{model_name}.chkpt')
    return chkpt_path


def __download_request(url: str, out_path: str):
    with requests.get(url) as r:
        with open(out_path, 'w') as w:
            w.write(r.content)


def download_and_get_chkpath(vocoder_name: str, model_name: str, url: str, is_gdrive: bool = False) -> str:
    """
    Check given vocoder_name and model_name, and if checkpoint is not already downloaded,
    this function downloads to checkpoint path.
    Finally it returns the stored file path of checkpoint by given arguments.
    :param vocoder_name: the name of vocoder
    :param model_name: the sub-name of vocoder
    :param url: url for downloading checkpoint file, it will be an url of google drive or general internet url.
    :param is_gdrive: check given url is an url of google drive
    :return: checkpoint file path
    """
    # get chkpt path
    chkpt_path = get_chkpt_path(vocoder_name, model_name)

    # check file exists
    if os.path.exists(chkpt_path):
        print(f'Use cached file of {model_name}')
    else:
        if is_gdrive:
            gdown.download(url, output=chkpt_path, quiet=False)
        else:
            __download_request(url, chkpt_path)
    return chkpt_path
