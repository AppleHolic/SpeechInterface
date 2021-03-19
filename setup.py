from setuptools import setup, find_packages
from typing import List


def get_requirements(filename: str = 'requirements.txt') -> List[str]:
    with open(filename, 'r') as r:
        return [x.strip() for x in r.readlines()]


setup(
    name='speech_interface',
    version='0.0.1',
    license='MIT',
    author='ILJI CHOI',
    author_email='choiilji@gmail.com',
    description='An interface for neural speech synthesis with Pytorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AppleHolic/SpeechInterface',
    keywords='speech',
    packages=find_packages(),
    install_requires=get_requirements(),
    python_requires='>=3.6',
    classifiers=[
        # 패키지에 대한 태그
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ]
)
