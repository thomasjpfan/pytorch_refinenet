from setuptools import setup, find_packages


requirements = [
    'torch',
    'torchvision'
]

setup(
    name='pytorch_refinenet',
    version='0.2.4',
    author='Thomas Fan',
    author_email='thomasjpfan@gmail.com',
    url='https://github.com/thomasjpfan/pytorch_refinenet',
    description='Pytorch Implementation of RefineNet',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements
)
