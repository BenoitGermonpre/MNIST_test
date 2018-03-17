from setuptools import find_packages
from setuptools import setup

setup(
    name='mlengine-boilerplate',
    version='0.1',
    author='Benoit Germonpre',
    author_email='benoitgermonpre@gmail.com',
    install_requires=['tensorflow==1.6.0'],
    packages=find_packages(exclude=['data', 'predictions']),
    description='ML Engine boilerplate code',
    url='https://github.com/BenoitGermonpre/MNIST_test'
)