from setuptools import find_packages
from setuptools import setup

setup(
    name='mlengine-mnist',
    version='0.1',
    author='Benoit Germonpre',
    author_email='benoitgermonpre@gmail.com',
    install_requires=['tensorflow==1.4.0', 'numpy==1.14.0'],
    packages=find_packages(),
    scripts=[],
    package_data={
        'trainer': ['*'],  # include any none python files in trainer
    },
    description='ML Engine MNIST code',
    url='https://github.com/BenoitGermonpre/MNIST_test'
)