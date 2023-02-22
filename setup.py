from setuptools import find_packages, setup

config = {
    'name': 'MLPlatform',
    'version': '0.1.0',
    'description': 'Data exploration, cleaning, and modelling tool',
    'packages': find_packages(),
}

setup(**config)