from setuptools import setup
# reads in the requirements.txt
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(name='dog_classifier',
      install_requires=install_requires,
      version='1.0',
      author="Steven Becker, Felix Geyer",
      packages=['dog_classifier'])
