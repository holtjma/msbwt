from setuptools import setup
from MUS import CommandLineInterface

setup(name='msbwt',
      #version='0.1.3',
      version=CommandLineInterface.VERSION,
      description='Allows for merging and querying of multi-string BWTs for genomic strings',
      url='http://code.google.com/p/suspenders',
      author='James Holt',
      author_email='holtjma@cs.unc.edu',
      license='MIT',
      packages=['MUS'],
      install_requires=['pysam'],
      scripts=['bin/msbwt'],
      zip_safe=False)
