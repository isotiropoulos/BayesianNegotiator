from setuptools import setup, find_packages
from os import path

working_directory = path.abspath(path.dirname(__file__))

with open(path.join(working_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='bambon', # name of packe which will be package dir below project
    url='https://www.singularlogic.eu/',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    zip_safe=False
)
