from setuptools import setup
from setuptools import find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
	long_description = fh.read()

setup(
    name='ggrn_backend2',
    version='0.0.1',
    description='A wrapper connecting the Grammar of Gene Regulatory Networks to DCD-FG.',
    long_description=long_description,
	long_description_content_type='text/markdown',
    #url
    author='Eric Kernfeld',
    author_email='eric.kern13@gmail.com',
    packages=find_packages(),
    install_requires=[
        'torch', 'pytorch-lightning','numpy','scikit-learn','pandas','anndata'
    ],
    python_requires=">=3.7", 
    url='https://github.com/ekernf01/ggrn_backend2',
)