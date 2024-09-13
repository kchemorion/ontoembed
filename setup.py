# setup.py

from setuptools import setup, find_packages

setup(
    name='ontoembed',
    version='0.1.0',
    description='A tool for converting ontologies into vector embeddings',
    author='Your Name',
    author_email='kchemorion@gmail.com',
    url='https://github.com/kchemorion/ontoembed',
    packages=find_packages(),
    install_requires=[
        'rdflib>=5.0.0',
        'numpy>=1.18.0',
        'gensim>=4.0.0',
        'networkx>=2.4',
    ],
    entry_points={
        'console_scripts': [
            'ontoembed=ontoembed.cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.6',
)
