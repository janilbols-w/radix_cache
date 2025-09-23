from setuptools import setup, find_packages

setup(
    name='radix_cache',
    version='0.1.0',
    package_dir={'': 'src'},
    # packages=find_packages(),
    install_requires=[ 
        'torch',
        ],
    author='janilbols',
    author_email='janilbols@outlook.com',
    description='naive radix tree for kvcache',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
