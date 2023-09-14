
from setuptools import setup, find_packages

# Define package metadata
NAME = 'OpenAPS Cleaner'
VERSION = '0.1.0'
DESCRIPTION = 'Converts OpenAPS device logs into time-series analysis compatible form.'
AUTHOR = 'Harry Emerson'
EMAIL = 'harry.emerson@bristol.ac.uk'
URL = 'https://github.com/yourusername/OpenAPS_Cleaner'

# Specify package dependencies
INSTALL_REQUIRES = [
    # List your dependencies here, e.g., 'numpy>=1.0.0'
]

# Define long description (usually from a README file)
with open('README.md', 'r', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
