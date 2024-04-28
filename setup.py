from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='EvoCluster',
    version='0.1.0',
    description='Evolutionary clustering algorithm package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='joshuakattapuram10@gmail.com',
    url='https://github.com/joshkatt/EvoCluster',
    download_url='https://github.com/yourusername/EvoCluster/1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
)
