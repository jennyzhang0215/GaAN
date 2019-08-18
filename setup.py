import os
import setuptools

setuptools.setup(
    name='GaAN',
    version="0.1.dev0",
    author="Jiani Zhang, Xingjian Shi",
    author_email="jnzhang@cuhk.edu.hk, xshiab@cse.ust.hk",
    packages=setuptools.find_packages(),
    description='GluonGraph',
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.md')).read(),
    license='MIT',
    url='https://github.com/sxjscience/MXGraph',
    install_requires=['numpy', 'scipy', 'matplotlib', 'six', 'pyyaml', 'networkx', 'sklearn', 'pandas'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)